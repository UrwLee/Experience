#include "caffe/layers/connect_limb_layer.hpp"
#include <vector>
#include <map>
#include <stdexcept>

namespace caffe {

// 按照PA得分进行排序
template <typename Dtype>
bool ConnectDescend(const std::vector<Dtype>& lhs,
                    const std::vector<Dtype>& rhs) {
    return lhs[2] > rhs[2];
}

template <typename Dtype>
void ConnectlimbLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const ConnectLimbParameter& connect_limb_param = this->layer_param_.connect_limb_param();
	// 读入估计类型
	type_coco_ = connect_limb_param.is_type_coco();
	// 设置参数
	if (type_coco_) {
		part_name_ = get_coco_part_name();
		limb_seq_ = get_coco_limb_seq();
		limb_channel_id_ = get_coco_limb_channel_id();
		num_parts_ = get_coco_num_parts();
		num_limbs_ = get_coco_num_limbs();
		CHECK_EQ(2*num_limbs_, limb_seq_.size());
		CHECK_EQ(limb_seq_.size(), limb_channel_id_.size());
		for (int i = 0; i < limb_seq_.size() / 2; ++i) {
			const int la = limb_seq_[2*i];
			const int lb = limb_seq_[2*i + 1];
			const int mx = limb_channel_id_[2*i];
			const int my = limb_channel_id_[2*i + 1];
			part_name_[mx] = part_name_[la] + "->" + part_name_[lb] + "(X)";
			part_name_[my] = part_name_[la] + "->" + part_name_[lb] + "(Y)";
		}
	} else {
    LOG(FATAL) << "Error - only coco type is supported.";
	}
  max_peaks_use_ = connect_limb_param.max_peaks_use();
	max_persons_ = connect_limb_param.max_person();
	iters_pa_cal_ = connect_limb_param.iters_pa_cal();
	connect_inter_threshold_ = connect_limb_param.connect_inter_threshold();
	connect_inter_min_nums_ = connect_limb_param.connect_inter_min_nums();
	connect_min_subset_cnt_ = connect_limb_param.connect_min_subset_cnt();
	connect_min_subset_score_ = connect_limb_param.connect_min_subset_score();
}

template <typename Dtype>
void ConnectlimbLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	std::vector<int> bottom0_shape = bottom[0]->shape();
	std::vector<int> bottom1_shape = bottom[1]->shape();
	CHECK_EQ(1, bottom0_shape[0]);
	CHECK_EQ(num_parts_+2*num_limbs_, bottom0_shape[1]);
	CHECK_EQ(1, bottom1_shape[0]);
	CHECK_EQ(num_parts_, bottom1_shape[1]);
	max_peaks_ = bottom1_shape[2] - 1;
	CHECK_EQ(3, bottom1_shape[3]);
  // last 3 param: count, SUM_score, avg_score
  top[0]->Reshape(1,1,num_parts_+1,3);
}

template <typename Dtype>
void ConnectlimbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* heatmap_ptr = bottom[0]->cpu_data();
	const Dtype* peaks_ptr = bottom[1]->cpu_data();
  Dtype* peaks_mptr = bottom[1]->mutable_cpu_data();

	const int width = bottom[0]->width();
	const int height = bottom[0]->height();
	const int map_offset = width * height;
	const int peaks_offset = 3 * (max_peaks_+1);

	vector<vector<Dtype> > subset;
	subset.clear();

	// 0~num_parts_: {parts}
	// num_parts_+1: score
	// num_parts_+2: count
	const int SUBSET_CNT = num_parts_+2;
	const int SUBSET_SCORE = num_parts_+1;
	const int SUBSET_SIZE = num_parts_+3;

  // 处理所有peaks, 按照置信度排列,取最高的若干个进行排序
  for (int i = 0; i < num_parts_; ++i) {
    const Dtype* cand = peaks_ptr + i * peaks_offset;
    int num_peaks = cand[0] > max_peaks_ ? max_peaks_ : cand[0];
    vector<vector<Dtype> > peak_vec;
    for (int j = 1; j <= num_peaks; ++j) {
      vector<Dtype> peak;
      peak.push_back(cand[j * 3] * width);
      peak.push_back(cand[j * 3 + 1] * height);
      peak.push_back(cand[j * 3 + 2]);
      peak_vec.push_back(peak);
    }
    if (peak_vec.size() > 0) {
      std::stable_sort(peak_vec.begin(), peak_vec.end(), ConnectDescend<Dtype>);
      int use_peaks = peak_vec.size() > max_peaks_use_ ? max_peaks_use_ : peak_vec.size();
      peak_vec.resize(use_peaks);
      // 重新写入
      Dtype* mcand = peaks_mptr + i * peaks_offset;
      mcand[0] = use_peaks;
      for (int k = 1; k <= use_peaks; ++k) {
        mcand[k * 3] = peak_vec[k-1][0];
        mcand[k * 3 + 1] = peak_vec[k-1][1];
        mcand[k * 3 + 2] = peak_vec[k-1][2];
      }
    }
  }

	// 依次处理num_limbs_个片段
	for(int k = 0; k < num_limbs_; k++) {
		// 指定limb两个关节点的PA-map
		const Dtype* map_x = heatmap_ptr + limb_channel_id_[2*k] * map_offset;
		const Dtype* map_y = heatmap_ptr + limb_channel_id_[2*k+1] * map_offset;
		// 指定limb两个关节点的peaks-map
		const Dtype* candA = peaks_ptr + limb_seq_[2*k] * peaks_offset;
		const Dtype* candB = peaks_ptr + limb_seq_[2*k+1] * peaks_offset;

		// 第k类limb的连接
		vector<vector<Dtype> > connection_k;
		// 两个节点的peaks数
		const int nA = candA[0] > max_peaks_ ? max_peaks_ : candA[0];
		const int nB = candB[0] > max_peaks_ ? max_peaks_ : candB[0];
		if (nA == 0 && nB == 0) {
			continue;
		}	else if (nA == 0) {
			for(int i = 1; i <= nB; i++) {
				bool found = false;
				const int indexB = limb_seq_[2*k+1];
				for(int j = 0; j < subset.size(); j++) {
					int offs = limb_seq_[2*k+1] * peaks_offset + i * 3 + 2;
					if (int(subset[j][indexB]) == offs) {
						found = true;
						break;
					}
				}
				if (!found) {
					vector<Dtype> row_vec(SUBSET_SIZE, 0);
				  row_vec[limb_seq_[2*k+1]] = limb_seq_[2*k+1] * peaks_offset + i * 3 + 2;
				  row_vec[SUBSET_CNT] = 1;
				  row_vec[SUBSET_SCORE] = candB[i*3+2];
				  subset.push_back(row_vec);
				}
			}
			continue;
		} else if (nB == 0) {
			for(int i = 1; i <= nA; i++) {
				bool found = false;
				const int indexA = limb_seq_[2*k];
				for(int j = 0; j < subset.size(); j++) {
					int offs = limb_seq_[2*k] * peaks_offset + i * 3 + 2;
					if (int(subset[j][indexA]) == offs) {
						found = true;
						break;
					}
				}
				if (!found) {
					vector<Dtype> row_vec(SUBSET_SIZE, 0);
				  row_vec[limb_seq_[2*k]] = limb_seq_[2*k] * peaks_offset + i * 3 + 2;
				  row_vec[SUBSET_CNT] = 1;
				  row_vec[SUBSET_SCORE] = candA[i*3+2];
				  subset.push_back(row_vec);
				}
			}
			continue;
		}

		// 进入正常处理流程
		vector<vector<Dtype> > temp;
		for(int i = 1; i <= nA; i++) {
	 		for(int j = 1; j <= nB; j++) {
			 Dtype s_x = candA[i*3];
			 Dtype s_y = candA[i*3+1];
			 Dtype d_x = candB[j*3] - candA[i*3];
			 Dtype d_y = candB[j*3+1] - candA[i*3+1];
			 Dtype norm_vec = sqrt(d_x*d_x + d_y*d_y);
			 if (norm_vec < 1e-4) {
					 continue;
			 }
			 // 获取归一化方向矢量
			 Dtype vec_x = d_x / norm_vec;
			 Dtype vec_y = d_y / norm_vec;

			 Dtype sum = 0;
			 int count = 0;

			 // 在方向矢量上积分
			 for(int lm = 0; lm < iters_pa_cal_; ++lm) {
					 int my = round(s_y + lm * d_y / iters_pa_cal_);
					 int mx = round(s_x + lm * d_x / iters_pa_cal_);
					 // 计算该点的PA矢量与方向矢量的内积
					 // 该值越大,表明是肢体的可能越大 (PA代表了肢体方向的可能性)
					 int idx = my * width + mx;
					 Dtype score = (vec_x * map_x[idx] + vec_y * map_y[idx]);
					 if (score > connect_inter_threshold_) {
							 sum = sum + score;
							 count ++;
					 }
			 }
			 // count滤波,必须超过给定限值
			 if (count > connect_inter_min_nums_) {
					 // 创建一条连接记录
					 vector<Dtype> row_vec(4, 0);
					 row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
					 row_vec[2] = sum/count; //score PA
					 row_vec[0] = i;
					 row_vec[1] = j;
					 temp.push_back(row_vec);
			 }
	 		}
		}

		// 对连接结果排序
		if (temp.size() > 0) {
			std::stable_sort(temp.begin(), temp.end(), ConnectDescend<Dtype>);
		} else {
			continue;
		}
		// 获取最高的几条记录
		const int num_sel = std::min(nA, nB);
		int cnt = 0;
		vector<bool> has_conn_nA(nA, false);
		vector<bool> has_conn_nB(nB, false);
		for(int row =0; row < temp.size(); row++) {
			if (cnt >= num_sel) {
					break;
			}	else {
				int i = int(temp[row][0]);
				int j = int(temp[row][1]);
				float score = temp[row][2];
				if ((!has_conn_nA[i-1])  && (!has_conn_nB[j-1])) {
						vector<Dtype> row_vec(3, 0);
						row_vec[0] = limb_seq_[2*k] * peaks_offset + i * 3 + 2;
						row_vec[1] = limb_seq_[2*k+1] * peaks_offset + j * 3 + 2;
						row_vec[2] = score;
						connection_k.push_back(row_vec);
						cnt++;
						has_conn_nA[i-1] = true;
						has_conn_nB[j-1] = true;
				}
			}
		}

		// 第一次拼接, subset是空的, 先初始化
		if (k==0) {
		/**************************************************************/
		/**
		 * 每一行分别为:
		 * 0,1,2,3,4,..., num_parts-1, num_parts, num_parts+1, num_parts+2
		 * row_vec[num_parts+2] -> CNT, 关键点的数量
		 * row_vec[num_parts+1] -> score, 总得分
		 */
			vector<Dtype> row_vec(SUBSET_SIZE, 0);
			for(int i = 0; i < connection_k.size(); i++) {
					Dtype indexA = connection_k[i][0];
					Dtype indexB = connection_k[i][1];
					row_vec[limb_seq_[2*k]] = indexA;
					row_vec[limb_seq_[2*k+1]] = indexB;
					row_vec[SUBSET_CNT] = 2;
					row_vec[SUBSET_SCORE] = peaks_ptr[int(indexA)] + peaks_ptr[int(indexB)] + connection_k[i][2];
					subset.push_back(row_vec);
			}
		} else {
			if (connection_k.size() == 0) continue;
			for(int i = 0; i < connection_k.size(); i++) {
				int num = 0;
				const int indexA = connection_k[i][0];
				const int indexB = connection_k[i][1];
				// 继续拼接
				for(int j = 0; j < subset.size(); j++) {
						if (int(subset[j][limb_seq_[2*k]]) == indexA) {
							subset[j][limb_seq_[2*k+1]] = indexB;
							num++;
							subset[j][SUBSET_CNT]++;
							subset[j][SUBSET_SCORE] += (peaks_ptr[int(indexB)] + connection_k[i][2]);
						}
				}
				// 如果在现有连接子集没有找到A,说明是一个新的person对象,重新创建之
				if (num == 0) {
					std::vector<Dtype> row_vec(SUBSET_SIZE, 0);
					row_vec[limb_seq_[2*k]] = indexA;
					row_vec[limb_seq_[2*k+1]] = indexB;
					row_vec[SUBSET_CNT] = 2;
					row_vec[SUBSET_SCORE] = peaks_ptr[int(indexA)] + peaks_ptr[int(indexB)] + connection_k[i][2];
					subset.push_back(row_vec);
				}
			}
		}
	}
	// 所有片段全部连接
	// subset是所有的连接对象
	// 统计输出person数量
	int person_cnt = 0;
  vector<vector<Dtype> > person_vec;
	for(int i = 0; i < subset.size(); i++) {
		if (subset[i][SUBSET_CNT] >= connect_min_subset_cnt_) {
      float avg_score = 1.0 - 0.1 * (subset[i][SUBSET_CNT]-3);
      avg_score = avg_score < connect_min_subset_score_ ? connect_min_subset_score_ : avg_score;
      if ((subset[i][SUBSET_SCORE] / subset[i][SUBSET_CNT]) > avg_score) {
        vector<Dtype> person;
        person.push_back(i); //idx
        person.push_back(subset[i][SUBSET_CNT]); // cnt of parts
        person.push_back(subset[i][SUBSET_SCORE]); // score of person
        person_vec.push_back(person);
      }
	  }
  }
  if (person_vec.size() > 0) {
    std::stable_sort(person_vec.begin(), person_vec.end(), ConnectDescend<Dtype>);
    person_cnt = person_vec.size() > max_persons_ ? max_persons_ : person_vec.size();
    person_vec.resize(person_cnt);
    top[0]->Reshape(1,person_cnt,num_parts_+1,3);
    Dtype* joints_ptr = top[0]->mutable_cpu_data();
    for(int i = 0; i < person_cnt; i++) {
      int person_id = person_vec[i][0];
      for(int j = 0; j < num_parts_; j++) {
        int idx = int(subset[person_id][j]);
        if (idx) {
            joints_ptr[i * (num_parts_+1) * 3 + j * 3 + 2] = peaks_ptr[idx];
            joints_ptr[i * (num_parts_+1) * 3 + j * 3 + 1] = peaks_ptr[idx-1] / Dtype(height);
            joints_ptr[i * (num_parts_+1) * 3 + j * 3] = peaks_ptr[idx-2] / Dtype(width);
        } else {
            joints_ptr[i * (num_parts_+1) * 3 + j * 3 + 2] = 0;
            joints_ptr[i * (num_parts_+1) * 3 + j * 3 + 1] = 0;
            joints_ptr[i * (num_parts_+1) * 3 + j * 3] = 0;
        }
      }
      // cnt, sum_score, avg_score
      joints_ptr[i * (num_parts_+1) * 3 + num_parts_ * 3] = person_vec[i][1];
      joints_ptr[i * (num_parts_+1) * 3 + num_parts_ * 3 + 1] = person_vec[i][2];
      joints_ptr[i * (num_parts_+1) * 3 + num_parts_ * 3 + 2] = person_vec[i][2] / person_vec[i][1];
    }
  } else {
    top[0]->Reshape(1,1,num_parts_+1,3);
    caffe_set<Dtype>(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
  }

  /**
   * rewrite joints to normalized values
   */
   for (int i = 0; i < num_parts_; ++i) {
     const Dtype* cand = peaks_ptr + i * peaks_offset;
     int num_peaks = cand[0] > max_peaks_ ? max_peaks_ : cand[0];
     Dtype* mcand = peaks_mptr + i * peaks_offset;
     mcand[0] = num_peaks;
     for (int k = 1; k <= num_peaks; ++k) {
       mcand[k * 3] = cand[k * 3] / width;
       mcand[k * 3 + 1] = cand[k * 3 + 1] / height;
       mcand[k * 3 + 2] = cand[k * 3 + 2];
     }
   }
}

#ifdef CPU_ONLY
STUB_GPU(ConnectlimbLayer);
#endif

INSTANTIATE_CLASS(ConnectlimbLayer);
REGISTER_LAYER_CLASS(Connectlimb);

} // namespace caffe
