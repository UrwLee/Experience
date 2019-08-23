#include "caffe/layers/easy_match_layer.hpp"
#include <vector>
#include <map>
#include <cmath>

namespace caffe {

static int LOSS_MAX = 30;
// static float MAX_VAR_HW = 2;

template <typename Dtype>
void EasymatchLayer<Dtype>::get_similarity(person_t& person_pro, person_t& person_his, matchsts_t& sts) {
  // iou
  if (person_his.xmin > person_pro.xmax || person_his.xmax < person_pro.xmin ||
      person_his.ymin > person_pro.ymax || person_his.ymax < person_pro.ymin) {
    sts.iou = 0;
    sts.oks = 0;
    sts.similarity = 0;
  } else {
    Dtype intersec_xmin = std::max(person_pro.xmin, person_his.xmin);
    Dtype intersec_ymin = std::max(person_pro.ymin, person_his.ymin);
    Dtype intersec_xmax = std::min(person_pro.xmax, person_his.xmax);
    Dtype intersec_ymax = std::min(person_pro.ymax, person_his.ymax);
    Dtype intersec_width = intersec_xmax - intersec_xmin;
    Dtype intersec_height = intersec_ymax - intersec_ymin;
    if (intersec_width > 0 && intersec_height > 0) {
      Dtype  boxsize_pro = person_pro.width * person_pro.height;
      Dtype  boxsize_his = person_his.width * person_his.height;
      Dtype  intersec_size = intersec_width * intersec_height;
      sts.iou = intersec_size / (boxsize_pro + boxsize_his - intersec_size);
      // jiegouhua juli
      const int limb_coco[34] = {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17};
      int num_compare_limbs = 0;
      float sum_compare_limbs = 0;
      for (int l = 0; l < 17; ++l) {
        int part_a = limb_coco[2*l];
        int part_b = limb_coco[2*l+1];
        if ((person_pro.kps[part_a].v > 0.01) && (person_pro.kps[part_b].v > 0.01) &&
            (person_his.kps[part_a].v > 0.01) && (person_his.kps[part_b].v > 0.01)) {
          num_compare_limbs++;
          float dx_his = person_his.kps[part_a].x - person_his.kps[part_b].x;
          float dy_his = person_his.kps[part_a].y - person_his.kps[part_b].y;
          float dx_pro = person_pro.kps[part_a].x - person_pro.kps[part_b].x;
          float dy_pro = person_pro.kps[part_a].y - person_pro.kps[part_b].y;
          float dx = dx_pro - dx_his;
          float dy = dy_pro - dy_his;
          sum_compare_limbs += (dx*dx + dy*dy);
        }
      }
      // int num_vis = 0;
      // float sum = 0;
      // for (int k = 0; k < 18; ++k) {
      //   if ((person_pro.kps[k].v > 0.01) && (person_his.kps[k].v > 0.01)) {
      //     num_vis++;
      //     float dx = person_pro.kps[k].x - person_his.kps[k].x;
      //     float dy = person_pro.kps[k].y - person_his.kps[k].y;
      //     float e = (dx*dx+dy*dy)/2.0/vars_[k]/boxsize_his;
      //     sum += exp(-e);
      //   }
      // }
      if (num_compare_limbs >= 2) {
        sts.oks = sum_compare_limbs / num_compare_limbs;
        sts.similarity = sts.iou * (1+exp(-100*sts.oks));
      } else {
        sts.oks = 0;
        sts.similarity = sts.iou;
      }
    } else {
      sts.iou = 0;
      sts.oks = 0;
      sts.similarity = 0;
    }
  }
}

template <typename Dtype>
void EasymatchLayer<Dtype>::update_person(person_t& person_pro, person_t& person_his) {
  // supress the thrink
  // Dtype pro_width = person_pro.xmax - person_pro.xmin;
  // Dtype pro_height = person_pro.ymax - person_pro.ymin;
  // Dtype his_width = person_his.xmax - person_his.xmin;
  // Dtype his_height = person_his.ymax - person_his.ymin;
  // if ((pro_width / his_width) > MAX_VAR_HW ||
  //     (pro_height / his_height) > MAX_VAR_HW ||
  //     (his_width / pro_width) > MAX_VAR_HW ||
  //     (his_height / pro_height) > MAX_VAR_HW)
  if (false) {
    // do nothing
    person_his.active = false;
    person_his.loss_cnt = 0;
  } else {
    // x/y/w/h
    person_his.xmin = person_pro.xmin;
    person_his.ymin = person_pro.ymin;
    person_his.xmax = person_pro.xmax;
    person_his.ymax = person_pro.ymax;
    person_his.center_x = person_pro.center_x;
    person_his.center_y = person_pro.center_y;
    person_his.width = person_pro.width;
    person_his.height = person_pro.height;
    // kps
    for (int i = 0; i < 18; ++i) {
      person_his.kps[i].x = person_pro.kps[i].x;
      person_his.kps[i].y = person_pro.kps[i].y;
      person_his.kps[i].v = person_pro.kps[i].v;
    }
    // num_points
    person_his.num_points = person_pro.num_points;
    // score
    person_his.score = person_pro.score;
    // Note: ID not updated.
    person_his.active = true;
    person_his.loss_cnt = 0;
  }
}

template <typename Dtype>
bool EasymatchLayer<Dtype>::at_edge(person_t& person, float gap) {
  if (person.xmin < gap) return true;
  if (person.xmax > (1.0-gap)) return true;
  if (person.ymin < gap) return true;
  if (person.ymax > (1.0-gap)) return true;
  return false;
}

template <typename Dtype>
int EasymatchLayer<Dtype>::get_active_id() {
  int id = 1;
  while(1) {
    bool found = false;
    for (int i = 0; i < cur_persons_.size(); ++i) {
      if (cur_persons_[i].id == id) {
        found = true;
        break;
      }
    }
    if (found) {
      id++;
    } else {
      return id;
    }
  }
}

template <typename Dtype>
void EasymatchLayer<Dtype>::get_matched_cols(map<int, map<int, int > >& match_status, int row,
                                             vector<int>* indices) {
  indices->clear();
  if (match_status.find(row) == match_status.end()) {
    return;
  } else {
    map<int, int>& col_status = match_status[row];
    for (map<int, int>::iterator it = col_status.begin(); it != col_status.end(); ++it) {
      if (it->second == 1) {
        indices->push_back(it->first);
      }
    }
  }
}

template <typename Dtype>
void EasymatchLayer<Dtype>::get_matched_rows(map<int, map<int, int > >& match_status, int col,
                                             vector<int>* indices) {
  indices->clear();
  for (map<int, map<int, int> >::iterator it = match_status.begin(); it != match_status.end(); ++it) {
    if (it->second.find(col) == it->second.end()) continue;
    if (it->second[col] == 1) {
      indices->push_back(it->first);
    }
  }
}

template <typename Dtype>
void EasymatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const EasymatchParameter& easy_match_param = this->layer_param_.easy_match_param();
  // vars_
  float sigma[18] = {0.26,0.79,0.79,0.72,0.62,0.79,0.72,0.62,1.07,0.87,0.89,1.07,0.87,0.89,0.25,0.25,0.35,0.35};
  for (int i = 0; i < 18; ++i) {
    float var = pow(2*sigma[i], 2);
    vars_.push_back(var);
  }
  match_iou_thre_ = easy_match_param.match_iou_thre();
  edge_gap_ = easy_match_param.edge_gap();
}

template <typename Dtype>
void EasymatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  // bottom[0] -> proposals
	CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 61);
  // bottom[1] -> heatmaps & vecmaps
	CHECK_EQ(bottom[1]->num(),1);
  CHECK_EQ(bottom[1]->channels(),52);
  // top
  top[0]->Reshape(1,1,1,61);
}

template <typename Dtype>
void EasymatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* proposal_ptr = bottom[0]->cpu_data();
  temp_persons_.clear();
  // // get proposals
  vector<person_t> proposals;
  int id = 1;
  for(int i = 0; i < bottom[0]->height(); ++i) {
    int idx = i * bottom[0]->width();
    if (proposal_ptr[idx] < 0) continue;
    person_t person;
    // x/y/w/h
    person.xmin = proposal_ptr[idx];
    person.ymin = proposal_ptr[idx+1];
    person.xmax = proposal_ptr[idx+2];
    person.ymax = proposal_ptr[idx+3];
    person.center_x = (person.xmin + person.xmax) / 2;
    person.center_y = (person.ymin + person.ymax) / 2;
    person.width = person.xmax - person.xmin;
    person.height = person.ymax - person.ymin;
    // kps
    person.kps.resize(18);
    for (int j = 0; j < 18; ++j) {
      person.kps[j].x = proposal_ptr[idx+4+3*j];
      person.kps[j].y = proposal_ptr[idx+5+3*j];
      person.kps[j].v = proposal_ptr[idx+6+3*j];
    }
    // num_points
    person.num_points = proposal_ptr[idx+58];
    // score
    person.score = proposal_ptr[idx+59];
    // id
    person.id = id;
    id++;
    // push
    proposals.push_back(person);
  }
  // get matches for proposals & cur_persons_
  if (cur_persons_.size() == 0) {
    // Add active targets
    for (int i = 0; i < proposals.size(); ++i) {
      // float size = proposals[i].width * proposals[i].height;
      proposals[i].active = true;
      proposals[i].loss_cnt = 0;
      cur_persons_.push_back(proposals[i]);
      // if (size < 0.05) {
      //   temp_persons_.push_back(proposals[i]);
      // } else {
      //   cur_persons_.push_back(proposals[i]);
      // }
    }
  } else if (proposals.size() == 0) {
    // process history targets
    // for (int j = 0; j < cur_persons_.size(); ++j) {
    //   if (at_edge(cur_persons_[j], edge_gap_)) {
    //     cur_persons_.erase(cur_persons_.begin()+j);
    //   } else {
    //     float size = cur_persons_[j].width * cur_persons_[j].height;
    //     if (size < 0.05) {
    //       cur_persons_.erase(cur_persons_.begin()+j);
    //     }
    //   }
    // }
    for (int j = 0; j < cur_persons_.size(); ++j) {
      cur_persons_[j].active = false;
      cur_persons_[j].loss_cnt++;
      if (cur_persons_[j].loss_cnt > LOSS_MAX) {
        cur_persons_.erase(cur_persons_.begin()+j);
      }
    }
  } else {
    // match and update
    vector<bool> pro_matched(proposals.size(), false);
    vector<bool> his_matched(cur_persons_.size(), false);
    map<int, map<int, int > > match_status;
    map<int, map<int, vector<float> > > match_res;
    for (int i = 0; i < proposals.size(); ++i) {
      for (int j = 0; j < cur_persons_.size(); ++j) {
        matchsts_t sts;
        get_similarity(proposals[i], cur_persons_[j], sts);
        if (sts.iou > match_iou_thre_) {
          match_status[i][j] = 1;
          match_res[i][j].push_back(sts.iou);
          match_res[i][j].push_back(sts.oks);
          match_res[i][j].push_back(sts.similarity);
        } else {
          match_status[i][j] = 0;
        }
      }
    }
    // curr proposals
    for (int i = 0; i < proposals.size(); ++i) {
      if (pro_matched[i]) continue;
      vector<int> matched_idx_pro;
      get_matched_cols(match_status, i, &matched_idx_pro);
      if (matched_idx_pro.size() > 0) {
        // search matched person
        int best_col = -1;
        float best_val = -1;
        for (int j = 0; j < matched_idx_pro.size(); ++j) {
          int col = matched_idx_pro[j];
          if (his_matched[col]) continue;
          if (match_res[i][col][2] > best_val) {
            best_col = col;
            best_val = match_res[i][col][2];
          }
        }
        if (best_col >= 0) {
          update_person(proposals[i], cur_persons_[best_col]);
          pro_matched[i] = true;
          his_matched[best_col] = true;
        }
      } else {
        proposals[i].id = get_active_id();
        proposals[i].active = true;
        proposals[i].loss_cnt = 0;
        cur_persons_.push_back(proposals[i]);
      }
    }
    // history
    for (int j = 0; j < cur_persons_.size(); ++j) {
      if (his_matched[j]) continue;
      cur_persons_[j].active = false;
      cur_persons_[j].loss_cnt++;
      if (cur_persons_[j].loss_cnt > LOSS_MAX) {
        cur_persons_.erase(cur_persons_.begin()+j);
      }
      // vector<int> matched_idx_his;
      // get_matched_rows(match_status, j, &matched_idx_his);
      // if (matched_idx_his.size() > 0) {
      //   // select the best one to update
      //   int best_row = -1;
      //   float best_val = -1;
      //   for (int i = 0; i < matched_idx_his.size(); ++i) {
      //     int row = matched_idx_his[i];
      //     if (pro_matched[row]) continue;
      //     if (match_res[row][j][0] > best_val) {
      //       best_row = row;
      //       best_val = match_res[row][j][0];
      //     }
      //   }
      //   if (best_row >= 0) {
      //     update_person(proposals[best_row], cur_persons_[j]);
      //     pro_matched[best_row] = true;
      //     his_matched[j] = true;
      //   }
      // } else {
      //   // del it or Tracking
      //   // if (at_edge(cur_persons_[j], edge_gap_)) {
      //   //   // del
      //   //   cur_persons_.erase(cur_persons_.begin()+j);
      //   // } else {
      //   //   float size = cur_persons_[j].width * cur_persons_[j].height;
      //   //   if (size < 0.05) {
      //   //     cur_persons_.erase(cur_persons_.begin()+j);
      //   //   }
      //   // }
      //   // cur_persons_.erase(cur_persons_.begin()+j);
      //   cur_persons_[j].active = false;
      //   cur_persons_[j].loss_cnt++;
      //   if (cur_persons_[j].loss_cnt > LOSS_MAX) {
      //     cur_persons_.erase(cur_persons_.begin()+j);
      //   }
      // }
    }
    // 为proposals进行匹配
    // 已经匹配过的,跳过
    // 未匹配的, 只有没有匹配的对象,为新对象
    // 存在匹配的, 放入临时列表,但不予注册
    for (int i = 0; i < proposals.size(); ++i) {
      if (pro_matched[i]) continue;
      vector<int> matched_idx_pro;
      get_matched_cols(match_status, i, &matched_idx_pro);
      if (matched_idx_pro.size() > 0) {
        // push temp_persons
        proposals[i].id = -1;
        temp_persons_.push_back(proposals[i]);
      } else {
        // float size = proposals[i].width * proposals[i].height;
        // if (size < 0.05) {
        //   proposals[i].id = -1;
        //   temp_persons_.push_back(proposals[i]);
        // } else {
        // // append
        // proposals[i].id = get_active_id();
        // cur_persons_.push_back(proposals[i]);
        // }
        proposals[i].id = get_active_id();
        proposals[i].active = true;
        proposals[i].loss_cnt = 0;
        cur_persons_.push_back(proposals[i]);
      }
    }
  }
  // output
  int cur_nums = 0;
  for (int i = 0; i < cur_persons_.size(); ++i) {
    if (cur_persons_[i].active) {
      cur_nums++;
    }
  }
  int num_p = cur_nums + temp_persons_.size();
  if (num_p == 0) {
    top[0]->Reshape(1,1,1,61);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(1,1,num_p,61);
    Dtype* top_data = top[0]->mutable_cpu_data();
    int idx = 0;
    // 首先输出cur_persons_
    for (int i = 0; i < cur_persons_.size(); ++i) {
      if (!cur_persons_[i].active) continue;
      top_data[idx++] = cur_persons_[i].xmin;
      top_data[idx++] = cur_persons_[i].ymin;
      top_data[idx++] = cur_persons_[i].xmax;
      top_data[idx++] = cur_persons_[i].ymax;
      for (int k = 0; k < 18; ++k) {
        top_data[idx++] = cur_persons_[i].kps[k].x;
        top_data[idx++] = cur_persons_[i].kps[k].y;
        top_data[idx++] = cur_persons_[i].kps[k].v;
      }
      top_data[idx++] = cur_persons_[i].num_points;
      top_data[idx++] = cur_persons_[i].score;
      top_data[idx++] = cur_persons_[i].id;
      // top_data[idx++] = 1;
    }
    // 再输出temp_persons_
    for (int i = 0; i < temp_persons_.size(); ++i) {
      top_data[idx++] = temp_persons_[i].xmin;
      top_data[idx++] = temp_persons_[i].ymin;
      top_data[idx++] = temp_persons_[i].xmax;
      top_data[idx++] = temp_persons_[i].ymax;
      for (int k = 0; k < num_parts_; ++k) {
        top_data[idx++] = temp_persons_[i].kps[k].x;
        top_data[idx++] = temp_persons_[i].kps[k].y;
        top_data[idx++] = temp_persons_[i].kps[k].v;
      }
      top_data[idx++] = temp_persons_[i].num_points;
      top_data[idx++] = temp_persons_[i].score;
      top_data[idx++] = temp_persons_[i].id;
      // top_data[idx++] = 1;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EasymatchLayer);
#endif

INSTANTIATE_CLASS(EasymatchLayer);
REGISTER_LAYER_CLASS(Easymatch);

} // namespace caffe
