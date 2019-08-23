#include <string>
#include <vector>
#include <map>
#include <utility>
#include <stdio.h>

#include "caffe/layers/pose_det_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void PoseDetLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const PoseDetParameter &pose_det_param =
      this->layer_param_.pose_det_param();

  num_parts_ = 18;

  // threshold
  coverage_min_thre_ = pose_det_param.coverage_thre();
  score_pose_ebox_ = pose_det_param.score_pose_ebox();
  keep_det_box_thre_ = pose_det_param.keep_det_box_thre();
}

template <typename Dtype>
void PoseDetLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // bottom[0]: pose_proposal
  // [1, num_people, num_parts+1, 3]
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->height(), num_parts_+1);
  CHECK_EQ(bottom[0]->width(), 3);
  // bottom[1]: det_proposal
  // [1,1,num_people,7]
  if (bottom.size() > 1) {
    CHECK_EQ(bottom[1]->num(), 1);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->width(), 7);
  }
  // bottom[2]: peaks
  // [1, num_parts, max_peaks_+1, 3]
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[2]->num(), 1);
    CHECK_EQ(bottom[2]->channels(), num_parts_);
    max_peaks_ = bottom[2]->height() - 1;
    CHECK_EQ(bottom[2]->width(), 3);
  }
  // bottom[3]: heatmaps
  // [1,18+34,h,w]
  if (bottom.size() > 3) {
    CHECK_EQ(bottom[3]->num(), 1);
    CHECK_EQ(bottom[3]->channels(), 52);
  }

  // top[0]: [1,1,n,60]
  // 0-3: box [xmin,ymin,xmax,ymax] (normalized)
  // 4-57: kps (18x3) (normalized)
  // 58: num of points
  // 59: score, and 60: id (-1)
  top[0]->Reshape(1,1,1,7+num_parts_*3);
}

template <typename Dtype>
void PoseDetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype UNFOUND_THRE = 0.05;

  const Dtype* pose_data = bottom[0]->cpu_data();
  const Dtype* det_data = NULL;
  if (bottom.size() > 1) {
    det_data = bottom[1]->cpu_data();
  } else {
    Blob<Dtype> b1(1,1,1,7);
    caffe_set(7,Dtype(-1),b1.mutable_cpu_data());
    det_data = b1.cpu_data();
  }
  const Dtype* peak_data = NULL;
  if (bottom.size() > 2) {
    peak_data = bottom[2]->cpu_data();
  } else {
    Blob<Dtype> b2(1,18,10,3);
    caffe_set(b2.count(),Dtype(0),b2.mutable_cpu_data());
    peak_data = b2.cpu_data();
  }
  // if (bottom.size() == 4) {
  //   // const Dtype* map_data = bottom[3]->cpu_data();
  // }
  //----------------------------------------------------------------------------
  // 获取proposal的boxes: pose & det
  vector<NormalizedBBox> pose_bboxes;
  vector<vector<Dtype> > pose_kps;
  vector<NormalizedBBox> det_bboxes;
  // pose proposal push -> pose_bboxes;
  // pose proposal push -> pose_kps;
  int num_pose_person = bottom[0]->channels();
  if (pose_data[0] < 0 && num_pose_person == 1) {
    num_pose_person = 0;
  }
  for (int i = 0; i < num_pose_person; ++i) {
    NormalizedBBox box;
    vector<Dtype> kps(3*num_parts_+3,Dtype(0));
    Dtype xmin = FLT_MAX, ymin = FLT_MAX, xmax = -FLT_MAX, ymax = -FLT_MAX;
    int num_keypoints = 0;
    for (int k = 0; k < num_parts_; ++k) {
      Dtype x = pose_data[i*(num_parts_+1)*3 + k*3];
      Dtype y = pose_data[i*(num_parts_+1)*3 + k*3 + 1];
      Dtype v = pose_data[i*(num_parts_+1)*3 + k*3 + 2];
      kps[3*k] = x;
      kps[3*k+1] = y;
      kps[3*k+2] = v;
      if (v > 0.01) {
        num_keypoints++;
        xmin = (x < xmin) ? x : xmin;
        ymin = (y < ymin) ? y : ymin;
        xmax = (x > xmax) ? x : xmax;
        ymax = (y > ymax) ? y : ymax;
      }
    }
    // count of parts
    kps[3*num_parts_] = pose_data[i*(num_parts_+1)*3 + num_parts_*3];
    // sum of score
    kps[3*num_parts_+1] = pose_data[i*(num_parts_+1)*3 + num_parts_*3 + 1];
    // avg_score
    kps[3*num_parts_+2] = pose_data[i*(num_parts_+1)*3 + num_parts_*3 + 2];
    // push到pose_bboxes
    if (num_keypoints >= 3) {
      // get boxes
      box.set_xmin(xmin);
      box.set_ymin(ymin);
      box.set_xmax(xmax);
      box.set_ymax(ymax);
      pose_bboxes.push_back(box);
      pose_kps.push_back(kps);
    }
  }
  //----------------------------------------------------------------------------
  // det-proposal push
  int num_det_person = bottom[1]->height();
  if (det_data[0] < 0 && num_det_person == 1) {
    num_det_person = 0;
  }
  for (int i = 0; i < num_det_person; ++i) {
    Dtype score = det_data[i * 7 + 2];
    if (score > 0.01) {
      NormalizedBBox box;
      box.set_xmin(det_data[i * 7 + 3]);
      box.set_ymin(det_data[i * 7 + 4]);
      box.set_xmax(det_data[i * 7 + 5]);
      box.set_ymax(det_data[i * 7 + 6]);
      box.set_score(score);
      det_bboxes.push_back(box);
    }
  }
  //----------------------------------------------------------------------------
  // 遍历pose-boxes, 完成匹配
  vector<vector<Dtype> > matched_vecs;
  vector<bool> matched_det(det_bboxes.size(), false);
  for (int g = 0; g < pose_bboxes.size(); ++g) {
    // found matched indices
    int matched_idx = -1;
    float matched_max_coverage = 0;
    float matched_max_iou = 0;
    for (int p = 0; p < det_bboxes.size(); ++p) {
      if (matched_det[p]) continue;
      float coverage = BBoxCoverage(pose_bboxes[g], det_bboxes[p]);
      float iou = JaccardOverlap(pose_bboxes[g], det_bboxes[p]);
      // 由coverage进行匹配选择
      if ((coverage > coverage_min_thre_) && (std::abs(coverage-matched_max_coverage) > 0.01)
          && (coverage > matched_max_coverage)) {
        matched_max_coverage = coverage;
        matched_max_iou = iou;
        matched_idx = p;
      } else if ((coverage > coverage_min_thre_) && (std::abs(coverage-matched_max_coverage) < 0.01)) {
        // 由IOU来进行匹配选择
        if (iou > matched_max_iou) {
          matched_max_coverage = coverage;
          matched_max_iou = iou;
          matched_idx = p;
        }
      } else {
        continue;
      }
    }
    // 匹配成功,完成数据融合
    if (matched_idx >= 0) {
      matched_det[matched_idx] = true;
      vector<Dtype> out_vec(7+3*num_parts_,0);
      // output x,y,w,h
      float xmin_pose = pose_bboxes[g].xmin();
      float xmax_pose = pose_bboxes[g].xmax();
      float ymin_pose = pose_bboxes[g].ymin();
      float ymax_pose = pose_bboxes[g].ymax();
      float xmin_det = det_bboxes[matched_idx].xmin();
      float xmax_det = det_bboxes[matched_idx].xmax();
      float ymin_det = det_bboxes[matched_idx].ymin();
      float ymax_det = det_bboxes[matched_idx].ymax();
      if (xmin_det > xmin_pose) {
        xmin_det = xmin_pose - (xmax_pose - xmin_pose) * 0.1;
        xmin_det = std::min(std::max(xmin_det, (float)0),(float)1);
      }
      if (xmax_det < xmax_pose) {
        xmax_det = xmax_pose + (xmax_pose - xmin_pose) * 0.1;
        xmax_det = std::min(std::max(xmax_det, (float)0),(float)1);
      }
      if (ymin_det > ymin_pose) {
        ymin_det = ymin_pose - (ymax_pose - ymin_pose) * 0.1;
        ymin_det = std::min(std::max(ymin_det, (float)0),(float)1);
      }
      if (ymax_det < ymax_pose) {
        ymax_det = ymax_pose + (ymax_pose - ymin_pose) * 0.1;
        ymax_det = std::min(std::max(ymax_det, (float)0),(float)1);
      }
      out_vec[0] = xmin_det;
      out_vec[1] = ymin_det;
      out_vec[2] = xmax_det;
      out_vec[3] = ymax_det;
      // output points
      int n_points = 0;
      for (int k = 0; k < num_parts_; ++k) {
        float xk = pose_kps[g][3*k];
        float yk = pose_kps[g][3*k+1];
        float vk = pose_kps[g][3*k+2];
        if (vk > 0.05) {
          n_points++;
          out_vec[4+3*k] = xk;
          out_vec[4+3*k+1] = yk;
          out_vec[4+3*k+2] = vk;
        } else {
          // search in roi, found the max point in this area
          int idx_k = k * (max_peaks_ + 1) * 3;
          int max_peaks_k = (int)peak_data[idx_k];
          int max_p = -1;
          float max_val = UNFOUND_THRE;
          float x_k,y_k,v_k;
          for (int nk = 1; nk <= max_peaks_k; ++nk) {
            float xp = peak_data[idx_k+nk*3];
            float yp = peak_data[idx_k+nk*3+1];
            float vp = peak_data[idx_k+nk*3+2];
            if ((xp > xmin_det) && (xp < xmax_det) && (yp > ymin_det) && (yp < ymax_det) && (vp > max_val)) {
              max_val = vp;
              max_p = nk;
              x_k = xp;
              y_k = yp;
              v_k = vp;
            }
          }
          if (max_p >= 1) {
            n_points++;
            out_vec[4+3*k] = x_k;
            out_vec[4+3*k+1] = y_k;
            out_vec[4+3*k+2] = v_k;
          }
        }
      }
      out_vec[4+3*num_parts_] = n_points;
      // LOG(INFO) << "matched pose & bbox, Find " << n_points << " points.";
      // 输出评分
      out_vec[5+3*num_parts_] = det_bboxes[matched_idx].score() * pose_kps[g][3*num_parts_+2];
      out_vec[6+3*num_parts_] = -1;
      matched_vecs.push_back(out_vec);
    } else {
      // 匹配失败, 使用关键点估计box
      vector<Dtype> out_vec(7+3*num_parts_,0);
      // output box
      float xmin_pose = pose_bboxes[g].xmin();
      float xmax_pose = pose_bboxes[g].xmax();
      float ymin_pose = pose_bboxes[g].ymin();
      float ymax_pose = pose_bboxes[g].ymax();
      xmin_pose -= (xmax_pose - xmin_pose) * 0.1;
      xmin_pose = std::min(std::max(xmin_pose, (float)0),(float)1);
      xmax_pose += (xmax_pose - xmin_pose) * 0.1;
      xmax_pose = std::min(std::max(xmax_pose, (float)0),(float)1);
      ymin_pose -= (ymax_pose - ymin_pose) * 0.1;
      ymin_pose = std::min(std::max(ymin_pose, (float)0),(float)1);
      ymax_pose += (ymax_pose - ymin_pose) * 0.1;
      ymax_pose = std::min(std::max(ymax_pose, (float)0),(float)1);
      out_vec[0] = xmin_pose;
      out_vec[1] = ymin_pose;
      out_vec[2] = xmax_pose;
      out_vec[3] = ymax_pose;
      int n_points = 0;
      for (int k = 0; k < num_parts_; ++k) {
        float xk = pose_kps[g][3*k];
        float yk = pose_kps[g][3*k+1];
        float vk = pose_kps[g][3*k+2];
        // detected
        if (vk > 0.05) {
          n_points++;
          out_vec[4+3*k] = xk;
          out_vec[4+3*k+1] = yk;
          out_vec[4+3*k+2] = vk;
        } else {
          // unfound
          int idx_k = k * (max_peaks_ + 1) * 3;
          int max_peaks_k = (int)peak_data[idx_k];
          int max_p = -1;
          float max_val = UNFOUND_THRE;
          float x_k,y_k,v_k;
          for (int nk = 1; nk <= max_peaks_k; ++nk) {
            float xp = peak_data[idx_k+nk*3];
            float yp = peak_data[idx_k+nk*3+1];
            float vp = peak_data[idx_k+nk*3+2];
            if ((xp > xmin_pose) && (xp < xmax_pose) && (yp > ymin_pose) && (yp < ymax_pose) && (vp > max_val)) {
              max_val = vp;
              max_p = nk;
              x_k = xp;
              y_k = yp;
              v_k = vp;
            }
          }
          if (max_p >= 1) {
            n_points++;
            out_vec[4+3*k] = x_k;
            out_vec[4+3*k+1] = y_k;
            out_vec[4+3*k+2] = v_k;
          }
        }
      }
      // LOG(INFO) << "Unmatched pose, Find " << n_points << " points.";
      out_vec[4+3*num_parts_] = n_points;
      out_vec[5+3*num_parts_] = score_pose_ebox_ * pose_kps[g][3*num_parts_+2];
      out_vec[6+3*num_parts_] = -1;
      matched_vecs.push_back(out_vec);
    }
  }
  //----------------------------------------------------------------------------
  // 处理剩余的det-boxes
  // 找出检测box内部的关键点
  for (int i = 0; i < det_bboxes.size(); ++i) {
    // 已经匹配,跳过
    if (matched_det[i]) continue;
    // 评分不足,跳过
    if (det_bboxes[i].score() < keep_det_box_thre_) continue;
    vector<Dtype> out_vec(7+3*num_parts_, 0);
    // box
    Dtype xmin_det = det_bboxes[i].xmin();
    Dtype ymin_det = det_bboxes[i].ymin();
    Dtype xmax_det = det_bboxes[i].xmax();
    Dtype ymax_det = det_bboxes[i].ymax();
    out_vec[0] = xmin_det;
    out_vec[1] = ymin_det;
    out_vec[2] = xmax_det;
    out_vec[3] = ymax_det;
    int n_points = 0;
    for (int k = 0; k < num_parts_; ++k) {
      int idx_k = k * (max_peaks_ + 1) * 3;
      int max_peaks_k = (int)peak_data[idx_k];
      // LOG(INFO) << "[unmatched bbox] part-index (" << k << ")" << ": found " << max_peaks_k << " points.";
      int max_p = -1;
      float max_val = UNFOUND_THRE;
      float x_k,y_k,v_k;
      for (int nk = 1; nk <= max_peaks_k; ++nk) {
        float xp = peak_data[idx_k+nk*3];
        float yp = peak_data[idx_k+nk*3+1];
        float vp = peak_data[idx_k+nk*3+2];
        if ((xp > xmin_det) && (xp < xmax_det) && (yp > ymin_det) && (yp < ymax_det) && (vp > max_val)) {
          max_val = vp;
          max_p = nk;
          x_k = xp;
          y_k = yp;
          v_k = vp;
        }
      }
      if (max_p >= 1) {
        n_points++;
        out_vec[4+3*k] = x_k;
        out_vec[4+3*k+1] = y_k;
        out_vec[4+3*k+2] = v_k;
      }
    }
    // score
    out_vec[4+3*num_parts_] = n_points;
    out_vec[5+3*num_parts_] = det_bboxes[i].score();
    out_vec[6+3*num_parts_] = -1;
    matched_vecs.push_back(out_vec);
  }
  //----------------------------------------------------------------------------
  // filter
  int num_p = matched_vecs.size();
  if (num_p == 0) {
    top[0]->Reshape(1,1,1,7+num_parts_*3);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    // filter -> embedded boxes del.
    vector<bool> embedded(num_p,false);
    int real_np = 0;
    for (int i = 0; i < num_p; ++i) {
      NormalizedBBox box_i;
      box_i.set_xmin(matched_vecs[i][0]);
      box_i.set_ymin(matched_vecs[i][1]);
      box_i.set_xmax(matched_vecs[i][2]);
      box_i.set_ymax(matched_vecs[i][3]);
      for (int j = 0; j < num_p; ++j) {
        if (i == j) continue;
        NormalizedBBox box_j;
        box_j.set_xmin(matched_vecs[j][0]);
        box_j.set_ymin(matched_vecs[j][1]);
        box_j.set_xmax(matched_vecs[j][2]);
        box_j.set_ymax(matched_vecs[j][3]);
        float coverage = BBoxCoverage(box_i,box_j);
        if (coverage > 0.95) {
          embedded[i] = true;
          break;
        }
      }
      if (!embedded[i]) {
        real_np++;
      }
    }
    top[0]->Reshape(1,1,real_np,7+num_parts_*3);
    Dtype* top_data = top[0]->mutable_cpu_data();
    int idx = 0;
    for (int i = 0; i < num_p; ++i) {
      if (embedded[i]) continue;
      // 输出box信息
      top_data[idx++] = matched_vecs[i][0];
      top_data[idx++] = matched_vecs[i][1];
      top_data[idx++] = matched_vecs[i][2];
      top_data[idx++] = matched_vecs[i][3];
      // 输出kps信息
      for (int k = 0; k < num_parts_; ++k) {
        top_data[idx++] = matched_vecs[i][4+3*k];
        top_data[idx++] = matched_vecs[i][4+3*k+1];
        top_data[idx++] = matched_vecs[i][4+3*k+2];
      }
      top_data[idx++] = matched_vecs[i][4+3*num_parts_];
      // LOG(INFO) << "Find " << matched_vecs[i][4+3*num_parts_] << " points.";
      // 输出评分信息
      top_data[idx++] = matched_vecs[i][5+3*num_parts_];
      // ID
      top_data[idx++] = matched_vecs[i][6+3*num_parts_];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(PoseDetLayer, Forward);
#endif

INSTANTIATE_CLASS(PoseDetLayer);
REGISTER_LAYER_CLASS(PoseDet);

} // namespace caffe
