#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "caffe/tracker/tracker_mcout_layer.hpp"

namespace caffe {

template <typename Dtype>
void TrackerMcOutLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const TrackerMcOutParameter& tracker_mcout_param = this->layer_param_.tracker_mcout_param();
  prior_width_ = tracker_mcout_param.prior_width();
  prior_height_ = tracker_mcout_param.prior_height();
}

template <typename Dtype>
void TrackerMcOutLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->num(), 1) << "bottom[0]->num() has an invalid value of "
                                << bottom[0]->num() << ", it should be one.";
  grids_ = bottom[0]->height();
  CHECK_EQ(grids_, bottom[0]->width());
  CHECK_EQ(bottom[0]->channels(), 5) << "bottom[0]->channels() has an invalid value of "
           << bottom[0]->channels() << ", it should be 5.";
  vector<int> top_shape(4, 1);
  // [score, x, y, w, h] (Normalized value)
  top_shape[1] = 5;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TrackerMcOutLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *pred_data = bottom[0]->cpu_data();
  // get all predictions
  vector<Dtype> all_scores;
  vector<NormalizedBBox> all_bboxes;
  for (int i = 0; i < grids_; ++i) {
    for (int j = 0; j < grids_; ++j) {
      int idx = i * grids_ + j;
      all_scores.push_back(logistic(pred_data[idx]));
      NormalizedBBox pred_bbox, corner_pred_bbox;
      Dtype px, py, pw, ph;
      px = (j + logistic(pred_data[idx + grids_ * grids_])) / grids_;
      py = (i + logistic(pred_data[idx + 2 * grids_ * grids_])) / grids_;
      pw = exp(pred_data[idx + 3 * grids_ * grids_]) * prior_width_;
      ph = exp(pred_data[idx + 4 * grids_ * grids_]) * prior_height_;
      pred_bbox.set_xmin(px);
      pred_bbox.set_ymin(py);
      pred_bbox.set_xmax(pw);
      pred_bbox.set_ymax(ph);
      CenterToCorner(pred_bbox, &corner_pred_bbox);
      all_bboxes.push_back(corner_pred_bbox);
    }
  }
  // choose the maximum score bbox
  int best_idx = 0;
  Dtype best_val = 0;
  for (int i = 0; i < all_scores.size(); ++i) {
    if (all_scores[i] > best_val) {
      best_val = all_scores[i];
      best_idx = i;
    }
  }
  // get the best bbox
  Dtype* top_data = top[0]->mutable_cpu_data();
  // clip the bbox
  ClipBBox(all_bboxes[best_idx], &all_bboxes[best_idx]);
  top_data[0] = best_val;
  top_data[1] = all_bboxes[best_idx].xmin();
  top_data[2] = all_bboxes[best_idx].ymin();
  top_data[3] = all_bboxes[best_idx].xmax();
  top_data[4] = all_bboxes[best_idx].ymax();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(TrackerMcOutLayer, Forward);
#endif

INSTANTIATE_CLASS(TrackerMcOutLayer);
REGISTER_LAYER_CLASS(TrackerMcOut);

} // namespace caffe
