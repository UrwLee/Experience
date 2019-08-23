#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/tracker/tracker_mcloss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void TrackerMcLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  // 获取multibox的损失参数
  const TrackerMcLossParameter& tracker_mcloss_param =
      this->layer_param_.tracker_mcloss_param();
  // weights of score & bbox
  score_scale_ = tracker_mcloss_param.score_scale();
  loc_scale_ = tracker_mcloss_param.loc_scale();
  // used anchors -> .5 / .5
  prior_width_ = tracker_mcloss_param.prior_width();
  prior_height_ = tracker_mcloss_param.prior_height();
  overlap_threshold_ = tracker_mcloss_param.overlap_threshold();
}

template <typename Dtype>
void TrackerMcLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //add loss
  vector<int> top_shape(4,1);
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->num();
  grids_ = bottom[0]->height();
  CHECK_EQ(grids_, bottom[0]->width());
  CHECK_EQ(bottom[0]->channels(), 5) << "bottom[0]->channels() should be 5.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Unmatched for numbers of labels & images.";
  CHECK_EQ(bottom[1]->channels(), 4);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // diff_
  pred_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TrackerMcLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  num_ = bottom[0]->num();
  grids_ = bottom[0]->height();
  // bottom[0] -> [N, 5, G, G]
  int outsize = grids_* grids_* 5;
  /**************************************************************************#
  Step-0: Logistic for score
  #***************************************************************************/
  Dtype* mutable_pred_data = bottom[0]->mutable_cpu_data();
  for (int item_id = 0; item_id < num_; ++item_id) {
    for (int i = 0; i < grids_; ++i) {
      for (int j = 0; j < grids_; ++j) {
        int idx = item_id * outsize + i * grids_ + j;
        mutable_pred_data[idx] = logistic(pred_data[idx]);
      }
    }
  }
  // initialize the pred-diff
  Dtype* pred_diff = pred_diff_.mutable_cpu_data();
  caffe_set(pred_diff_.count(), Dtype(0), pred_diff);
  // 开始逐个样本进行处理
  for (int item_id = 0; item_id < num_; ++item_id) {
    // 获取GT
    NormalizedBBox gt_bbox;
    // [x1,y1,x2,y2]
    gt_bbox.set_xmin(gt_data[item_id * 4]);
    gt_bbox.set_ymin(gt_data[item_id * 4 + 1]);
    gt_bbox.set_xmax(gt_data[item_id * 4 + 2]);
    gt_bbox.set_ymax(gt_data[item_id * 4 + 3]);
    /**************************************************************************#
    Step-1: 初始化为BG
    #***************************************************************************/
    for (int i = 0; i < grids_; ++i) {
      for (int j = 0; j < grids_; ++j) {
        int idx = item_id * outsize + i * grids_ + j;
        NormalizedBBox pred_bbox, corner_pred_box;
        // 计算该anchor的坐标
        Dtype xcenter = (Dtype(j) + 0.5) / grids_;
        Dtype ycenter = (Dtype(i) + 0.5) / grids_;
        pred_bbox.set_xmin(xcenter);
        pred_bbox.set_ymin(ycenter);
        pred_bbox.set_xmax(prior_width_);
        pred_bbox.set_ymax(prior_height_);
        CenterToCorner(pred_bbox, &corner_pred_box);
        Dtype iou = JaccardOverlap(corner_pred_box, gt_bbox);
        // 作为反例 -> Logistic Loss
        if (iou < overlap_threshold_) {
          pred_diff[idx] =
            score_scale_ * (pred_data[idx] - 0) * logistic_gradient(pred_data[idx]);
        }
      }
    }
    /**************************************************************************#
    Step-2： 计算匹配的LOC+分类损失
    #***************************************************************************/
    NormalizedBBox gt_center;
    CornerToCenter(gt_bbox,&gt_center);
    // get the (i,j) for matching
    int j = (int)(gt_center.xmin() * grids_);
    int i = (int)(gt_center.ymin() * grids_);
    int idx = item_id * outsize + i * grids_ + j;
    // 获取估计的pred & IOU
    // NormalizedBBox pred_bbox, corner_pred_bbox;
    // Dtype px, py, pw, ph;
    // px = (j + logistic(pred_data[idx + grids_ * grids_])) / grids_;
    // py = (i + logistic(pred_data[idx + 2 * grids_ * grids_])) / grids_;
    // pw = exp(pred_data[idx + 3 * grids_ * grids_]) * prior_width_;
    // ph = exp(pred_data[idx + 4 * grids_ * grids_]) * prior_height_;
    // pred_bbox.set_xmin(px);
    // pred_bbox.set_ymin(py);
    // pred_bbox.set_xmax(pw);
    // pred_bbox.set_ymax(ph);
    // CenterToCorner(pred_bbox, &corner_pred_bbox);
    // Dtype iou = JaccardOverlap(corner_pred_bbox, gt_bbox);
    // 计算LOC误差
    Dtype center_x_gt = gt_center.xmin() * grids_ - j;
    Dtype center_y_gt = gt_center.ymin() * grids_ - i;
    Dtype w_gt = log(gt_center.xmax() / prior_width_);
    Dtype h_gt = log(gt_center.ymax() / prior_height_);
    Dtype act_x = logistic(pred_data[idx + grids_ * grids_]);
    Dtype act_y = logistic(pred_data[idx + 2 * grids_ * grids_]);
    Dtype pred_w = pred_data[idx + 3 * grids_ * grids_];
    Dtype pred_h = pred_data[idx + 4 * grids_ * grids_];
    pred_diff[idx + 1 * grids_ * grids_] = loc_scale_ * (act_x - center_x_gt) * logistic_gradient(act_x);
    pred_diff[idx + 2 * grids_ * grids_] = loc_scale_ * (act_y - center_y_gt) * logistic_gradient(act_y);
    pred_diff[idx + 3 * grids_ * grids_] = loc_scale_ * (pred_w - w_gt);
    pred_diff[idx + 4 * grids_ * grids_] = loc_scale_ * (pred_h - h_gt);
    // score 分类误差
    pred_diff[idx] = score_scale_ * (pred_data[idx] - (Dtype)1) * logistic_gradient(pred_data[idx]);
  }
  /**************************************************************************#
  Step-Final： Loss
  #***************************************************************************/
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype loc_loss = 0, score_loss = 0;
  Dtype loss;
  const Dtype* pred_diff_data = pred_diff_.cpu_data();
  for (int item_id = 0; item_id < num_; ++item_id) {
    for (int i = 0; i < grids_; ++i) {
      for (int j = 0; j < grids_; ++j) {
        int idx = item_id * outsize + i * grids_ + j;
        // compute score_loss
        score_loss += pred_diff_data[idx] * pred_diff_data[idx];
        // compute loc_loss
        loc_loss += pred_diff_data[idx + 1 * grids_ * grids_] * pred_diff_data[idx + 1 * grids_ * grids_];
        loc_loss += pred_diff_data[idx + 2 * grids_ * grids_] * pred_diff_data[idx + 2 * grids_ * grids_];
        loc_loss += pred_diff_data[idx + 3 * grids_ * grids_] * pred_diff_data[idx + 3 * grids_ * grids_];
        loc_loss += pred_diff_data[idx + 4 * grids_ * grids_] * pred_diff_data[idx + 4 * grids_ * grids_];
      }
    }
  }
  loss = loc_loss + score_loss;
  top_data[0] = loss / num_;
  // you can print the loss info here.
}

template <typename Dtype>
void TrackerMcLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to ground truth.";
  }
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / num_;
    const Dtype* diff_data = pred_diff_.cpu_data();
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_data,
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(TrackerMcLossLayer);
REGISTER_LAYER_CLASS(TrackerMcLoss);

}  // namespace caffe
