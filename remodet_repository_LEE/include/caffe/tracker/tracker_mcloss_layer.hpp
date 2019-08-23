#ifndef CAFFE_TRACKER_MCLOSS_LAYER_HPP_
#define CAFFE_TRACKER_MCLOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * 该层是模仿YOLO-Detector的方式对回归器进行改造，以对相似物体进行区分。
 * 训练结果比较糟糕，目前不推荐使用。
 */

template <typename Dtype>
class TrackerMcLossLayer : public LossLayer<Dtype> {
 public:
  explicit TrackerMcLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TrackerMcLoss"; }
  // bottom[0] stores the predictions. [N,5,D,D]
  // where 5 -> [score, px ,py, pw, ph]
  // bottom[1] stores the GT [gx,gy,gw,gh] (Normalized) [N, 4, 1, 1]
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Blobs stores the diff of predictions
  Blob<Dtype> pred_diff_;

  // number of minibatch
  int num_;
  int grids_;
  // weight of loss-score
  Dtype score_scale_;
  // weight of loss-bbox
  Dtype loc_scale_;
  // 0.5 / 0.5
  Dtype prior_width_;
  Dtype prior_height_;
  Dtype overlap_threshold_;
};

}  // namespace caffe

#endif  // CAFFE_TRACKER_MCLOSS_LAYER_HPP_
