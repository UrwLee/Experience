#ifndef CAFFE_MASK_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_MASK_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class MaskSoftmaxWithLossLayer : public LossLayer<Dtype> {
 public:
  explicit MaskSoftmaxWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskSoftmaxWithLoss"; }
  /**
  * bottom[0]: -> Mask preds (Nroi,18,RH*RW)
  * bottom[1]: -> Mask GTMap (Nroi,18)
  * bottom[2]: -> Active flags for each map (1,1,Nroi,18)
  * if active_flag = 1 -> 误差正常反馈和计算
  * if active_flag = 0 -> 误差清零，无反向传播
   */
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  /**
   * top[0]: -> loss
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Layer<Dtype> > softmax_layer_;
  Blob<Dtype> prob_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  bool has_ignore_label_;
  int ignore_label_;
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_;

  Dtype scale_;
};

}

#endif
