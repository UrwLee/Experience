// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#ifndef CAFFE_SMOOTH_LN_LOSS_LAYER_HPP_
#define CAFFE_SMOOTH_LN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the SmoothL1 loss as introduced in:@f$
 *  Fast R-CNN, Ross Girshick, ICCV 2015.
 */
template <typename Dtype>
class SmoothLnLossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothLnLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothLnLoss"; }

  // 输入2-3个Blobs
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * Unlike most loss layers, in the SmoothLnLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  // 该类损失层可以对所有输入都进行传播
  // 也可以将[1]修改为False
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc SmoothLnLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //  梯度 
  Blob<Dtype> diff_;
  // 误差
  Blob<Dtype> errors_;
  // 如果有3个输入,则该值为True
  // 如果有2个输入,则该值为False
  bool has_weights_;
  Dtype sigma;
};

}  // namespace caffe

#endif  // CAFFE_SMOOTH_LN_LOSS_LAYER_HPP_
