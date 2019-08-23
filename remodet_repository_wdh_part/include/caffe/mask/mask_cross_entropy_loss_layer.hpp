#ifndef CAFFE_MASK_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_MASK_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * 该层提供了带Mask的交叉熵损失函数。
 * 计算如下：
 * 对于处于Active的ROI,正常计算
 * 对于处于Unactive的ROI，误差无参与计算，也不传播
 * 注意：误差传播具有一个增益参数scale，可以由proto参数设定
 */

template <typename Dtype>
class MaskCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit MaskCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskCrossEntropyLoss"; }

  /**
   * bottom[0]: -> Kps Preds  (Nroi,18,RH*RW)
   * bottom[1]: -> Kps Label  (Nroi,18,RH,RW)
   * bottom[2]: -> ROI Flags  (1,1,Nroi,18)
   */
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  /**
   * top[0]: -> loss (1)
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc MultinomialLogisticLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}

#endif
