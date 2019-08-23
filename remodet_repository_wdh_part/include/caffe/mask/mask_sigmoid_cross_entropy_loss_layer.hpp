#ifndef CAFFE_MASK_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_MASK_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * 该层提供了带Mask的sigmoid交叉熵损失层。
 * 方法如下：
 * １．如果对应于(H,W)的标记为false，则这个Map的所有误差不进行传播，且不计入损失。
 */

template <typename Dtype>
class MaskSigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit MaskSigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskSigmoidCrossEntropyLoss"; }

  /**
   * bottom[0]: -> Mask preds (Nroi,1,RH,RW)
   * bottom[1]: -> Mask GTMap (Nroi,1,RH,RW)
   * bottom[2]: -> Active flags for each map (1,1,Nroi,1)
   * if active_flag = 1 -> 误差正常反馈和计算
   * if active_flag = 0 -> 误差清零，无反向传播
   */
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  /**
   * top[0]: -> loss
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// 内部封装的sigmoid_layer用于计算sigmoid函数
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid层的输出
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// sigmoid层的bottom列表
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// sigmoid层的top列表
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  // 误差传播的增益系数
  Dtype scale_;
};

}

#endif
