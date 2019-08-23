#ifndef CAFFE_MASK_GRAD_CLIP_LAYER_HPP_
#define CAFFE_MASK_GRAD_CLIP_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层提供了对反向误差进行裁剪的方法。 (注意：该层暂时未使用)
 * 前向计算：不执行任何计算，直接令输出＝输入
 * 反向计算：根据Flags：
 * 　　　　　为正，则执行error_o = scale * error_i
 * 　　　　　为负，则执行error_o = 0
 */

template <typename Dtype>
class GradClipLayer : public Layer<Dtype> {
 public:
  explicit GradClipLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GradClip"; }
  // bottom[0] -> 输入特征Blob (N,C,H,W)
  // bottom[1] -> flags       (1,1,N,C)
  // 每个[H,W]都有一个控制flag
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // top[0] -> 输出特征Blob  (N,C,H,W)
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

  // 误差增益
  Dtype scale_;
};

}  // namespace caffe

#endif
