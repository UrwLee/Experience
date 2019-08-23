#ifndef CAFFE_TRACKER_HALFMERGE_LAYER_HPP_
#define CAFFE_TRACKER_HALFMERGE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层用于对双通道样本进行合并。
 * 计算方法：
 * Input(x): [2N,C,H,W] -> Output(y): [N,2C,H,W]
 * y(n,C+l,i,j) = x(N+n,l,i,j)
 * 该层在Tracker中使用，用于对prev/curr特征进行融合。
 */

template <typename Dtype>
void Merge(Dtype* bottom_data, const vector<int> shape, const bool forward,
           Dtype* top_data);

template <typename Dtype>
class HalfmergeLayer : public Layer<Dtype> {
 public:
  explicit HalfmergeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Halfmerge"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif
