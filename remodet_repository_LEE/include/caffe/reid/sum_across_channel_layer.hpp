#ifndef CAFFE_REID_SUM_ACROSSCHANNEL_LAYER_HPP_
#define CAFFE_REID_SUM_ACROSSCHANNEL_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层已停止使用，禁止使用该层。
 * 该层作用：对特征图(Feature Maps)沿着通道方向进行累加。
 * Input(x): [N,C,H,W] -> Output(y): [N,1,H,W]
 * y(n,0,i,j) = SigmaByChannels:x(n,c,i,j) from all c = 0,1,2,...,C-1
 */

template <typename Dtype>
class SumAcrossChanLayer : public Layer<Dtype> {
 public:
  explicit SumAcrossChanLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SumAcrossChan"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif
