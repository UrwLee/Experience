#ifndef CAFFE_NHWC_DECONV_LAYER_HPP_
#define CAFFE_NHWC_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/nhwc/base_conv_nhwc_layer.hpp"

namespace caffe {

template <typename Dtype>
class DeconvolutionNHWCLayer : public BaseConvolutionNHWCLayer<Dtype> {
 public:
  explicit DeconvolutionNHWCLayer(const LayerParameter& param)
      : BaseConvolutionNHWCLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "DeconvolutionNHWC"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

}

#endif
