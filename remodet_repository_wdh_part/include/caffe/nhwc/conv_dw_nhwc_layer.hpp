#ifndef CAFFE_NHWC_CONV_DW_LAYER_HPP_
#define CAFFE_NHWC_CONV_DW_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
// we use Eigen to accelerate the DepthwiseConv2D computation
#include <Eigen/Dense>

/**
 * NOTE: 只支持通道完全分离的卷积，即:输入通道＝输出通道
 * 建议：通道数为４的倍数会加快运行速度
 */

namespace caffe {

template <typename Dtype>
class ConvolutionDepthwiseNHWCLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionDepthwiseNHWCLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "ConvolutionDepthwiseNHWC"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // NOTE: it's only CPU implemention, GPU use CUDA Kernels to accelerate the computation.
  void FilterPadKernel(const Dtype* filter, Dtype* padded_filter, const int channels, const int filter_spatial_dim);
  void InputCopyKernel(const Dtype* input, Dtype* input_buffer, const int channels, const int height, const int width,
                       const int out_r, const int out_c, const int stride_h, const int stride_w, const int pad_h,
                       const int pad_w, const int kernel_h, const int kernel_w);
  void DepthwiseConv2DKernel(const Dtype* filter, const Dtype* input_buffer, Dtype* output,
                             const int channels, const int out_h, const int out_w, const int top_height,
                             const int top_width, const int filter_spatial_dim, const int padded_size);
  bool has_bias_;
  unsigned int kernel_h_;
  unsigned int kernel_w_;
  unsigned int stride_h_;
  unsigned int stride_w_;
  unsigned int pad_h_;
  unsigned int pad_w_;
  unsigned int dilation_h_;
  unsigned int dilation_w_;
  unsigned int num_output_;
  unsigned int channels_;
  // for input copy & filter pad
  Blob<Dtype> padded_filter_;
  Blob<Dtype> padded_input_buf_;
  // for bias operation
  Blob<Dtype> bias_multiplier_;
  // weight operation
  Blob<Dtype> weight_buffer_;
  Blob<Dtype> weight_multiplier_;
};

}

#endif
