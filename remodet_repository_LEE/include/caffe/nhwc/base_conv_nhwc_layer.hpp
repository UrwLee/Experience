#ifndef CAFFE_NHWC_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_NHWC_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
class BaseConvolutionNHWCLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionNHWCLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[1 + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int bottom_dim_;
  int top_dim_;

  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;

 private:
  inline void conv_nhwc_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    im2col_nhwc_cpu(data, conv_in_channels_,
        conv_input_shape_.cpu_data()[0], conv_input_shape_.cpu_data()[1],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
  }
  inline void conv_nhwc_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    col2im_nhwc_cpu(col_buff, conv_in_channels_,
        conv_input_shape_.cpu_data()[0], conv_input_shape_.cpu_data()[1],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
  }
#ifndef CPU_ONLY
  inline void conv_nhwc_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    im2col_nhwc_gpu(data, conv_in_channels_,
        conv_input_shape_.cpu_data()[0], conv_input_shape_.cpu_data()[1],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
  }
  inline void conv_nhwc_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    col2im_nhwc_gpu(col_buff, conv_in_channels_,
        conv_input_shape_.cpu_data()[0], conv_input_shape_.cpu_data()[1],
        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        pad_.cpu_data()[0], pad_.cpu_data()[1],
        stride_.cpu_data()[0], stride_.cpu_data()[1],
        dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}

#endif
