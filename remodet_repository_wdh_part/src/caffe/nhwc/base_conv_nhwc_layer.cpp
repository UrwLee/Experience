#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/nhwc/base_conv_nhwc_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
namespace caffe {

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK_EQ(bottom[0]->num_axes(),4) << "NHWC only supports 4 shape-dim.";
  vector<int> kernel_shape_size(1,2);
  kernel_shape_.Reshape(kernel_shape_size);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == 2);
    for (int i = 0; i < 2; ++i) {
      kernel_shape_data[i] =
          conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int i = 0; i < 2; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(kernel_shape_size);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == 2);
    const int kDefaultStride = 1;
    for (int i = 0; i < 2; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(kernel_shape_size);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == 2);
    const int kDefaultPad = 0;
    for (int i = 0; i < 2; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(kernel_shape_size);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == 2);
  const int kDefaultDilation = 1;
  for (int i = 0; i < 2; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  is_1x1_ = true;
  for (int i = 0; i < 2; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  // [NHWC]
  channels_ = bottom[0]->shape(3);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  // DO NOT USE GROUP CONV, USE
  CHECK_EQ(group_,1) << "Please use ConvolutionDepthwiseNHWCLayer instead.";
  // CHECK_EQ(channels_ % group_, 0);
  // CHECK_EQ(num_output_ % group_, 0)
      // << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // weights [D,D,Nc,No]
  vector<int> weight_shape;
  for (int i = 0; i < 2; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  weight_shape.push_back(conv_in_channels_);
  weight_shape.push_back(conv_out_channels_);
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(1, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // [D,D,Nc]
  kernel_dim_ = kernel_shape_data[0] * kernel_shape_data[1] * conv_in_channels_;
  // weight [D,D,Nc,No]
  weight_offset_ = conv_out_channels_ * kernel_dim_;
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->shape(0);
  CHECK_EQ(bottom[0]->shape(3), channels_) << "Input channels must keep unchanged.";
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(1,num_);
  for (int i = 0; i < 2; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  top_shape.push_back(num_output_);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    // IH * IW
    conv_out_spatial_dim_ = (*bottom_shape_)[1] * (*bottom_shape_)[2];
  } else {
    // OH * OW
    conv_out_spatial_dim_ = output_shape_[0] * output_shape_[1];
  }
  // col_offset_ = [H,W,D,D,Nc], rsvd for group
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  // output_offset_ = No*H*W, rsvd for group
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, 3);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  // input conv shape [H,W,C]
  for (int i = 0; i < 3; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(1 + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(1 + i);
    }
  }
  col_buffer_shape_.clear();
  // [H,W,D,D,Nc]
  for (int i = 0; i < 2; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_shape_.push_back(kernel_dim_);
  col_buffer_.Reshape(col_buffer_shape_);
  // bottom_dim_ = H * W * Nc
  bottom_dim_ = bottom[0]->count(1);
  // top_dim_ = H * W * No
  top_dim_ = top[0]->count(1);
  // H * W * Nc
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  // H * W
  out_spatial_dim_ = output_shape_[0] * output_shape_[1];
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      // col_buffer_ -> [H,W,D,D,Nc]
      // caffe::Timer im_col;
      // im_col.Start();
      conv_nhwc_im2col_cpu(input, col_buffer_.mutable_cpu_data());
      // LOG(INFO) << "[IM2COL]: " << im_col.MicroSeconds();
    }
    col_buff = col_buffer_.cpu_data();
  }
  // caffe::Timer gemm;
  // gemm.Start();
  // for (int g = 0; g < group_; ++g) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_spatial_dim_,
      conv_out_channels_, kernel_dim_, (Dtype)1., col_buff, weights,
      (Dtype)0., output);
  // }
  // LOG(INFO) << "[SGEMM]: " << gemm.MicroSeconds();
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, out_spatial_dim_,
      num_output_, 1, (Dtype)1., bias_multiplier_.cpu_data(), bias,
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  // for (int g = 0; g < group_; ++g) {
  // CblasNoTrans,CblasTrans
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_spatial_dim_,
      kernel_dim_, conv_out_channels_, (Dtype)1., output, weights,
        (Dtype)0., col_buff);
  // }
  if (!is_1x1_) {
    conv_nhwc_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_nhwc_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  // for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_channels_,
        conv_out_spatial_dim_, (Dtype)1., col_buff, output,
        (Dtype)1., weights);
  // }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      // caffe::Timer im_col;
      // im_col.Start();
      conv_nhwc_im2col_gpu(input, col_buffer_.mutable_gpu_data());
      // LOG(INFO) << "[IM2COL]: " << im_col.MicroSeconds();
    }
    col_buff = col_buffer_.gpu_data();
  }
  // for (int g = 0; g < group_; ++g) {
  // caffe::Timer gemm;
  // gemm.Start();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_spatial_dim_,
        conv_out_channels_, kernel_dim_, (Dtype)1., col_buff, weights,
        (Dtype)0., output);
  // LOG(INFO) << "[SGEMM]: " << gemm.MicroSeconds();
  // }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, out_spatial_dim_,
      num_output_, 1, (Dtype)1., bias_multiplier_.gpu_data(), bias,
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  // for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_spatial_dim_,
      kernel_dim_, conv_out_channels_, (Dtype)1., output, weights,
        (Dtype)0., col_buff);
  // }
  if (!is_1x1_) {
    conv_nhwc_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_nhwc_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  // for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_channels_,
        conv_out_spatial_dim_, (Dtype)1., col_buff, output,
        (Dtype)1., weights);
  // }
}

template <typename Dtype>
void BaseConvolutionNHWCLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionNHWCLayer);

}  // namespace caffe
