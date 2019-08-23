#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/nhwc/conv_dw_nhwc_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    CHECK_GT(conv_param.kernel_size_size(), 0) << "Must provide kernel_size.";
    if (conv_param.kernel_size_size() == 1) {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(0);
    } else {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(1);
    }
  }
  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    CHECK_GT(conv_param.stride_size(), 0) << "Must provide stride_size.";
    if (conv_param.stride_size() == 1) {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(0);
    } else {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(1);
    }
  }
  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    CHECK_GT(conv_param.pad_size(), 0) << "Must provide pad_size.";
    if (conv_param.pad_size() == 1) {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(0);
    } else {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(1);
    }
  }
  if (conv_param.dilation_size() > 0) {
    if (conv_param.dilation_size() == 1) {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(0);
    } else {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(1);
    }
  } else {
    dilation_h_ = 1;
    dilation_w_ = 1;
  }
  CHECK_EQ(dilation_h_,1) << "dilation size must be 1.";
  CHECK_EQ(dilation_w_,1) << "dilation size must be 1.";
  channels_ = bottom[0]->shape(3);
  if (conv_param.has_num_output()) {
    num_output_ = conv_param.num_output();
  } else {
    num_output_ = channels_;
  }
  CHECK_EQ(num_output_, channels_);
  vector<int> weight_shape;
  weight_shape.push_back(kernel_h_);
  weight_shape.push_back(kernel_w_);
  weight_shape.push_back(1);
  weight_shape.push_back(channels_);
  has_bias_ = conv_param.bias_term();
  vector<int> bias_shape;
  if (has_bias_) {
    bias_shape.push_back(channels_);
  }
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + has_bias_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (has_bias_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (has_bias_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (has_bias_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  // output reshape
  CHECK_EQ(channels_, bottom[0]->shape(3));
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->shape(0));
  top_shape.push_back((bottom[0]->shape(1) + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1);
  top_shape.push_back((bottom[0]->shape(2) + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1);
  top_shape.push_back(channels_);
  top[0]->Reshape(top_shape);
  // weight_multiplier
  // bias multiplier : [H * W]
  vector<int> bias_shape(1, top[0]->shape(1) * top[0]->shape(2));
  bias_multiplier_.Reshape(bias_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
  // padded_filter & padded_input_buf_
  const bool pad_filter = channels_ % kPacketSize == 0 ? false : true;
  const int filter_spatial_dim = kernel_h_ * kernel_w_;
  const int pad_size = pad_filter ? ((channels_ + kPacketSize - 1) / kPacketSize) * kPacketSize : channels_;
  vector<int> pad_shape(1,filter_spatial_dim * pad_size);
  if (pad_filter) {
    padded_filter_.Reshape(pad_shape);
    FilterPadKernel(this->blobs_[0]->cpu_data(), padded_filter_.mutable_cpu_data(), channels_, filter_spatial_dim);
  }
  padded_input_buf_.Reshape(pad_shape);
  // bias_multiplier_
  if (has_bias_) {
    vector<int> bias_multiplier_shape;
    bias_multiplier_shape.push_back(top[0]->shape(0));
    bias_multiplier_shape.push_back(top[0]->shape(1));
    bias_multiplier_shape.push_back(top[0]->shape(2));
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_gpu_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_gpu_data());
  }
  // weights buffer
  // [D,D,1,C,N,H,W]
  vector<int> weight_buffer_shape;
  weight_buffer_shape.push_back(kernel_h_);
  weight_buffer_shape.push_back(kernel_w_);
  weight_buffer_shape.push_back(1);
  weight_buffer_shape.push_back(channels_);
  weight_buffer_shape.push_back(top[0]->shape(0));
  weight_buffer_shape.push_back(top[0]->shape(1));
  weight_buffer_shape.push_back(top[0]->shape(2));
  weight_buffer_.Reshape(weight_buffer_shape);
  // [N,H,W]
  vector<int> weight_multiplier_shape;
  weight_multiplier_shape.push_back(top[0]->shape(0));
  weight_multiplier_shape.push_back(top[0]->shape(1));
  weight_multiplier_shape.push_back(top[0]->shape(2));
  weight_multiplier_.Reshape(weight_multiplier_shape);
  caffe_gpu_set(weight_multiplier_.count(), Dtype(1), weight_multiplier_.mutable_gpu_data());
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::FilterPadKernel(const Dtype* filter, Dtype* padded_filter,
                                                  const int channels, const int filter_spatial_dim) {
  // Weights -> [D,D,1,C+Pad]
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  const int vectorized_size = (channels / kPacketSize) * kPacketSize;
  const int scalar_size = channels - vectorized_size;
  const int pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
  const int padded_size = channels + pad_size;
  for (int i = 0; i < filter_spatial_dim; ++i) {
    const int input_base = i * channels;
    const int output_base = i * padded_size;
    // write vectorized_size
    for (int j = 0; j < vectorized_size; j += kPacketSize) {
      const Packet v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
      Eigen::internal::pstoreu<Dtype>(padded_filter + output_base + j, v);
    }
    // write scalar
    for (int j = 0; j < scalar_size; ++j) {
      padded_filter[output_base + vectorized_size + j] =
          filter[input_base + vectorized_size + j];
    }
    // write pad
    for (int j = 0; j < pad_size; ++j) {
      padded_filter[output_base + vectorized_size + scalar_size + j] = 0;
    }
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::InputCopyKernel(const Dtype* input, Dtype* input_buffer,
                      const int channels, const int height, const int width, const int out_h,
                      const int out_w, const int stride_h, const int stride_w, const int pad_h,
                      const int pad_w, const int kernel_h, const int kernel_w) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  // Inputs -> [H,W,C]
  // Input_buffer -> [D,D,C+Pad]
  const int vectorized_size = (channels / kPacketSize) * kPacketSize;
  const int scalar_size = channels % kPacketSize;
  const int pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
  const int padded_size = channels + pad_size;
  // copy
  const int in_h_start = out_h * stride_h - pad_h;
  const int in_w_start = out_w * stride_w - pad_w;
  for (int fh = 0; fh < kernel_h; ++fh) {
    const int in_h = in_h_start + fh;
    for (int fw = 0; fw < kernel_w; ++fw) {
      const int in_w = in_w_start + fw;
      if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
        const Dtype* in = input + (in_h * width + in_w) * channels;
        // write vectorized_size
        for (int d = 0; d < vectorized_size; d += kPacketSize) {
          const Packet v = Eigen::internal::ploadu<Packet>(in + d);
          Eigen::internal::pstoreu<Dtype>(input_buffer + d, v);
        }
        // write scalar
        for (int d = 0; d < scalar_size; ++d) {
          input_buffer[vectorized_size + d] = in[vectorized_size + d];
        }
        // write pad
        for (int d = 0; d < pad_size; ++d) {
          input_buffer[vectorized_size + scalar_size + d] = 0;
        }
      } else {
        memset(input_buffer, 0, sizeof(Dtype) * padded_size);
      }
      // pointer to next location of kernel spatial space.
      input_buffer += padded_size;
    }
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::DepthwiseConv2DKernel(const Dtype* filter,
                        const Dtype* input_buffer, Dtype* output, const int channels,
                        const int out_h, const int out_w, const int top_height,
                        const int top_width, const int filter_spatial_dim,
                        const int padded_size) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  const int vectorized_size = (channels / kPacketSize) * kPacketSize;
  const int scalar_size = channels % kPacketSize;

  const int output_base_index = (out_h * top_width + out_w) * channels;
  // compute vectorized_size
  for (int i = 0; i < vectorized_size; i += kPacketSize) {
    Packet vaccum = Eigen::internal::pset1<Packet>(0);
    for (int j = 0; j < filter_spatial_dim; ++j) {
      const int index = i + j * padded_size;
      const Packet filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      const Packet data_block = Eigen::internal::ploadu<Packet>(input_buffer + index);
      vaccum = Eigen::internal::pmadd<Packet>(filter_block,data_block,vaccum);
    }
    Eigen::internal::pstoreu<Dtype>(output + output_base_index + i, vaccum);
  }
  // compute scalar
  if (scalar_size > 0) {
    Packet vaccum = Eigen::internal::pset1<Packet>(0);
    for (int j = 0; j < filter_spatial_dim; ++j) {
      const int index = vectorized_size + j * padded_size;
      const Packet filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      const Packet data_block = Eigen::internal::ploadu<Packet>(input_buffer + index);
      vaccum = Eigen::internal::pmadd<Packet>(filter_block,data_block,vaccum);
    }
    Dtype out_buf[kPacketSize];
    Eigen::internal::pstoreu<Dtype>(out_buf, vaccum);
    const int last_output_index = output_base_index + vectorized_size;
    for (int j = 0; j < scalar_size; ++j) {
      output[last_output_index + j] = out_buf[j];
    }
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  const bool pad_filter = channels_ % kPacketSize == 0 ? false : true;
  const int filter_spatial_dim = kernel_h_ * kernel_w_;
  const int pad_size = pad_filter ? ((channels_ + kPacketSize - 1) / kPacketSize) * kPacketSize : channels_;
  const Dtype* filter_data = pad_filter ? padded_filter_.cpu_data() : this->blobs_[0]->cpu_data();
  const int num = bottom[0]->shape(0);
  const int height = bottom[0]->shape(1);
  const int width = bottom[0]->shape(2);
  const int top_height = top[0]->shape(1);
  const int top_width = top[0]->shape(2);
  const int top_dim = top[0]->count(1);
  const int bottom_dim = bottom[0]->count(1);
  const Dtype* input = bottom[0]->cpu_data();
  Dtype* output = top[0]->mutable_cpu_data();
  // do conv
  for (int n = 0; n < num; ++n) {
    const Dtype* input_ptr = input + n * bottom_dim;
    Dtype* output_ptr = output + n * top_dim;
    for (int h = 0; h < top_height; ++h) {
      for (int w = 0; w < top_width; ++w) {
        InputCopyKernel(input_ptr, padded_input_buf_.mutable_cpu_data(),
                        channels_, height, width, h, w, stride_h_, stride_w_, pad_h_,
                        pad_w_, kernel_h_, kernel_w_);
        DepthwiseConv2DKernel(filter_data, padded_input_buf_.cpu_data(), output_ptr,
                        channels_, h, w, top_height, top_width, filter_spatial_dim, pad_size);
      }
    }
  }
  // do bias
  if (this->layer_param_.convolution_param().bias_term()) {
    Dtype* top_out = top[0]->mutable_cpu_data();
    const Dtype* bias = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, top[0]->shape(1) * top[0]->shape(2),
        channels_, 1, (Dtype)1., bias_multiplier_.cpu_data(), bias,
        (Dtype)1., top_out);
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0),bottom_diff);
  const int num = bottom[0]->shape(0);
  const int out_spatial_dim = top[0]->shape(1) * top[0]->shape(2);
  const int top_dim = out_spatial_dim * channels_;
  const int top_height = top[0]->shape(1);
  const int top_width = top[0]->shape(2);
  const int input_spatial_dim = bottom[0]->shape(1) * bottom[0]->shape(2);
  const int bottom_dim = input_spatial_dim * channels_;
  const int bottom_height = bottom[0]->shape(1);
  const int bottom_width = bottom[0]->shape(2);
  // backward to bias
  if (has_bias_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int n = 0; n < num; ++n) {
      caffe_cpu_gemv<Dtype>(CblasTrans, channels_, out_spatial_dim, 1.,
          top_diff + n * top_dim, bias_multiplier_.cpu_data(), 1., bias_diff);
    }
  }
  // backward to weights & inputs
  if (this->param_propagate_down_[0] || propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      // backward to weights
      const Dtype* bottom_data_ptr = bottom_data + n * bottom_dim;
      const Dtype* top_diff_ptr = top_diff + n * top_dim;
      Dtype* bottom_diff_ptr = bottom_diff + n * bottom_dim;
      if (this->param_propagate_down_[0]) {
        for (int h = 0; h < top_height; ++h) {
          const int in_start_h = h * stride_h_ - pad_h_;
          for (int w = 0; w < top_width; ++w) {
            const int in_start_w = w * stride_w_ - pad_w_;
            for (int c = 0; c < channels_; ++c) {
              Dtype* weight_diff_ptr = weight_diff + c;
              // scan over all kernel spatial dim
              for (int fh = 0; fh < kernel_h_; ++fh) {
                const int in_h = in_start_h + fh;
                for (int fw = 0; fw < kernel_w_; ++fw) {
                  const int in_w = in_start_w + fw;
                  if (in_h >= 0 && in_w >= 0 && in_h < bottom_height && in_w < bottom_width) {
                    int bottom_offset = (in_h * bottom_width + in_w) * channels_ + c;
                    *weight_diff_ptr += bottom_data_ptr[bottom_offset] * (*top_diff_ptr);
                  }
                  weight_diff_ptr += channels_;
                }
              }
              top_diff_ptr++;
            }
          }
        }
      }
      // backward to inputs
      if (propagate_down[0]) {
        // repointer
        top_diff_ptr = top_diff + n * top_dim;
        for (int h = 0; h < top_height; ++h) {
          const int in_start_h = h * stride_h_ - pad_h_;
          for (int w = 0; w < top_width; ++w) {
            const int in_start_w = w * stride_w_ - pad_w_;
            for (int c = 0; c < channels_; ++c) {
              // pointer to weight [D,D,1,C]
              const Dtype* weight_data_ptr = weight + c;
              for (int fh = 0; fh < kernel_h_; ++fh) {
                const int in_h = in_start_h + fh;
                for (int fw = 0; fw < kernel_w_; ++fw) {
                  const int in_w = in_start_w + fw;
                  if (in_h >= 0 && in_w >= 0 && in_h < bottom_height && in_w < bottom_width) {
                    int bottom_offset = (in_h * bottom_width + in_w) * channels_ + c;
                    bottom_diff_ptr[bottom_offset] += (*weight_data_ptr) * (*top_diff_ptr);
                  }
                  weight_data_ptr += channels_;
                }
              }
              top_diff_ptr++;
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionDepthwiseNHWCLayer);
#endif

INSTANTIATE_CLASS(ConvolutionDepthwiseNHWCLayer);
REGISTER_LAYER_CLASS(ConvolutionDepthwiseNHWC);

}  // namespace caffe
