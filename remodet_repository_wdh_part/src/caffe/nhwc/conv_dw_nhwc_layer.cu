#include <vector>
#include "caffe/nhwc/conv_dw_nhwc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DepthwiseConv2DNHWCWeightForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int w = (index / channels) % top_width;
    const int h = (index / channels / top_width) % top_height;
    const int n = index / channels / top_width / top_height;
    // current weights
    const Dtype* weight = weight_data + c;
    // current input data
    const Dtype* bottom_base_data = bottom_data + n * bottom_height * bottom_width * channels + c;
    Dtype value = 0;
    for (int fh = 0; fh < kernel_h; ++fh) {
      for (int fw = 0; fw < kernel_w; ++fw) {
        const int h_in = -pad_h + h * stride_h + fh;
        const int w_in = -pad_w + w * stride_w + fw;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset = (h_in * bottom_width + w_in) * channels;
          value += (*weight) * bottom_base_data[offset];
        }
        weight += channels;
      }
    }
    top_data[index] = value;
  }
}

template <typename Dtype>
__global__ void DepthwiseConv2DNHWCBiasForward(const int nthreads,
    const Dtype* const bias_data, const int channels, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    top_data[index] += bias_data[c];
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  const int num = top[0]->shape(0);
  const int channels = top[0]->shape(3);
  const int top_height = top[0]->shape(1);
  const int top_width = top[0]->shape(2);
  const int bottom_height = bottom[0]->shape(1);
  const int bottom_width = bottom[0]->shape(2);
  DepthwiseConv2DNHWCWeightForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, weight_data, num, channels, top_height, top_width, bottom_height,
    bottom_width, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
  if (has_bias_) {
    const Dtype* bias_data = this->blobs_[1]->gpu_data();
    DepthwiseConv2DNHWCBiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bias_data, channels, top_data);
  }
}

template <typename Dtype>
__global__ void DepthwiseConv2DWeightBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const bottom_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, Dtype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % top_width;
    const int h = (index / top_width) % top_height;
    const int kw = (index / num / channels / top_height / top_width) % kernel_w;
    const int kh = (index / kernel_w / num / channels / top_height / top_width) % kernel_h;
    const int h_in = -pad_h + h * stride_h + kh;
    const int w_in = -pad_w + w * stride_w + kw;
    if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
      const int c = (index / top_height / top_width / num) % channels;
      const int n = (index / top_height / top_width) % num;
      const int top_offset = ((n * top_height + h) * top_width + w) * channels + c;
      const int bottom_offset = ((n * bottom_height + h_in) * bottom_width + w_in) * channels + c;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void DepthwiseConv2DBottomBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int iw = (index / channels) % bottom_width;
    const int ih = (index / channels / bottom_width) % bottom_height;
    const int n = index / channels / bottom_width / bottom_height;
    const Dtype* top_diff_ptr = top_diff + n * top_height * top_width * channels + c;
    const Dtype* weight_data_ptr = weight_data + c;
    Dtype value = 0;
    for (int fh = 0; fh < kernel_h; ++fh) {
      for (int fw = 0; fw < kernel_w; ++fw) {
        const int h_out_s = ih + pad_h - fh;
        const int w_out_s = iw + pad_w - fw;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int top_offset = (h_out * top_width + w_out) * channels;
            const int weight_offset = (fh * kernel_w + fw) * channels;
            value += weight_data_ptr[weight_offset] * top_diff_ptr[top_offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}

template <typename Dtype>
void ConvolutionDepthwiseNHWCLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const int bottom_count = bottom[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  const int length = num * top_height * top_width;
  caffe_gpu_set(bottom_count, Dtype(0), bottom[0]->mutable_gpu_diff());
  // bias
  if (has_bias_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    const Dtype* bias_multiplier_data = bias_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasTrans, channels, length, Dtype(1), top_diff, bias_multiplier_data, Dtype(1), bias_diff);
  }
  // weights
  if (this->param_propagate_down_[0]) {
    const int weight_buffer_count = weight_buffer_.count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_buffer_mutable_data = weight_buffer_.mutable_gpu_data();
    DepthwiseConv2DWeightBackward<Dtype><<<CAFFE_GET_BLOCKS(weight_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
      weight_buffer_count,top_diff,bottom_data,num,channels,top_height,top_width,bottom_height,
      bottom_width,kernel_h_,kernel_w_,stride_h_,stride_w_,pad_h_,pad_w_,weight_buffer_mutable_data);
    const int weight_count = this->blobs_[0]->count();
    const Dtype* weight_buffer_data = weight_buffer_.gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* weight_multiplier_data = weight_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasNoTrans, weight_count, length, Dtype(1), weight_buffer_data, weight_multiplier_data, Dtype(1), weight_diff);
  }
  // input
  if (propagate_down[0]) {
    const Dtype* weight_data = this->blobs_[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    DepthwiseConv2DBottomBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count,top_diff,weight_data,num,channels,top_height,top_width,bottom_height,bottom_width,
      kernel_h_,kernel_w_,stride_h_,stride_w_,pad_h_,pad_w_,bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionDepthwiseNHWCLayer);

}
