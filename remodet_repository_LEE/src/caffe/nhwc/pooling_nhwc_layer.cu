#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/nhwc/pooling_nhwc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolNHWCForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const int input_base = n * height;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int input_offset = ((input_base + h) * width + w) * channels + c;
        if (bottom_data[input_offset] > maxval) {
          maxidx = input_offset;
          maxval = bottom_data[input_offset];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void AvePoolNHWCForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const int input_base = n * height;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int input_offset = ((input_base + h) * width + w) * channels + c;
        aveval += bottom_data[input_offset];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
void PoolingNHWCLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX: {
      int* mask = max_idx_.mutable_gpu_data();
      MaxPoolNHWCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->shape(0), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, mask);
    }
    break;
    case PoolingParameter_PoolMethod_AVE: {
      AvePoolNHWCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->shape(0), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    }
    break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolNHWCBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int w = (index / channels) % width;
    const int h = (index / channels / width) % height;
    const int n = index / channels / width / height;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int top_base = n * pooled_height;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        const int top_offset = ((top_base + ph) * pooled_width + pw) * channels + c;
        if (mask[top_offset] == index) {
          gradient += top_diff[top_offset];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolNHWCBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int w = (index / channels) % width;
    const int h = (index / channels / width) % height;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int top_base = n * pooled_height;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        const int top_offset = ((top_base + ph) * pooled_width + pw) * channels + c;
        gradient += top_diff[top_offset] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingNHWCLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX: {
      const int* mask = max_idx_.gpu_data();
      MaxPoolNHWCBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, top[0]->shape(0), channels_, height_, width_,
          pooled_height_, pooled_width_, kernel_h_, kernel_w_, stride_h_,
          stride_w_, pad_h_, pad_w_, bottom_diff);
    }
    break;
    case PoolingParameter_PoolMethod_AVE: {
      AvePoolNHWCBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, top[0]->shape(0), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    }
    break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PoolingNHWCLayer);
}  // namespace caffe
