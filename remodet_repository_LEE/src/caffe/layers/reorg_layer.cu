#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReorgKernel(const int nthreads,
    Dtype* const bottom_data, const bool forward,
    const int num, const int channels, const int height,
    const int width, const int stride, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (index >= nthreads) return;
    int in_w = index % width;
    int part = index / width;
    int in_h = part % height;
    part = part / height;
    int in_c = part % channels;
    part = part / channels;
    int in_b = part % num;

    int out_channels = channels * stride * stride;
    int out_height = height / stride;
    int out_width = width / stride;
    int out_w = in_w / stride;
    int out_h = in_h / stride;
    int out_c = (in_h % stride) * stride + in_w % stride + in_c * stride * stride;
    int out_idx = ((in_b * out_channels + out_c) * out_height
                          + out_h) * out_width + out_w;
    if (forward)
      top_data[out_idx] = bottom_data[index];
    else
      top_data[index] = bottom_data[out_idx];
  }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  CHECK_EQ(bottom[0]->count(), top[0]->count())
    << "bottom and top blobs should have the same length.";

  if (up_down_ == ReorgParameter_SampleType_DOWN) {
    const vector<int>& shape = bottom[0]->shape();
    const int count = top[0]->count();
    bool forward = true;
    ReorgKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, forward, shape[0], shape[1], shape[2],
      shape[3], stride_, top_data);
  } else if (up_down_ == ReorgParameter_SampleType_UP) {
    const vector<int>& shape = top[0]->shape();
    const int count = top[0]->count();
    bool forward = false;
    ReorgKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, forward, shape[0], shape[1], shape[2],
      shape[3], stride_, top_data);
  } else {
    LOG(FATAL) << "Unknown Reorg SampleType.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    CHECK_EQ(bottom[0]->count(), top[0]->count())
      << "bottom and top blobs should have the same length.";

    if (up_down_ == ReorgParameter_SampleType_DOWN) {
      const vector<int>& shape = bottom[0]->shape();
      const int count = top[0]->count();
      bool forward = false;
      ReorgKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, forward, shape[0], shape[1], shape[2],
        shape[3], stride_, bottom_diff);
    } else if (up_down_ == ReorgParameter_SampleType_UP) {
      const vector<int>& shape = top[0]->shape();
      const int count = top[0]->count();
      bool forward = true;
      ReorgKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, forward, shape[0], shape[1], shape[2],
        shape[3], stride_, bottom_diff);
    } else {
      LOG(FATAL) << "Unknown Reorg SampleType.";
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);

}  // namespace caffe
