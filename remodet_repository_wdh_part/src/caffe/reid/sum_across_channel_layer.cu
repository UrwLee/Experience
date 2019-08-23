#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/reid/sum_across_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumAcrossChanKernel(const int nthreads,
    Dtype* bottom_data, const int channels, const int height,
    const int width, const int offs_item, const int offs_chan, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % width;
    const int i = (index / width) % height;
    const int num = index / width / height;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += bottom_data[num*offs_item + c*offs_chan + i*width + j];
    }
    top_data[index] = sum;
  }
}

template <typename Dtype>
void SumAcrossChanLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int offs_item = channels * height * width;
  const int offs_chan = height * width;

  const int count = num * offs_chan;

  SumAcrossChanKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, channels, height, width, offs_item, offs_chan, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SumAcrossChanLayer);

}  // namespace caffe
