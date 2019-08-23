#include "caffe/tracker/halfmerge_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <vector>

namespace caffe {

template <typename Dtype>
__global__ void MergeKernel(const int nthreads,
    Dtype* bottom_data, const bool forward,
    const int N, const int C, const int H,
    const int W, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (index >= nthreads) return;
    int j = index % W;
    int part = index / W;
    int i = part % H;
    part = part / H;
    int c = part % C;
    part = part / C;
    int n = part % N;

    int HN = N / 2;
    int on = n % HN;
    int oc = C * (n / HN) + c;
    int out_index = ((on*2*C+oc)*H+i)*W+j;
    if (forward)
      top_data[out_index] = bottom_data[index];
    else
      bottom_data[index] = top_data[out_index];
  }
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CHECK_EQ(bottom[0]->count(), top[0]->count())
    << "bottom and top blobs should have the same length.";
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  MergeKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count,bottom_data,true,num,channels,height,width,top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    CHECK_EQ(bottom[0]->count(), top[0]->count())
      << "bottom and top blobs should have the same length.";
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    MergeKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count,bottom_diff,false,num,channels,height,width,top_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HalfmergeLayer);

}  // namespace caffe
