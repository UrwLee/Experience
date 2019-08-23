#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CrossEntropyLossKernel(const int nthreads, const Dtype thre_min, const Dtype thre_max,
          const Dtype* bottom_data, const Dtype* bottom_label, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype val = bottom_data[index];
    if (bottom_data[index] < thre_min)
      val = thre_min;
    else if (bottom_data[index] > thre_max)
      val = thre_max;
    else;
    Dtype temp = bottom_label[index] * log(val) + (1.0 - bottom_label[index]) * log (1.0 - val);
    loss[index] = -temp;
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  Dtype loss = 0;
  Blob<Dtype> loss_elements;
  loss_elements.ReshapeLike(*bottom[0]);
  CrossEntropyLossKernel<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
    count, kLOG_THRESHOLD, 1.0 - kLOG_THRESHOLD, bottom_data, bottom_label,
    loss_elements.mutable_gpu_data());
  const Dtype* loss_ptr = loss_elements.cpu_data();
  for (int i = 0; i < count; ++i) {
    loss += loss_ptr[i];
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
__global__ void CrossEntropyLossBPKernel(const int nthreads, const Dtype thre_min, const Dtype thre_max,
          const Dtype* bottom_data, const Dtype* bottom_label, const Dtype scale, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype val = bottom_data[index];
    if (bottom_data[index] < thre_min)
      val = thre_min;
    else if (bottom_data[index] > thre_max)
      val = thre_max;
    else;
    diff[index] = scale * (bottom_label[index] / val - (1.0 - bottom_label[index]) / (1.0 - val));
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    CrossEntropyLossBPKernel<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
      count, kLOG_THRESHOLD, 1.0 - kLOG_THRESHOLD, bottom_data, bottom_label, scale, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEntropyLossLayer);

}  // namespace caffe
