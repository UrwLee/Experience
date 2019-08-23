#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/mask/mask_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskCrossEntropyLossBPKernel(const int nthreads, const Dtype thre_min, const Dtype thre_max,
          const Dtype* bottom_data, const Dtype* bottom_label, const Dtype* flags, const int offsize,
          const Dtype scale, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int id = index / offsize;
    int flag = flags[id];
    if (flag > 0) {
      Dtype pred = bottom_data[index];
      pred = min(max(pred,thre_min),thre_max);
      diff[index] = scale * (bottom_label[index] / pred - (1.0 - bottom_label[index]) / (1.0 - pred));
    } else {
      diff[index] = 0;
    }
  }
}

template <typename Dtype>
void MaskCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //              << " Layer cannot backpropagate to label inputs.";
  // }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->gpu_data();
    const Dtype* flags = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int offsize = bottom[0]->count() / num / channels;
    Dtype scale_temp = this->layer_param_.mask_loss_param().scale();
    const Dtype scale = - top[0]->cpu_diff()[0] / num * scale_temp;
    const int count = bottom[0]->count();
    MaskCrossEntropyLossBPKernel<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
      count, kLOG_THRESHOLD, 1.0 - kLOG_THRESHOLD, bottom_data, bottom_label, flags, offsize,
      scale, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(MaskCrossEntropyLossLayer);
}
