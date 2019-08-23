#include <vector>

#include "caffe/mask/mask_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskSigmoidCrossEntropyBackward(
     const int count, Dtype* diff, const Dtype* flags, const Dtype scale,
     const int offsize) {
  CUDA_KERNEL_LOOP(index, count) {
    const int id = index / offsize;
    const int flag = flags[id];
    if (flag > 0) {
      // normal
      diff[index] *= scale;
    } else {
      // back zero
      diff[index] = 0;
    }
  }
}

template <typename Dtype>
void MaskSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //              << " Layer cannot backpropagate to label inputs.";
  // }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int offsize = bottom[0]->height() * bottom[0]->width();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    const Dtype* flags = bottom[2]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype scale = loss_weight * scale_ / num;
    MaskSigmoidCrossEntropyBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_diff, flags, scale, offsize);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(MaskSigmoidCrossEntropyLossLayer);
}
