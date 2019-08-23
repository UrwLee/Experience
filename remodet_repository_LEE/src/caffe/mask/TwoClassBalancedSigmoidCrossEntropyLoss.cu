#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/mask/TwoClassBalancedSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// template <typename Dtype>
// __global__ void ForwardKernel(const int counts,
//           const Dtype* target, Dtype* out_data,
//           Dtype zn, Dtype zp, bool only_pos){
//   CUDA_KERNEL_LOOP(index, counts) {
//     Dtype loss1 = 0;

//     if (index % 2 != 0) {//class 1
//       loss1 -= (target[index] == 1) * out_data[index] * zp;
//       loss1 -= (target[index] == 0) * out_data[index] * zn;
//     } else if (!only_pos) { //class 0 use the opposite alpha
//       loss1 -= (target[index] == 1) * out_data[index] * zn;
//       loss1 -= (target[index] == 0) * out_data[index] * zp;
//     } else { // class 0 if only pos will not be balance
//       loss1 -= out_data[index];
//     }

//     out_data[index] = loss1;
//   }
// }

template <typename Dtype>
__global__ void BackwardKernel(const int counts, const Dtype* target,
                  Dtype* bottom_diff,Dtype zn, Dtype zp, bool only_pos) {
  CUDA_KERNEL_LOOP(index, counts) {

    Dtype d_logits = 0;
    Dtype term = bottom_diff[index];
    if (index % 2 != 0) {//class 1
        d_logits += (target[index] == 1) * term * zp;
        d_logits += (target[index] == 0) * term * zn;
      } else if (!only_pos) { //class 0 use the opposite alpha
        d_logits += (target[index] == 1) * term * zn;
        d_logits += (target[index] == 0) * term * zp;
      } else { // class 0 if only pos will not be balance
        d_logits += term;
      }
    bottom_diff[index] = d_logits;
  }
}

// template <typename Dtype>
// void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
//     const vector<Blob<Dtype>*>& top,  const vector<Blob<Dtype>*>& bottom) {
//     // The forward pass computes the sigmoid outputs.
//     const int count = bottom[0]->count();
//     // Stable version of loss computation from input data
//     const Dtype* input_data = bottom[0]->cpu_data();
//     const Dtype* target = bottom[1]->gpu_data();
//     Dtype* out_data = top[0]->mutable_gpu_data();

//     Dtype zn = (1.0 - alpha_);
//     Dtype zp = (alpha_);
//     LOG(INFO)<<"????";
//     for(int i = 0; i < count; ++i) {
//       LOG(INFO)<<"????";
//       if (input_data[i] >= 0){
//         out_data[i] = input_data[i] * (target[i] - 1) - log(1 + exp(-input_data[i]));
//       }else{
//         out_data[i] = input_data[i] * target[i] - log(1 + exp(input_data[i]));
//       }
//     }
//     LOG(INFO)<<"????";
//     ForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count),
//       CAFFE_CUDA_NUM_THREADS>>>(count, target ,out_data, zn, zp, only_pos_);
//     CUDA_POST_KERNEL_CHECK;
//     // LOG(INFO)<<out_data[122];
// 		// top[0]->mutable_cpu_data()[0] = loss / num;
// }

template <typename Dtype>
void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    Dtype zn = (1.0 - alpha_);
    Dtype zp = (alpha_);
    BackwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, target, bottom_diff, zn, zp, only_pos_);
    CUDA_POST_KERNEL_CHECK;
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

// INSTANTIATE_LAYER_GPU_FUNCS(TwoClassBalancedSigmoidCrossEntropyLossLayer);
INSTANTIATE_LAYER_GPU_BACKWARD(TwoClassBalancedSigmoidCrossEntropyLossLayer);

}  // namespace caffe
