#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/mask/FocalSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// template <typename Dtype>
// __global__ void ForwardKernel(const int count,
//           const Dtype* target, const Dtype* input_data, Dtype* out_data,
//           Dtype zn, Dtype zp, Dtype gamma_, Dtype flt_min){
//   CUDA_KERNEL_LOOP(index, count) {
//     Dtype loss1 = 0;

//     Dtype p = 1. / (1. + exp(-input_data[index]));
//     // // (1-p)**gamma * log(p) where
//     Dtype term1 = powf((1. - p), gamma_) * log(max(p, flt_min));
//     // // p**gamma * log(1-p)
//     Dtype term2 = powf(p, gamma_) * (-1. * input_data[index] * (input_data[index] >= 0) -
//                                     log(1. + exp(input_data[index] - 2. * input_data[index] * (input_data[index] >= 0))));
//     loss1 += -(target[index] == 1) * term1 * zp;
//     loss1 += -(target[index] == 0) * term2 * zn;
//     out_data[index] = loss1;
//   }
// }

template <typename Dtype>
__global__ void BackwardKernel(const int counts, const Dtype* pred,
          const Dtype* target, const Dtype* input_data, Dtype* bottom_diff,
          Dtype zn, Dtype zp, Dtype gamma_, Dtype flt_min) {
  CUDA_KERNEL_LOOP(index, counts) {
    Dtype d_logits = 0;
    Dtype p = pred[index];
    // (1-p)**g * (1 - p - g*p*log(p)
    Dtype term1 = powf((1. - p), gamma_) * (1. - p - (p * gamma_ * log(max(p, flt_min))));
    // (p**g) * (g*(1-p)*log(1-p) - p)
    Dtype term2 = powf(p, gamma_) * ((-1. * input_data[index] * (input_data[index] >= 0) -
                                     log(1. + exp(input_data[index] - 2. * input_data[index] * (input_data[index] >= 0)))) * (1. - p) * gamma_ - p);
    d_logits += -(target[index] == 1) * term1 * zp;
    d_logits += -(target[index] == 0) * term2 * zn;
    bottom_diff[index] = d_logits;
  }
}

// template <typename Dtype>
// void FocalSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
//     const vector<Blob<Dtype>*>& top,  const vector<Blob<Dtype>*>& bottom) {
//     // The forward pass computes the sigmoid outputs.
//     // sigmoid_bottom_vec_[0] = bottom[0];
//     // sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
//     // Compute the loss (negative log likelihood)
//     const int count = bottom[0]->count();
//     // Stable version of loss computation from input data
//     const Dtype* input_data = bottom[0]->gpu_data();
//     const Dtype* target = bottom[1]->gpu_data();
//     // const Dtype* pred = sigmoid_top_vec_[0]->gpu_data();
//     Dtype* out_data = top[0]->mutable_gpu_data();

//     Dtype zn = (1.0 - alpha_);
//     Dtype zp = (alpha_);

//     ForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count),
//       CAFFE_CUDA_NUM_THREADS>>>(count, target, input_data,
//       out_data, zn, zp, gamma_, Dtype(FLT_MIN));
//     CUDA_POST_KERNEL_CHECK;
// 		// top[0]->mutable_cpu_data()[0] = loss / num;
// }

template <typename Dtype>
void FocalSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // const Dtype* out_diff = top[0]->gpu_diff();
    const Dtype* pred = sigmoid_top_vec_[0]->gpu_data();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype zn = (1.0 - alpha_);
    Dtype zp = (alpha_);

    BackwardKernel<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, pred, target, input_data,
      bottom_diff, zn, zp, gamma_, Dtype(FLT_MIN));
    CUDA_POST_KERNEL_CHECK;
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

// INSTANTIATE_LAYER_GPU_FUNCS(FocalSigmoidCrossEntropyLossLayer);
INSTANTIATE_LAYER_GPU_BACKWARD(FocalSigmoidCrossEntropyLossLayer);
}  // namespace caffe
