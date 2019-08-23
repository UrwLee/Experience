#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/mask/GHMCSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GHMCSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_sub(count, sigmoid_output_data, target, bottom_diff);

    const Dtype* beta_data = beta_.gpu_data();
    caffe_gpu_mul(count, beta_data, bottom_diff, bottom_diff);

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

// INSTANTIATE_LAYER_GPU_FUNCS(GHMCSigmoidCrossEntropyLossLayer);
INSTANTIATE_LAYER_GPU_BACKWARD(GHMCSigmoidCrossEntropyLossLayer);
}  // namespace caffe
