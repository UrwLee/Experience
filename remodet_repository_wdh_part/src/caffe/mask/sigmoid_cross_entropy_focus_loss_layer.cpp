#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>

#include "caffe/mask/sigmoid_cross_entropy_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyFocusLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  gama_ = this->layer_param_.focus_loss_param().gama();
}

template <typename Dtype>
void SigmoidCrossEntropyFocusLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyFocusLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  // const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype prob = sigmoid_output_data[i];
    Dtype p = std::min(std::max(prob, Dtype(FLT_MIN)), Dtype(1.0) - Dtype(FLT_MIN));
    loss -= target[i] * pow(Dtype(1.0)-p,gama_) * log(p) + (1-target[i]) * pow(p,gama_) * log(Dtype(1.0) - p);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyFocusLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      Dtype prob = sigmoid_output_data[i];
      Dtype p = std::min(std::max(prob, Dtype(FLT_MIN)), Dtype(1.0) - Dtype(FLT_MIN));
      Dtype pwr_p = pow(Dtype(1.0)-p,gama_);
      Dtype pwr_n = pow(p,gama_);
      Dtype y = target[i];
      Dtype coeffp = Dtype(1.0) - p - p * gama_ * log(p);
      Dtype coeffn = gama_ * (Dtype(1.0) - p) * log(Dtype(1.0) - p) - p;
      bottom_diff[i] = -y * pwr_p * coeffp - (Dtype(1.0) - y) * pwr_n * coeffn;
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

// #ifdef CPU_ONLY
// STUB_GPU_BACKWARD(SigmoidCrossEntropyFocusLossLayer, Backward);
// #endif

INSTANTIATE_CLASS(SigmoidCrossEntropyFocusLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyFocusLoss);

}
