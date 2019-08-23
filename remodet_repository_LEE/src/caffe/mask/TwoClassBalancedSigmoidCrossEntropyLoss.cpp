#include <vector>
#include <math.h>
#include <cfloat>
#include "caffe/mask/TwoClassBalancedSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  alpha_ = (float)this->layer_param_.two_class_balanced_sigmoid_cross_entropy_loss_param().alpha();
  only_pos_ = this->layer_param_.two_class_balanced_sigmoid_cross_entropy_loss_param().only_pos();
}

template <typename Dtype>
void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  CHECK_EQ(bottom[0]->height(), 2) << "only support 2 class";
}

template <typename Dtype>
void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* pred = sigmoid_top_vec_[0]->cpu_data();
  Dtype* out_data = top[0]->mutable_cpu_data();

  Dtype zn = (1.0 - alpha_);
  Dtype zp = (alpha_);

  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype term = input_data[i] * (target[i] - (input_data[i] >= 0)) -
                 log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    if (i % 2 != 0) {//class 1
      loss -= (target[i] == 1) * term * zp;
      loss -= (target[i] == 0) * term * zn;
    } else if (!only_pos_) { //class 0 use the opposite alpha
      loss -= (target[i] == 1) * term * zn;
      loss -= (target[i] == 0) * term * zp;
    } else { // class 0 if only pos will not be balance
      loss -= term;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void TwoClassBalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    Dtype zn = (1.0 - alpha_);
    Dtype zp = (alpha_);

    for (int i = 0; i < count; ++i) {
      Dtype d_logits = 0;
      Dtype term = bottom_diff[i];

      if (i % 2 != 0) {//class 1
        d_logits += (target[i] == 1) * term * zp;
        d_logits += (target[i] == 0) * term * zn;
      } else if (!only_pos_) { //class 0 use the opposite alpha
        d_logits += (target[i] == 1) * term * zn;
        d_logits += (target[i] == 0) * term * zp;
      } else { // class 0 if only pos will not be balance
        d_logits += term;
      }
      bottom_diff[i] = d_logits;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
// STUB_GPU(TwoClassBalancedSigmoidCrossEntropyLossLayer);
STUB_GPU_BACKWARD(TwoClassBalancedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(TwoClassBalancedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(TwoClassBalancedSigmoidCrossEntropyLoss);

}  // namespace caffe
