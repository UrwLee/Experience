#include <vector>
#include <math.h>
#include <cfloat>
#include "caffe/mask/FocalSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  alpha_ = (float)this->layer_param_.focus_loss_param().alpha();
  gamma_ = (float)this->layer_param_.focus_loss_param().gama();
}

template <typename Dtype>
void FocalSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FocalSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
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
  // Dtype* out_data = top[0]->mutable_cpu_data();

  Dtype zn = (1.0 - alpha_);
 // Dtype zn = (alpha_);
  Dtype zp = (alpha_);

  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype p = pred[i];
    // // (1-p)**gamma * log(p) where
    Dtype term1 = pow((1. - p), gamma_) * log(std::max(p, Dtype(FLT_MIN)));
    // // p**gamma * log(1-p)
    Dtype term2 = pow(p, gamma_) * (-1. * input_data[i] * (input_data[i] >= 0) -
                                    log(1. + exp(input_data[i] - 2. * input_data[i] * (input_data[i] >= 0))));
    loss += -(target[i] == 1) * term1 * zp;
    loss += -(target[i] == 0) * term2 * zn;
    // out_data[i] = loss;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void FocalSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // const Dtype* out_diff = top[0]->cpu_diff();
    const Dtype* pred = sigmoid_top_vec_[0]->cpu_data();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();

    Dtype zn = (1.0 - alpha_);
    //Dtype zn = (alpha_);
    Dtype zp = (alpha_);

    for (int i = 0; i < count; ++i) {
      Dtype d_logits = 0;
      Dtype p = pred[i];
      // (1-p)**g * (1 - p - g*p*log(p)
      Dtype term1 = pow((1. - p), gamma_) * (1. - p - (p * gamma_ * log(std::max(p, Dtype(FLT_MIN)))));
      // (p**g) * (g*(1-p)*log(1-p) - p)
      Dtype term2 = pow(p, gamma_) * ((-1. * input_data[i] * (input_data[i] >= 0) -
                                       log(1. + exp(input_data[i] - 2. * input_data[i] * (input_data[i] >= 0)))) * (1. - p) * gamma_ - p);
      d_logits += -(target[i] == 1) * term1 * zp;
      d_logits += -(target[i] == 0) * term2 * zn;
      // bottom_diff[i] = d_logits * out_diff[i];
      bottom_diff[i] = d_logits;
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
// STUB_GPU(FocalSigmoidCrossEntropyLossLayer);
STUB_GPU_BACKWARD(FocalSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(FocalSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(FocalSigmoidCrossEntropyLoss);

}  // namespace caffe
