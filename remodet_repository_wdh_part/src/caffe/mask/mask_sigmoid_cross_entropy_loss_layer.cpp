#include <vector>

#include "caffe/mask/mask_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  scale_ = this->layer_param_.mask_loss_param().scale();
}

template <typename Dtype>
void MaskSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  // bottom[0] -> [N,C,H,W]
  // bottom[1] -> [N,C,H,W]
  // bottom[2] -> [1,1,N,C]
  CHECK_EQ(bottom[0]->num(), bottom[2]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->width());
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MaskSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* flags = bottom[2]->cpu_data();
  Dtype loss = 0;
  const int offsize = height * width;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int flag = flags[n*channels+c];
      if (flag <= 0) continue;
      const int offs = (n*channels+c)*offsize;
      for (int i = 0; i < offsize; ++i) {
        loss -= input_data[offs+i] * (target[offs+i] - (input_data[offs+i] >= 0)) -
            log(1 + exp(input_data[offs+i] - 2 * input_data[offs+i] * (input_data[offs+i] >= 0)));
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num * scale_;
}

template <typename Dtype>
void MaskSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int offsize = height * width;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype* flags = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        int flag = flags[n*channels+c];
        if (flag > 0) {
          // normal
          caffe_scal(offsize, loss_weight * scale_ / num, bottom_diff);
        } else {
          // back zero
          caffe_set(offsize, Dtype(0), bottom_diff);
        }
        bottom_diff += offsize;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MaskSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MaskSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MaskSigmoidCrossEntropyLoss);

}  // namespace caffe
