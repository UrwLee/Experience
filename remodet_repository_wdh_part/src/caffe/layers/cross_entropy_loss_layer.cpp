#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype pred = std::min(std::max(bottom_data[i], Dtype(kLOG_THRESHOLD)), Dtype(1) - Dtype(kLOG_THRESHOLD));
    loss -= bottom_label[i] * log(pred) - (1.0 - bottom_label[i]) * log (1.0 - pred);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < count; ++i) {
      Dtype pred = std::min(std::max(bottom_data[i], Dtype(kLOG_THRESHOLD)), Dtype(1) - Dtype(kLOG_THRESHOLD));
      Dtype target = bottom_label[i];
      bottom_diff[i] = scale * (target / pred - (1.0 - target) / (1.0 - pred));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);
}
