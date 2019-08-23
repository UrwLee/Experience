#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/mask/mask_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->width());
}

template <typename Dtype>
void MaskCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* flags = bottom[2]->cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int offsize = bottom[0]->count() / num / channels;
  Dtype loss = 0;
  Dtype scale = this->layer_param_.mask_loss_param().scale();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int flag = flags[n*channels+c];
      if (flag <= 0) continue;
      int offs = (n*channels+c)*offsize;
      for (int i = 0; i < offsize; ++i) {
        Dtype pred = std::min(std::max(bottom_data[offs+i], Dtype(kLOG_THRESHOLD)), Dtype(1) - Dtype(kLOG_THRESHOLD));
        loss -= bottom_label[offs+i] * log(pred) - (1.0 - bottom_label[offs+i]) * log (1.0 - pred);
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss * scale / num;
}

template <typename Dtype>
void MaskCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //              << " Layer cannot backpropagate to label inputs.";
  // }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* flags = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int offsize = bottom[0]->count() / num / channels;
    Dtype scale_temp = this->layer_param_.mask_loss_param().scale();
    const Dtype scale = - top[0]->cpu_diff()[0] / num * scale_temp;
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        int flag = flags[n*channels+c];
        int offs = (n*channels+c)*offsize;
        if (flag > 0) {
          for (int i = 0; i < offsize; ++i) {
            Dtype pred = std::min(std::max(bottom_data[offs+i], Dtype(kLOG_THRESHOLD)), Dtype(1) - Dtype(kLOG_THRESHOLD));
            Dtype target = bottom_label[offs+i];
            bottom_diff[offs+i] = scale * (target / pred - (1.0 - target) / (1.0 - pred));
          }
        } else {
          caffe_set(offsize, Dtype(0), bottom_diff + offs);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaskCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(MaskCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MaskCrossEntropyLoss);
}
