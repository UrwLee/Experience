#include <vector>
#include "caffe/mask/kps_label_layer.hpp"

namespace caffe {

template <typename Dtype>
KpsLabelLayer<Dtype>::KpsLabelLayer(const LayerParameter& param): Layer<Dtype>(param) {
  const KpsGenParameter& kps_gen_param = this->layer_param_.kps_gen_param();
  resized_height_ = kps_gen_param.resized_height();
  resized_width_ = kps_gen_param.resized_width();
  // default: false
  use_softmax_ = kps_gen_param.use_softmax();
}

template <typename Dtype>
void KpsLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 18);
  CHECK_EQ(bottom[0]->num(), bottom[1]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->width());
  CHECK_EQ(bottom[0]->num(), bottom[2]->count());
  if (use_softmax_) {
    CHECK_EQ(bottom[0]->count(), bottom[0]->num() * bottom[0]->channels());
  } else {
    CHECK_EQ(bottom[0]->height(), resized_height_);
    CHECK_EQ(bottom[0]->width(), resized_width_);
  }
  vector<int> shape(1,1);
  shape.push_back(bottom[0]->num());
  shape.push_back(18);
  shape.push_back(3);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void KpsLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype* map_data = bottom[0]->cpu_data();
  const Dtype* flags_c = bottom[1]->cpu_data();
  const Dtype* flags_i = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // const int offs_n = bottom[0]->offset(1);
  if (use_softmax_) {
    for (int n = 0; n < bottom[0]->num(); ++n) {
      if (flags_i[n] <= 0) {
        caffe_set(54, Dtype(-1), top_data);
        map_data += bottom[0]->offset(1);
        top_data += 54;
        continue;
      }
      for (int p = 0; p < 18; ++p) {
        int v = flags_c[n * 18 + p];
        if (v) {
          int max_p = map_data[n*18 + p];
          top_data[p * 3] = Dtype(max_p % resized_width_) / Dtype(resized_width_);
          top_data[p * 3 + 1] = Dtype(max_p / resized_width_) / Dtype(resized_height_);
          top_data[p * 3 + 2] = 1;
        } else {
          top_data[p*3 + 2] = -1;
          top_data[p*3 + 1] = -1;
          top_data[p*3] = -1;
        }
      }
      map_data += bottom[0]->offset(1);
      top_data += 54;
    }
  } else {
    const int offs_c = bottom[0]->height() * bottom[0]->width();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // unknown roi
      if (flags_i[n] <= 0) {
        caffe_set(54, Dtype(-1), top_data);
        map_data += bottom[0]->offset(1);
        top_data += 54;
        continue;
      }
      // scan all channels
      for (int p = 0; p < 18; ++p) {
        int v = flags_c[n * 18 + p];
        if (v) {
          // visible, found the max points
          int max_p = 0;
          Dtype max_val = -1;
          for (int i = 0; i < offs_c; ++i) {
            if (map_data[p * offs_c + i] > max_val) {
              max_val = map_data[p * offs_c + i];
              max_p = i;
            }
          }
          top_data[p * 3] = Dtype(max_p % bottom[0]->width()) / Dtype(bottom[0]->width());
          top_data[p * 3 + 1] = Dtype(max_p / bottom[0]->width()) / Dtype(bottom[0]->height());
          top_data[p * 3 + 2] = 1;
        } else {
          // unvisible
          top_data[p*3 + 2] = -1;
          top_data[p*3 + 1] = -1;
          top_data[p*3] = -1;
        }
      }
      // next pointer
      map_data += bottom[0]->offset(1);
      top_data += 54;
    }
  }
}

INSTANTIATE_CLASS(KpsLabelLayer);
REGISTER_LAYER_CLASS(KpsLabel);

}
