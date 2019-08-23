#include "caffe/mask/peaks_find_layer.hpp"

namespace caffe {

template <typename Dtype>
void PeaksFindLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  height_ = this->layer_param_.peaks_find_param().height();
  width_ = this->layer_param_.peaks_find_param().width();
}

template <typename Dtype>
void PeaksFindLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->offset(0,1), height_ * width_);
  vector<int> shape(1,1);
  shape.push_back(bottom[0]->num());
  shape.push_back(bottom[0]->channels());
  shape.push_back(3);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PeaksFindLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  // get peaks of each map
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int offs = bottom[0]->offset(0,1);
  const Dtype* map_data = bottom[0]->cpu_data();
  //const int width = bottom[0]->width();
  //const int height = bottom[0]->height();
  Dtype* peaks = top[0]->mutable_cpu_data();
  int count = 0;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      // found peaks in this map
      Dtype max_v = map_data[0];
      int max_idx = 0;
      for (int i = 1; i < offs; ++i) {
        if (map_data[i] > max_v) {
          max_v = map_data[i];
          max_idx = i;
        }
      }
      peaks[count++] = Dtype(max_idx % width_) / Dtype(width_);
      peaks[count++] = Dtype(max_idx / width_) / Dtype(height_);
      peaks[count++] = max_v;
      map_data += bottom[0]->offset(0,1);
    }
  }
}

INSTANTIATE_CLASS(PeaksFindLayer);
REGISTER_LAYER_CLASS(PeaksFind);

}
