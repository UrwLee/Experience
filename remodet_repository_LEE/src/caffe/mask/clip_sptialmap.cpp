#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>

#include "caffe/mask/clip_sptialmap.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void ClipSptialmapLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ClipSptialmapParameter& clip_sptialmap_param = this->layer_param_.clip_sptialmap_param();
  map_scale_ = clip_sptialmap_param.map_scale();
  channels_ = clip_sptialmap_param.axis_size();
  for (int i = 0; i < channels_; ++i) {
    axis_.push_back(clip_sptialmap_param.axis(i));
  }
}

template <typename Dtype>
void ClipSptialmapLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), channels_,
                  bottom[0]->height() / map_scale_, bottom[0]->width() / map_scale_);
}

template <typename Dtype>
void ClipSptialmapLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int width_ori = bottom[0]->width();
  int height_ori = bottom[0]->height();
  int width = bottom[0]->width() / map_scale_;
  int height = bottom[0]->height() / map_scale_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num; n++) {
    int bottom_index;
    int top_index;
    for (int c = 0; c < channels_; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int offset = n * 6 + axis_[c];
          bottom_index = ( offset * height_ori + h) * width_ori + w;
          top_index = ( (n * channels_ + c) * height + h) * width + w;
          top_data[top_index] = bottom_data[bottom_index];
        }
      }
    }
  }
}

template <typename Dtype>
void ClipSptialmapLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
}

// #ifdef CPU_ONLY
// STUB_GPU_BACKWARD(ClipSptialmapLayer, Backward);
// #endif

INSTANTIATE_CLASS(ClipSptialmapLayer);
REGISTER_LAYER_CLASS(ClipSptialmap);

}

#endif