#include "caffe/mask/grad_clip_spatial.hpp"

namespace caffe {

template <typename Dtype>
void GradClipSpatialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void GradClipSpatialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());

  top[0]->ReshapeLike(*bottom[0]);
}


template <typename Dtype>
void GradClipSpatialLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  // top[0] -> same as bottom[0]
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GradClipSpatialLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* mask = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  caffe_mul(count, top_diff, mask, bottom_diff);
}

INSTANTIATE_CLASS(GradClipSpatialLayer);
REGISTER_LAYER_CLASS(GradClipSpatial);

}
