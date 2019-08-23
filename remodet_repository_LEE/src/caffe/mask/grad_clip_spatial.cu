#include "caffe/mask/grad_clip_spatial.hpp"

namespace caffe {

template <typename Dtype>
void GradClipSpatialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  caffe_gpu_memcpy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void GradClipSpatialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* mask = bottom[1]->gpu_data();
  const int count = bottom[0]->count();
  caffe_gpu_mul(count, top_diff, mask, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(GradClipSpatialLayer);
}
