#include <algorithm>
#include <vector>

#include "caffe/mask/grad_clip_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradClipLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  caffe_gpu_memcpy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void GradClipBackward(const int count, const int spatial, const Dtype scale,
                                 const Dtype* in_diff, const Dtype* flags,
                                 Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    const int idx = index / spatial;
    const int flag = flags[idx];
    Dtype f = 0;
    if (flag > 0) f = scale;
    out_diff[idx] = in_diff[idx] * f;
  }
}

template <typename Dtype>
void GradClipLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to flags inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const int offs = bottom[0]->height() * bottom[0]->width();
    const Dtype* top_flags = bottom[1]->gpu_data();
    GradClipBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, offs, scale_,top_diff, top_flags, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradClipLayer);
}
