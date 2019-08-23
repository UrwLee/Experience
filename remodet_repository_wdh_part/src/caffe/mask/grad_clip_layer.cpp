#include "caffe/mask/grad_clip_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradClipLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  scale_ = this->layer_param_.grad_clip_param().scale();
}

template <typename Dtype>
void GradClipLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  // check bottom[0] (maps)
  // bottom[0]: [N,C,H,W]
  // bottom[1]: [1,1,N,C] (flags)
  CHECK_EQ(bottom[0]->num(), bottom[1]->height());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->width());
  // top[0] -> bottom[0]
  top[0]->ReshapeLike(*bottom[0]);
}


template <typename Dtype>
void GradClipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  // top[0] -> same as bottom[0]
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GradClipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to flags inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int channel = bottom[0]->channels();
    const int offs = bottom[0]->height() * bottom[0]->width();
    const Dtype* top_flags = bottom[1]->cpu_data();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channel; ++c) {
        const int idx = n * channel + c;
        const int flag = top_flags[idx];
        Dtype scale = (flag > 0) ? scale_ : 0;
        caffe_cpu_scale(offs, scale, top_diff + idx * offs, bottom_diff + idx * offs);
      }
    }
  }
}

INSTANTIATE_CLASS(GradClipLayer);
REGISTER_LAYER_CLASS(GradClip);

}
