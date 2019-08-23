#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>

#include "caffe/mask/EnforceOne.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EnforceOneLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const EnforceOneParameter& enforce_one_param = this->layer_param_.enforce_one_param();
  threshold_ = enforce_one_param.threshold();
}

template <typename Dtype>
void EnforceOneLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EnforceOneLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(channels, 2); // 只适用于单纯人的分割
  int top_index;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        Dtype background = bottom_data[((n * channels + 0) * height + h) * width + w];
        Dtype foreground = bottom_data[((n * channels + 1) * height + h) * width + w];
        // LOG(INFO)<<background<<" "<<foreground<<" "<<threshold_;
        top_index = (n * height + h) * width + w;
        top_data[top_index] = (((foreground - background) > 0) && foreground > threshold_) ? 1.0 : 0.0;
        // LOG(INFO)<<top_data[top_index];
      }
    }
  }
  // for (int n = 0; n < top[0]->num(); n++) {
  //   for (int c = 0; c < top[0]->channels(); c++) {
  //     for (int h = 0; h < top[0]->height(); h++) {
  //       for (int w = 0; w < top[0]->width(); w++) {
  //         LOG(INFO)<<" data at ["<<n<<"]"<<"["<<c<<"]"<<"["<<h<<"]"<<"["<<w<<"]"<<top[0]->data_at(n,c,h,w);
  //       }
  //     }
  //   }
  // }
}
template <typename Dtype>
void EnforceOneLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(EnforceOneLayer);
#endif

INSTANTIATE_CLASS(EnforceOneLayer);
REGISTER_LAYER_CLASS(EnforceOne);

}