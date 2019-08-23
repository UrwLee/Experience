#include <algorithm>
#include <vector>

#include "caffe/mask/EnforceOne.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EnforceOneForward(const int n, const int channels, const int height, const int width, const int threshold, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int top_index;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        Dtype background = bottom_data[((index * channels + 0) * height + h) * width + w];
        Dtype foreground = bottom_data[((index * channels + 1) * height + h) * width + w];
        top_index = (index * height + h) * width + w;
        top_data[top_index] = (((foreground - background) > 0) && foreground > threshold) ? 1.0 : 0.0;
      }
    }
  }
}

template <typename Dtype>
void EnforceOneLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int width = bottom[0]->width();
    int height = bottom[0]->height();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    CHECK_EQ(channels, 2); // 只适用于单纯人的分割
    EnforceOneForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, channels, height, width, threshold_, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
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
void EnforceOneLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(EnforceOneLayer);
}
