#include <vector>

#include "caffe/reid/sum_across_channel_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumAcrossChanLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SumAcrossChanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GT(bottom[0]->channels(), 1);
  vector<int> shape(4,1);
  shape[0] = bottom[0]->num();
  shape[1] = 1;
  shape[2] = bottom[0]->height();
  shape[3] = bottom[0]->width();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void SumAcrossChanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int offs_item = channels * height * width;
  const int offs_chan = height * width;
  for (int item = 0; item < num; ++item) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int idx = item * height * width + i * width + j;
        // sum of all channels
        Dtype sum = 0;
        for (int c = 0; c < channels; ++c) {
          int bidx = item * offs_item + c * offs_chan + i * width + j;
          sum += bottom_data[bidx];
        }
        top_data[idx] = sum;
      }
    }
  }
}

INSTANTIATE_CLASS(SumAcrossChanLayer);
REGISTER_LAYER_CLASS(SumAcrossChan);

}  // namespace caffe
