#include <vector>

#include "caffe/tracker/halfmerge_layer.hpp"

namespace caffe {

template <typename Dtype>
void Merge(Dtype* bottom_data, const bool forward, const vector<int> shape, Dtype* top_data) {
  const int N = shape[0];
  const int C = shape[1];
  const int H = shape[2];
  const int W = shape[3];
  const int HN = N / 2;
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
          int on = n % HN;
          int oc = C * (n / HN) + c;
          int input_idx = ((n*C+c)*H+i)*W+j;
          int output_idx = ((on*2*C+oc)*H+i)*W+j;
          if (forward) {
            top_data[output_idx] = bottom_data[input_idx];
          } else {
            bottom_data[input_idx] = top_data[output_idx];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num() % 2, 0) << "input nums must be even numbers.";
  const int num_output = bottom[0]->num() / 2;
  const int num_channel = bottom[0]->channels() * 2;
  top[0]->Reshape(num_output,num_channel,bottom[0]->height(),bottom[0]->width());
  CHECK_EQ(top[0]->count(), bottom[0]->count()) << "output count does not match.";
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num() % 2, 0) << "input nums must be even numbers.";
  const int num_output = bottom[0]->num() / 2;
  const int num_channel = bottom[0]->channels() * 2;
  top[0]->Reshape(num_output,num_channel,bottom[0]->height(),bottom[0]->width());
  CHECK_EQ(top[0]->count(), bottom[0]->count()) << "output count does not match.";
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {

  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(bottom[0]->count(), top[0]->count())
    << "bottom and top blobs should have the same length.";
  Merge(bottom_data, true, bottom[0]->shape(), top_data);
}

template <typename Dtype>
void HalfmergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    CHECK_EQ(bottom[0]->count(), top[0]->count())
      << "bottom and top blobs should have the same length.";
    Merge(bottom_diff, false, bottom[0]->shape(), top_diff);
  }
}

INSTANTIATE_CLASS(HalfmergeLayer);
REGISTER_LAYER_CLASS(Halfmerge);

}  // namespace caffe
