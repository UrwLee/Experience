#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/reid/labeled_match_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabeledMatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // labeled identity nums (e.g., 5532)
  num_classes_ = this->layer_param_.labeled_match_param().num_classes();
  // feature vector update (gama, e.g., 0.5)
  momentum_ = this->layer_param_.labeled_match_param().momentum();
  // copy_status_
  copy_status_ = this->layer_param_.labeled_match_param().copy_status();
  // blobs_ -> save feature vectors
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // Intialize the matrix for saving instance features
    vector<int> shape(2);
    // L
    shape[0] = num_classes_;
    // D
    shape[1] = bottom[0]->count(1);
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    // Initialize the vector for storing the update timestamps
    shape.resize(1);
    shape[0] = num_classes_;
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    // Reset the blobs data to zero
    for (int i = 0; i < this->blobs_.size(); ++i) {
      caffe_set(this->blobs_[i]->count(), (Dtype)0,
                this->blobs_[i]->mutable_cpu_data());
    }
  }

  // "Parameters" will be updated, but not by standard backprop with gradients.
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LabeledMatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // check
  CHECK_EQ(this->blobs_[0]->count(1), bottom[0]->count(1))
      << "Input size incompatible with initialization.";
  vector<int> shape(2);
  // N
  shape[0] = bottom[0]->shape(0);
  // L
  shape[1] = num_classes_;
  // top[0] -> [N, L] (cosine similarity)
  top[0]->Reshape(shape);
  // top[1] -> bottom[1] (labels, [N, 1])
  top[1]->ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void LabeledMatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // num of proposals
  const int M = bottom[0]->shape(0);
  // D (vector len)
  const int K = bottom[0]->count(1);
  // num of stored instances
  const int N = num_classes_;
  // compute the cosine similarity
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
      (Dtype)1., bottom[0]->cpu_data(), this->blobs_[0]->cpu_data(),
      (Dtype)0., top[0]->mutable_cpu_data());
  // assign the labels
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  caffe_copy(M, bottom_label, top_label);
  for (int i = 0; i < M; ++i) {
    const int label_value = static_cast<int>(bottom_label[i]);
    if (label_value < 0 || label_value >= num_classes_)
      top_label[i] = (Dtype)(-1.);
  }
  // if copy, then update blobs_
  if (copy_status_) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    for (int i = 0; i < M; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (label_value < 0 || label_value >= N) continue;
      caffe_cpu_axpby(K, (Dtype)1, bottom_data + i * K,
                      (Dtype)0, weight + label_value * K);
    }
  }
}

template <typename Dtype>
void LabeledMatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // num of samples
  const int M = bottom[0]->shape(0);
  // D
  const int K = bottom[0]->count(1);
  const int N = num_classes_;
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N,
        (Dtype)1., top[0]->cpu_diff(), this->blobs_[0]->cpu_data(),
        (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
  CHECK_EQ(this->param_propagate_down_[0], this->param_propagate_down_[1])
      << "Instance matrix and timestamp vector should be both updated or not";
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    // increase all timestamp
    Dtype* tstamp = this->blobs_[1]->mutable_cpu_data();
    caffe_add_scalar(this->blobs_[1]->count(), (Dtype)1., tstamp);
    // update instance feature
    for (int i = 0; i < M; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (label_value < 0 || label_value >= N) continue;
      // w <- momentum * w + (1-momentum) * x
      caffe_cpu_axpby(K, (Dtype)1. - momentum_, bottom_data + i * K,
                      momentum_, weight + label_value * K);
      // reset the timestamp
      tstamp[label_value] = (Dtype)0.;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabeledMatchLayer);
#endif

INSTANTIATE_CLASS(LabeledMatchLayer);
REGISTER_LAYER_CLASS(LabeledMatch);

}  // namespace caffe
