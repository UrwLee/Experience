#include <algorithm>
#include <vector>

#include "caffe/nhwc/batch_norm_nhwc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormNHWCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  use_global_stats_ = this->phase_ == TEST;
  if (!use_global_stats_) {
    LOG(FATAL) << "Now BN_NHWC layer is only active in TEST-Phase, where use_global_stats must be true.";
  }
  if (bottom[0]->num_axes() == 1) {
    channels_ = 1;
  } else {
    channels_ = bottom[0]->shape(3);
  }
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void BatchNormNHWCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1) {
    CHECK_EQ(bottom[0]->shape(3), channels_);
  }
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
}

template <typename Dtype>
void BatchNormNHWCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // data pointer
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // num of cols
  const int NHW = bottom[0]->count() / channels_;
  // Eigen maps
  typedef Eigen::Map<const Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> > ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> > EigenMatrixMap;
  // if not in-place, then copy bottom to top
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  // consist of input-mat and output-mat
  ConstEigenMatrixMap input_mat(bottom_data, channels_, NHW);
  EigenMatrixMap output_mat(top_data, channels_, NHW);
  // get mean and variance, using scalar
  const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->blobs_[2]->cpu_data()[0];
  caffe_cpu_scale(variance_.count(), scale_factor,
      this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
  caffe_cpu_scale(variance_.count(), scale_factor,
      this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  // NOTE: (x-mean) / sqrt(eps+variance)
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());
  // get mean col vector
  ConstEigenMatrixMap mean_mat(mean_.cpu_data(), channels_, 1);
  ConstEigenMatrixMap variance_mat(variance_.cpu_data(), channels_, 1);
  // compute BN forward
  for (int i = 0; i < NHW; ++i) {
    // subtract mean
    output_mat.col(i) -= mean_mat.col(0);
    // div variance
    output_mat.col(i) = output_mat.col(i).cwiseQuotient(variance_mat.col(0));
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchNormNHWCLayer);
#endif

INSTANTIATE_CLASS(BatchNormNHWCLayer);
REGISTER_LAYER_CLASS(BatchNormNHWC);
}
