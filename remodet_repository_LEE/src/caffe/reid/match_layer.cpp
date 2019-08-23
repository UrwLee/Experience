#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/reid/match_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void MatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // feature vector update (gama, e.g., 0.5)
  momentum_ = this->layer_param_.match_param().momentum();
  initialized_ = false;
  pid_ = 1e4;
  // blobs_ -> save feature vectors for the target
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the matrix for saving instance features
    vector<int> shape(2);
    // only storage 1 .
    shape[0] = 1;
    // D length of feature vectors
    shape[1] = bottom[1]->count(1);
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    // Initialize the vector for storing the update timestamps
    // Reset the blobs data to zero
    caffe_set(this->blobs_[0]->count(), (Dtype)0,
              this->blobs_[0]->mutable_cpu_data());
  }
}

template <typename Dtype>
void MatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // check
  CHECK_EQ(this->blobs_[0]->count(1), bottom[1]->count(1)) << "Input size incompatible with initialization.";
  CHECK_EQ(bottom[0]->width(), 61) << "proposals should have length of 61.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->shape(0)) << "Unmatched for proposals & num of features."
          << " proposals : " << bottom[0]->height() << ", while feature numbers: " << bottom[1]->shape(0);
  vector<int> shape(2,1);
  // number of proposals
  shape.push_back(bottom[0]->height());
  shape.push_back(62);
  // add similarity to 61.
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // num of proposals
  const int N = bottom[0]->height();
  // vector len
  const int K = bottom[1]->count(1);
  // if no proposals , then skip
  const Dtype* proposal = bottom[0]->cpu_data();
  const Dtype* feature = bottom[1]->cpu_data();
  if ((N == 1) && ((int)proposal[60] < 0)) {
    // output
    top[0]->Reshape(1,1,1,62);
    caffe_set<Dtype>(top[0]->count(), (Dtype)-1, top[0]->mutable_cpu_data());
    return;
  }
  // reset
  if (!initialized_) {
    // find the smallest idx to save
    int idx = 0;
    for (int i = 0; i < N; ++i) {
      const int pid = proposal[61*i + 60];
      if (pid < pid_) {
        pid_ = pid;
        idx = i;
      }
    }
    // save feature
    caffe_copy(K, feature + idx*K, this->blobs_[0]->mutable_cpu_data());
    initialized_ = true;
  } else {
    // match
    // compute the cosine similarity
    vector<Dtype> similarity(N, 0);
    Blob<Dtype> simi_blob(1,1,1,N);
    // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N, 1, K,
    //     (Dtype)1., bottom[0]->cpu_data(), this->blobs_[0]->cpu_data(),
    //     (Dtype)0., top[0]->mutable_cpu_data());
    // caffe_cpu_gemv(CblasNoTrans, N, K,
    //     (Dtype)1., feature, this->blobs_[0]->cpu_data(),
    //     (Dtype)0., simi_blob.mutable_cpu_data());
    for (int i = 0; i < N; ++i) {
      simi_blob.mutable_cpu_data()[i] =
        caffe_cpu_dot<Dtype>(K, feature + i*K, this->blobs_[0]->cpu_data());
    }
    // softmax
    Blob<Dtype> softmax_simi_blob(1,1,1,N);
    Softmax(simi_blob.cpu_data(), N, softmax_simi_blob.mutable_cpu_data());
    // get similarity
    for (int i = 0; i < N; ++i) {
      // similarity[i] = softmax_simi_blob.cpu_data()[i];
      similarity[i] = simi_blob.cpu_data()[i];
    }
    /**
     * update & reid proposals
     */
    // found the same id
    int idx = -1;
    for (int i = 0; i < N; ++i) {
      const int pid = proposal[61*i + 60];
      if (pid == pid_) {
        idx = i;
        break;
      }
    }
    // update template & normalize
    if (idx >= 0) {
      caffe_cpu_axpby(K, (Dtype)1 - momentum_, feature + idx * K,
                     momentum_, this->blobs_[0]->mutable_cpu_data());
      // Normalize
      Blob<Dtype> buffer_data(1,1,1,K);
      // square
      caffe_sqr<Dtype>(K, this->blobs_[0]->cpu_data(), buffer_data.mutable_cpu_data());
      // norm-data
      Dtype norm_data = pow(caffe_cpu_asum<Dtype>(K, buffer_data.cpu_data()) + (Dtype)0.001, Dtype(0.5));
      // normalize
      caffe_cpu_scale<Dtype>(K, Dtype(1.0 / norm_data), this->blobs_[0]->cpu_data(), this->blobs_[0]->mutable_cpu_data());
    }

    // now we add similarity to all proposals
    top[0]->Reshape(1,1,N,62);
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < N; ++i) {
      // copy 0 - 60
      for (int j = 0; j < 61; ++j) {
        top_data[62*i+j] = proposal[61*i + j];
      }
      top_data[62*i+61] = similarity[i];
    }
  }
    // top[0]->Reshape(1,1,N,62);
    // Dtype* top_data = top[0]->mutable_cpu_data();
    // for (int i = 0; i < N; ++i) {
    //   // copy 0 - 60
    //   for (int j = 0; j < 61; ++j) {
    //     top_data[62*i+j] = proposal[61*i + j];
    //   }
    //   top_data[62*i+61] = 0;
    // }
}

#ifdef CPU_ONLY
STUB_GPU(MatchLayer);
#endif

INSTANTIATE_CLASS(MatchLayer);
REGISTER_LAYER_CLASS(Match);

}  // namespace caffe
