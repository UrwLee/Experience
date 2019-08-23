#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, // 虚类的继承 规则？？
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; } 
  virtual inline int ExactNumTopBlobs() const { return 1; }

// 判断是否a 包含 b， if (a.find(b) != std::string::npos) {return "yes "}
  virtual void ToNHWC(){
    if (this->layer_param_.name().find("conv_fc") != std::string::npos || //std::string::npos 不存在
        this->layer_param_.name().find("Conv_fc") != std::string::npos ||
        this->layer_param_.name().find("CONV_fc") != std::string::npos ||
        this->layer_param_.name().find("CONV_FC") != std::string::npos) {
      // NCHW: [N,C*H*W]
      // NHWC: [N,H*W*C]
      CHECK_GT(this->blobs_.size(), 0); // InnerProductLayer 自己的blobs_ 存储可学习参数 作为 a set of blobs
      CHECK_EQ(this->blobs_[0]->num_axes(), 2);
      const int N = this->blobs_[0]->shape(0);
      const int CHW = this->blobs_[0]->shape(1);
      const int HW = CHW / channels_;
      vector<int> shape;
      shape.push_back(N);
      shape.push_back(CHW);
      Blob<Dtype> weights(shape);
      caffe_copy(weights.count(),this->blobs_[0]->cpu_data(),weights.mutable_cpu_data());
      const Dtype* weight_original = weights.cpu_data();
      Dtype* weight_ptr = this->blobs_[0]->mutable_cpu_data();
      for (int n = 0; n < N; ++n) {
        for (int i = 0; i < CHW; ++i) {
          const int hw = i % HW;
          const int c = (i / HW) % channels_;
          const int new_i = c * HW + hw;
          const int idx = n * CHW + i;
          const int new_idx = n * CHW + new_i;
          weight_ptr[new_idx] = weight_original[idx];
        }
      }
    }
 }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_; // k个 
  int N_; //num_output
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  int channels_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
