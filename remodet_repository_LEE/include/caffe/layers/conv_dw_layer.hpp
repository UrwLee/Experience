#ifndef CAFFE_CONV_DW_LAYER_HPP_
#define CAFFE_CONV_DW_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ConvolutionDepthwiseLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionDepthwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "ConvolutionDepthwise"; }
  virtual void ToNHWC() {
    // NCHW: [N1, N2, KH, KW]
    // NHWC: [KH, KW, N2, N1]
    // Step1: Reshape blobs_[0]
    // Step2: fill data
    CHECK_GT(this->blobs_.size(), 0);
    CHECK_EQ(this->blobs_[0]->num_axes(), 4);
    const int N1 = this->blobs_[0]->shape(0);
    const int N2 = this->blobs_[0]->shape(1);
    const int KH = this->blobs_[0]->shape(2);
    const int KW = this->blobs_[0]->shape(3);
    Blob<Dtype> weights(N1,N2,KH,KW);
    caffe_copy(weights.count(),this->blobs_[0]->cpu_data(),weights.mutable_cpu_data());
    this->blobs_[0]->Reshape(KH,KW,N2,N1);
    Dtype* weight_ptr = this->blobs_[0]->mutable_cpu_data();
    const Dtype* weight_original = weights.cpu_data();
    for (int n1 = 0; n1 < N1; ++n1) {
      for (int n2 = 0; n2 < N2; ++n2) {
        for (int fh = 0; fh < KH; ++fh) {
          for (int fw = 0; fw < KW; ++fw) {
            const int original_offs = ((n1 * N2 + n2) * KH + fh) * KW + fw;
            const int offs = ((fh * KW + fw) * N2 + n2) * N1 + n1;
            weight_ptr[offs] = weight_original[original_offs];
          }
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
  unsigned int kernel_h_;
  unsigned int kernel_w_;
  unsigned int stride_h_;
  unsigned int stride_w_;
  unsigned int pad_h_;
  unsigned int pad_w_;
  unsigned int dilation_h_;
  unsigned int dilation_w_;
  unsigned int group_;
  unsigned int num_output_;
  Blob<Dtype> weight_buffer_;
  Blob<Dtype> weight_multiplier_;
  Blob<Dtype> bias_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_DW_LAYER_HPP_
