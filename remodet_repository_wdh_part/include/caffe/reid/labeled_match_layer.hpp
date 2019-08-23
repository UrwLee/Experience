#ifndef CAFFE_REID_LABELED_MATCH_LAYER_HPP_
#define CAFFE_REID_LABELED_MATCH_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层用于Re-Identification任务，基于Softmax以及目标对象特征建模。
 * 禁止直接使用。
 */

template <typename Dtype>
class LabeledMatchLayer : public Layer<Dtype> {
 public:
  explicit LabeledMatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabeledMatch"; }
  // bottom[0] -> [N, D] (features)
  // bottom[1] -> [N, 1] (label)
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // top[0] -> [N, L] (cosine similarity)
  // top[1] -> [N, 1] (label)
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_classes_;
  Dtype momentum_;
  bool copy_status_;
};

}  // namespace caffe

#endif
