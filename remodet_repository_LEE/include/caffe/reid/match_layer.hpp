#ifndef CAFFE_REID_MATCH_LAYER_HPP_
#define CAFFE_REID_MATCH_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/shared_ptr.hpp>

namespace caffe {

/**
 * 禁止使用该层，该层已停止使用。
 */

template <typename Dtype>
class MatchLayer : public Layer<Dtype> {
 public:
  explicit MatchLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Match"; }
  // bottom[0] -> [N, 61] (proposals), after easy_match_layer
  // 0-3: box [xmin,ymin,xmax,ymax] (normalized)
  // 4-57: kps (18x3) (normalized)
  // 58: num of points
  // 59: score, and 60: id
  // bottom[1] -> [N, D] (features)
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // top[0] -> [N, 62] (add similarity)
  // [0-60] keep same
  // [61] -> similarity
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // update the mask and then normalize it
  Dtype momentum_;
  // reset for the first object.
  bool initialized_;
  int pid_;
};

}  // namespace caffe

#endif
