#ifndef CAFFE_TRACKER_MCOUT_LAYER_HPP_
#define CAFFE_TRACKER_MCOUT_LAYER_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * 对应于tracker_mcloss_layer的网格输出。
 * 本层提供了测试时的输出方法。
 * 该层不推荐使用。
 */

template <typename Dtype>
class TrackerMcOutLayer : public Layer<Dtype> {
 public:
  explicit TrackerMcOutLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TrackerMcOut"; }
  // bottom[0] -> [1, 5, D, D]
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // bottom[1] -> [1, 5, 1, 1]
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int grids_;
  Dtype prior_width_;
  Dtype prior_height_;
};

}  // namespace caffe

#endif
