#ifndef CAFFE_IMRESIZE_LAYER_HPP_
#define CAFFE_IMRESIZE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层用于对某一个Feature按照不同尺度进行缩放。
 * 该层已停止使用。
 * 不建议后续使用。
 */

template <typename Dtype>
class ImResizeLayer : public Layer<Dtype> {
 public:
  explicit ImResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImResize"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void setTargetDimenions(int tw, int th);

  void SetStartScale(Dtype astart_scale) { start_scale_ = astart_scale; }
  void SetScaleGap(Dtype ascale_gap) { scale_gap_ = ascale_gap; }
  Dtype GetStartScale() { return start_scale_; }
  Dtype GetScaleGap() { return scale_gap_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
  }

  int targetSpatialWidth_;
  int targetSpatialHeight_;
  Dtype factor_;
  Dtype start_scale_;
  Dtype scale_gap_;
};

}

#endif
