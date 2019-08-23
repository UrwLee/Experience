#ifndef CAFFE_DETECTION_EVALUATE_LAYER_HPP_
#define CAFFE_DETECTION_EVALUATE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为检测器的输出提供了评估方法。
 * 评估方法包括mAP/AP(class)；
 * 评估策略包括Integral/MaxIntegral/11point等
 * 该方法已经在mask/det_eval_layer中重新实现了，可以参考该类的实现。
 */

template <typename Dtype>
class DetectionEvaluateLayer : public Layer<Dtype> {
 public:
  explicit DetectionEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionEvaluate"; }
  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // 参数定义
  int background_label_id_;
  float overlap_threshold_;
  bool evaluate_difficult_gt_;
  int num_classes_;

  map<int, float> size_thre_;
  map<int, float> iou_thre_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_EVALUATE_LAYER_HPP_
