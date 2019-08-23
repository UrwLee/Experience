#ifndef CAFFE_DETECTION_MC_OUTPUT_LAYER_HPP_
#define CAFFE_DETECTION_MC_OUTPUT_LAYER_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * 该层主要用于YOLO检测器的输出。
 * 使用中心匹配策略，请在阅读源码基础上使用。
 */

template <typename Dtype>
class DetectionMcOutputLayer : public Layer<Dtype> {
 public:
  explicit DetectionMcOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionMcOutput"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_;
  int num_priors_;
  int num_classes_;

  Dtype conf_threshold_;
  Dtype nms_threshold_;
  Dtype boxsize_threshold_;

  vector<Dtype> prior_width_;
  vector<Dtype> prior_height_;

  int top_k_;
  bool clip_;
  bool visualize_;
  CodeLocType code_loc_type_;
  VisualizeParameter visual_param_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_OUTPUT_LAYER_HPP_
