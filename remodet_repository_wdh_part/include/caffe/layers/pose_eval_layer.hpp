#ifndef CAFFE_POSE_EVAL_LAYER_HPP_
#define CAFFE_POSE_EVAL_LAYER_HPP_
#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

namespace caffe {

/**
 * 该层主要用于对pose的检测结果进行评估。
 * 原理：参照MSCOCO的OKS方法定义目标对象和检测对象之间相对精度。
 * 后期pose的精度计算参考mask/kps_eval_layer.hpp，请阅读源码进行参考。
 */

template <typename Dtype>
class PoseEvalLayer : public Layer<Dtype> {
 public:
  explicit PoseEvalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseEval"; }

  /**
   * bottom[0]: -> [1,N,18,3] proposals
   * bottom[1]: -> [1,4,H,W] gt (每个map都包含了多个18*3的结果gt)
   * bottom[1]: -> 来自于pose_data_layer的top[1]，请参考output_kps=TRUE时的数据结构组织
   */
  virtual inline int ExactBottomBlobs() const { return 2; }

  /**
   * 计算TPR/FPR
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // heatmap的stride_,默认为8
  int stride_;
  // 统计的精度阈值
  vector<float> oks_thre_;
  // size的阈值，小于该阈值不统计
  float area_thre_;
};

}  // namespace caffe

#endif  // CAFFE_POSE_EVAL_LAYER_HPP_
