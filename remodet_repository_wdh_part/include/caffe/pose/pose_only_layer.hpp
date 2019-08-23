#ifndef CAFFE_POSE_POSE_ONLY_LAYER_HPP_
#define CAFFE_POSE_POSE_ONLY_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * 该类仅仅输入姿态识别的proposal信息，不包含任意检测器的输出结果。
 * 方法：
 * １．根据姿态识别的proposals估计其自身的box
 * ２．然后打包输出新的proposal，新的proposal与PoseDetLayer的输出格式完全一致，box的结果由points估计得到
 *
 * 该层仅作为测试用，或熟练掌握源码基础上使用。
 */

template <typename Dtype>
class PoseOnlyLayer : public Layer<Dtype> {
 public:
  explicit PoseOnlyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseOnly"; }
  // bottom[0] -> pose_proposal
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_parts_;
};

}

#endif
