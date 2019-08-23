#ifndef CAFFE_POSE_NEW_POSE_DET_LAYER_HPP_
#define CAFFE_POSE_NEW_POSE_DET_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * 参考layers/PoseDetLayer
 * 基本一致，作为测试用。
 * 不建议使用，或在完全掌握源码基础上修改测试用。
 */

template <typename Dtype>
class NewPoseDetLayer : public Layer<Dtype> {
 public:
  explicit NewPoseDetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NewPoseDet"; }
  // bottom[0] -> pose_proposal
  // bottom[1] -> det_proposal
  // bottom[2] -> peaks
  // bottom[3] -> heatmaps(vecmaps) (52 channels)
  virtual inline int MinBottomBlobs() const { return 4; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_parts_;
  int max_peaks_;
  Dtype coverage_min_thre_;
  Dtype score_pose_ebox_;
  Dtype keep_det_box_thre_;

  // 历史值，only one
  vector<Dtype> history_;
};

}

#endif
