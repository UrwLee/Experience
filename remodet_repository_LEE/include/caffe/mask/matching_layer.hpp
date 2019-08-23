#ifndef CAFFE_MASK_MATCHING_LAYER_HPP_
#define CAFFE_MASK_MATCHING_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

/**
 * 匹配层：获取匹配的ROI进行Mask/Kps的训练
 */

template <typename Dtype>
class BoxMatchingLayer : public Layer<Dtype> {
 public:
  explicit BoxMatchingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BoxMatching"; }

  // bottom[0] stores the prior bounding boxes.
  // bottom[1] stores the ground truth bounding boxes.
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // top[0] -> ROIs [1,1,Nroi,7]
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 正例匹配IOU阈值：下限
  Dtype overlap_threshold_;

  // 是否使用标记为diff的GT-Box，默认是None
  bool use_difficult_gt_;

  // gt数量
  int num_gt_;
  // 每个样本的prior_bboxes数量
  int num_priors_;

  int top_k_;

  // 正例数量
  int num_pos_;

  /**
   * 正例匹配列表：
   * map<int=prior-id, int=gt-id>
   */
  vector<vector<Dtype> > match_rois_;

  // size thre
  Dtype size_threshold_;
};

}  // namespace caffe

#endif
