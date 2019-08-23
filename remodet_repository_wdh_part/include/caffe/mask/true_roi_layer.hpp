#ifndef CAFFE_MASK_TRUE_ROI_LAYER_HPP_
#define CAFFE_MASK_TRUE_ROI_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层将从所有的Gt-Boxes中筛选出具有Kps或Mask-Label的boxes，
 * 来进行Kps/Mask任务的评估。
 * １．选出具有Kps的boxes
 * ２．选出具有Mask的boxes
 */

template <typename Dtype>
class TrueRoiLayer : public Layer<Dtype> {
 public:
  explicit TrueRoiLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TrueRoi"; }

  /**
   * bottom[0]: -> Labels of each image (1,1,Ng,66+H*W)
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: -> ROIs for Kps or Mask (1,1,Nroi,7)
   *            <bindex,cid,pid,xmin,ymin,xmax,ymax>
   * top[1]: -> Flags for each output ROI (1,1,1,Nroi)
   *            1 -> active & -1 -> unactive
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  /**
   * 选择类型
   * "mask" or "pose"
   */
  string type_;
};

}  // namespace caffe

#endif
