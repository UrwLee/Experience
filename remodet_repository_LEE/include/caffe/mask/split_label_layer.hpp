#ifndef CAFFE_MASK_SPLIT_LABEL_LAYER_HPP_
#define CAFFE_MASK_SPLIT_LABEL_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层将输入的label分解为多个任务的label.
 * 分解的任务包含：
 * １．Detection: BBox
 * ２．Pose: Kps
 * ３．Segmention: Mask
 */

template <typename Dtype>
class SplitLabelLayer : public Layer<Dtype> {
 public:
  explicit SplitLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SplitLabel"; }

  /**
   * bottom[0]: -> Label of input data (1,1,Ng,66+H*W)
   * each gt-box has label of vector<66+H*W>
   * <bindex,cid,pid,is_diff,iscrowd,xmin,ymin,xmax,ymax,
   *  has_kps,num_kps,18*3,has_mask,H*W>
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: -> Label for detection (1,1,Ng,9) (cid == 0)
   *            <bindex,cid,pid,is_diff,iscrowd,xmin,ymin,xmax,ymax>
   * top[1]: -> Label for detection (1,1,Ngp,9) (cid > 0)
   *            <bindex,cid,pid,is_diff,iscrowd,xmin,ymin,xmax,ymax>
   * top[2]: -> Label for Pose (Kps) (1,1,Ng,61)
   *            <bindex,cid,pid,is_diff,iscrowd,has_kps,num_kps,18*3>
   * top[3]: -> Label for Mask (1,1,Ng,6+H*W)
   *            <bindex,cid,pid,is_diff,iscrowd,has_mask,H*W>
   */
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  bool add_parts_;
  bool add_kps_;
  bool add_mask_;
  int spatial_dim_;
};

}  // namespace caffe

#endif
