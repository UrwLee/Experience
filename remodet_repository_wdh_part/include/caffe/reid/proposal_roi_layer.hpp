#ifndef CAFFE_REID_PROPOSAL_ROI_LAYER_HPP_
#define CAFFE_REID_PROPOSAL_ROI_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层已停止使用，禁止使用。
 */

template <typename Dtype>
class ProposalRoiLayer : public Layer<Dtype> {
 public:
  explicit ProposalRoiLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ProposalRoi"; }
  // bottom[0] -> Propsals
  // [1,1,N,61]
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // top[0] -> [N, 5] (used for RoiPooling) (0, x1,y1,x2,y2) (AbsValue)
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  int net_input_width_;
  int net_input_height_;
};

}  // namespace caffe

#endif
