#ifndef CAFFE_REID_ROI_DATA_LAYER_HPP_
#define CAFFE_REID_ROI_DATA_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层已停止使用，禁止使用该层。
 */

template <typename Dtype>
class RoiDataLayer : public Layer<Dtype> {
 public:
  explicit RoiDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RoiData"; }
  // bottom[0] -> gtdata
  // [N, 7] (batch_id, cls_id, person_id, x1,y1,x2,y2)
  // x1,y1,x2,y2 -> normalized data
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // top[0] -> [N, 5] (used for RoiPooling) (batch_id, x1,y1,x2,y2)
  // top[1] -> [N, 1] (labels) (person_id)
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);]

  int net_input_width_;
  int net_input_height_;
};

}  // namespace caffe

#endif
