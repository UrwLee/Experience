#ifndef CAFFE_MASK_PEAKS_FIND_LAYER_HPP_
#define CAFFE_MASK_PEAKS_FIND_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层用于在kps的估计map中查找极大值，用来确定关节点的位置
 * 过程：
 * １．遍历每个channel的通道，查找最大值
 * ２．最大值的(x,y,v)
 * x/y -> 归一化值
 * v -> 估计的置信度
 */

template <typename Dtype>
class PeaksFindLayer : public Layer<Dtype> {
 public:
  explicit PeaksFindLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PeaksFind"; }

  /**
   * bottom[0]: -> kps maps preds (Nroi,18,RH*RW)
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: -> output of peaks  (1,Nroi,18,3)
   * each line includes <x,y,v>
   * x/y -> 归一化坐标
   * v -> 置信度
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  /**
   * kps的估计map的长度和宽度
   */
  int height_;
  int width_;
};

}

#endif
