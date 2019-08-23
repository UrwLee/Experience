#ifndef CAFFE_MASK_KPS_LABEL_LAYER_HPP_
#define CAFFE_MASK_KPS_LABEL_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层提供了从Kps-GT-Maps中恢复其(x,y,v)真实值的方法
 * 方法如下：
 * １．查找GT-Maps中的最大值处
 * ２．归一化在Map中的位置
 * 对于不存在的，或样本无效的情形，输出全部为－１
 */

template <typename Dtype>
class KpsLabelLayer : public Layer<Dtype> {
 public:
  explicit KpsLabelLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KpsLabel"; }

  /**
   * bottom[0] -> GT Maps         [N,18,H,W]
   *                              [N,18]
   * bottom[1] -> channel Flags   [1,1,N,18]
   * bottom[2] -> roi Flags       [1,1,1,N]
   */
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  /**
   * top[0] -> [1,N,18,3]
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  int resized_height_, resized_width_;
  bool use_softmax_;
};

}

#endif
