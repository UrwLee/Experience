#ifndef CAFFE_ROI_RESIZE_LAYER_HPP_
#define CAFFE_ROI_RESIZE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层完成了对Roi区域的Resize方法。
 * 不同于RoiPooling/RoiAlign层的做法，RoiResize对Roi区域进行双线性插值。
 * 其方法等同于opencv的cv::resize方法。
 * 注意：该方法没有提供反向传播
 * 如果需要使用反向传播，请使用RoiPooling/RoiAlign层。
 * 注意：RoiResize只支持一个输入box。
 */

template <typename Dtype>
class RoiResizeLayer : public Layer<Dtype> {
 public:
  explicit RoiResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RoiResize"; }

  /**
   * bottom[0] -> [N,C,H,W]
   * bottom[1] -> [1,1,1,4] (BoundingBox)
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0]: -> [N,C,RH,RW]
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 输出的尺度    
  int targetSpatialWidth_;
  int targetSpatialHeight_;
};

}

#endif
