#ifndef CAFFE_RESIZE_LAYER_HPP_
#define CAFFE_RESIZE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层提供了将Blob进行Resize的方法。
 * 类似于opencv，默认插值Resize的插值方法为LINEAR
 * 该层没有提供反向方法，请谨慎使用。
 */

template <typename Dtype>
class ResizeBlobLayer : public Layer<Dtype> {
 public:
  explicit ResizeBlobLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ResizeBlob"; }

  /**
   * bottom[0]: [N,C,H,W]
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: [N,C,RH,RW]
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

  // 目标resize尺寸    
  int targetSpatialWidth_;
  int targetSpatialHeight_;

};

}

#endif
