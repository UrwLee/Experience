#ifndef CAFFE_MASK_MASK_GEN_LAYER_HPP_
#define CAFFE_MASK_MASK_GEN_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类为每个实例（ROI）生成对应的Mask-Label-Map (binary Mask for every class)
 * 方法如下：
 * １．生成每个实例对应样本的Mask图：cv::Mat
 * ２．根据ROI位置进行裁剪
 * ３．resize为需要输出的尺寸
 */

template <typename Dtype>
class MaskGenLayer : public Layer<Dtype> {
 public:
  explicit MaskGenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskGen"; }

  /**
   * bottom[0]: -> ROI instance (1,1,Nroi,7)
   * bottom[1]: -> Mask data for each gt-box  (1,1,Ng,6+H*W) where H and W are height and width of input image
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0]: -> Mask Label Map for each ROI (Nroi,1,RH,RW)
   * top[1]: -> Active flags for each ROI (1,1,Nroi,1)
   * 注意：如果实例并没有标注Mask，则Active Flag = 0，对应的Mask-Label-Map全部是0
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  /**
   * height_/width_: -> 输入image的尺寸
   * resized_height_/resized_width_: -> Mask-Label-Map的resize尺寸
   */
  int height_, width_;
  int resized_height_, resized_width_;
};

}  // namespace caffe

#endif
