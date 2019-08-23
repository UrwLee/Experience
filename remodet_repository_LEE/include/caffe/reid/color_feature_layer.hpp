#ifndef CAFFE_REID_COLOR_FEATURE_LAYER_HPP_
#define CAFFE_REID_COLOR_FEATURE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层提供了提取对ROI和结构化颜色统计信息的方法。
 * 该统计方法仅作为参考，请在掌握源码基础上进行适当修改使用。
 * 禁止直接使用该代码。
 */

template <typename Dtype>
class ColorFeatureLayer : public Layer<Dtype> {
 public:
  explicit ColorFeatureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ColorFeature"; }

  /**
   * bottom[0] -> image data  [1,3,H,W]
   * bottom[1] -> proposals   [1,1,N,61] come from pose_det_layer
   *  * 0-3: bbox
      * 4-57: kps <x,y,v>
      * 58: num_vis
      * 59: score
      * 60: id (-1)
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0] -> [1,N,11,512]
   *  * 0 :-> <Neck-Throat>
   *  * 1 :-> <RS-RE>
   *  * 2 :-> <RE-RW>
   *  * 3 :-> <LS-LE>
   *  * 4 :-> <LE-LW>
   *  * 5 :-> <RH-RK>
   *  * 6 :-> <RK-RA>
   *  * 7 :-> <LH-LK>
   *  * 8 :-> <LK-LA>
   *  * 9 :-> <Torso: LS-RS-RH-LH>
   *  * 10 :-> <BBox: xmin,ymin,xmax,ymax>
   *  Color Dist: (8,8,8) bins
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}

#endif
