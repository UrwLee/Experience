#ifndef CAFFE_PRIORBOX_LAYER_HPP_
#define CAFFE_PRIORBOX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为某个特征Layer的检测器提供了所有Anchors的方法。
 * Anchors -> 等价于prior-boxes
 * 假定输入Map的尺度：　[N,C,H,W]
 * 每个点提出Np个prior-boxes，那么检测器的Loc -> num_outputs = Np*4 and Conf -> num_outputs = Np*(Nc+1)
 * 因此，需要为每个prior-box给定其实际坐标：
 * (i,j,n) {i=0~H-1, j=0~W-1, n=0~Np-1} 的实际坐标是：
 *      [#] cx = (j+0.5)/W and cy = (i+0.5)/H
 *      [#] pw = pro_width(n) and ph = pro_height(n)
 *      注意：pro_width/pro_height是外部定义的prior-boxes的归一化尺度列表
 * (i,j,n)唯一定义了一个prior-box。
 * 每个prior-box都有自身的编码信息：variance，每个坐标都具有，一般定义为：
 * 针对于CORNER: [xmin,ymin,xmax,ymax] -> [0.1,0.1,0.1,0.1]
 * 针对于CENTER: [cx,cy,pw,ph] -> [0.1,0.1,0.2,0.2]
 * 因此，prior-box-layer提供了每个prior-box的坐标信息以及编码信息，因此其输出为：
 * top[0]: -> [1,2,H*W*Np*4]
 * 第一行：　[[xmin,ymin,xmax,ymax]] x H*W*Np
 * 第二行：　[[var1,var2,var3,var4]] x H*W*Np
 */

template <typename Dtype>
class PriorBoxLayer : public Layer<Dtype> {
 public:

  explicit PriorBoxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PriorBox"; }

  /**
   * bottom[0]: -> [N,C,H,W] (featureMap)
   * bottom[1]: -> [N,3,IH,IW] (original data) (unused now, could be ignored)
   */
  virtual inline int ExactBottomBlobs() const { return 2; }

  /**
   * top[0]: -> [1,2,H*W*Np*4] (prior-boxes locations and variances)
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  // proposed widths and heights for Anchors (prior-boxes)
  vector<float> pro_widths_;
  vector<float> pro_heights_;
  // minsizes for RPN (unused)
  vector<float> min_size_;
  // maxsizes for RPN (unused)
  vector<float> max_size_;
  // aspects for RPN (unused)
  vector<float> aspect_ratios_;
  // if aspect_ratio need to be flipped
  // TRUE by default
  bool flip_;
  // if bboxes need to be clipped
  // TRUE by default
  bool clip_;
  // num of prior-boxes per location
  // as Np
  int num_priors_;

  // [0.1,0.1,0.2,0.2] for CENTER_SIZE
  // [0.1,0.1,0.1,0.1] for CORNER
  vector<float> variance_;
};

}

#endif
