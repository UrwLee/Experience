#ifndef CAFFE_MCBOX_LOSS_LAYER_HPP_
#define CAFFE_MCBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * 该层为YOLO检测器的损失层。
 * 该层主要实现了YOLO检测器的训练过程。
 * 匹配过程与SSD检测器的区别如下：
 * （１）匹配不再按照IOU判定，而是按照中心匹配原则；
 * （２）box的编解码方式不再使用固定增益，而是logistic/log方式；
 * （３）分类器不再使用(N+1)-Softmax分类器，而是使用如下方式：
 *      [1] logistic二分类，进行前景／背景分类
 *      [2] softmax进行N分类
 * （４）对于IOU超过阈值但非中心匹配的box,不传播分类和回归误差；
 * 具体实现参考源码。
 * 注意：请在熟练掌握源码基础上使用。
 * 仅作为YOLO检测器训练使用，不得在其他场景下使用该层。
 */

template <typename Dtype>
class McBoxLossLayer : public LossLayer<Dtype> {
 public:
  //  使用损失层的构造函数
  explicit McBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "McBoxLoss"; }

  /**
   * bottom[0]: -> [N,Np*4,H,W], loc predictions
   * bottom[1]: -> [N,Np*(Nc+1),H,W], conf predictions
   * bottom[2]: -> [1,1,Ng,8], gt labels
   */
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  /**
   * loss
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Blobs: 存储回归器和分类器误差
  Blob<Dtype> loc_diff_;
  Blob<Dtype> conf_diff_;

  // 参数定义
  // 样本数
  int num_;
  // GT数
  int num_gt_;
  // 分类树
  int num_classes_;
  // prior boxes数
  int num_priors_;
  // 0-默认值
  int background_label_id_;
  // IOU阈值：超过该阈值的非中心匹配boxes，将不反向传播任何误差
  Dtype overlap_threshold_;
  // 是否使用prior boxes匹配
  bool use_prior_for_matching_;
  // 默认是FALSE
  bool use_prior_for_init_;
  // 是否加载标记为gt的boxes
  bool use_difficult_gt_;
  // 是否使用实际的IOU作为分类器的label
  bool rescore_;
  // 当前迭代数，默认是０
  int iters_;
  // 不适用。
  int iter_using_bgboxes_;
  // 编码方式，CENTER_SIZE
  CodeLocType code_loc_type_;

  // 误差参数
  // 背景box的回归，默认为０，不传误差
  Dtype background_box_loc_scale_;
  // 前景对象的分类误差增益
  Dtype object_scale_;
  // 背景对象的分类误差增益
  Dtype noobject_scale_;
  // softmax分类误差的增益
  Dtype class_scale_;
  // 坐标回归器的误差增益
  Dtype loc_scale_;
  // prior boxes的长宽值
  vector<Dtype> prior_width_;
  vector<Dtype> prior_height_;

  // 误差传播模式，一般为VALID
  LossParameter_NormalizationMode normalization_;

  // 匹配数
  int match_count_;
  // 是否裁剪prior-boxes，默认是TRUE
  bool clip_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
