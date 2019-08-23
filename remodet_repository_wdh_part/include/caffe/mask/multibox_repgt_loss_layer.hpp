#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

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
 * 该层作为SSD的检测器损失层。
 * 该层实现了SSD的训练损失过程。
 * 该层的主要原理包括：
 * （１）prior-boxes和GT-boxes之间的匹配过程；
 * （２）根据匹配获得正例列表和反例列表；
 * （３）采用HDM方法获得保留的反例列表；
 * （４）加载loc/conf的数据
 * （５）loc/conf前向计算
 * （６）loc/conf反向计算
 * （７）根据正例和反例列表传播误差到bottom-blobs
 * 该层已经在mask/bbox_loss_layer中rewritten，可以参考该类的实现。
 * 注意：该类仅用于SSD检测器的训练。
 */

template <typename Dtype>
class MultiBoxRepgtLossLayer : public LossLayer<Dtype> {
 public:
  //  使用损失层的构造函数
  explicit MultiBoxRepgtLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBoxRepgtLoss"; }

  /**
   * bottom[0]: -> [N,H*W*Np*4] loc predictions
   * bottom[1]: -> [N,H*W*Np*(Nc+1)] conf predictions
   * bottom[2]: -> [1,2,H*W*Np*4] prior boxes and variances
   * bottom[3]: -> [1,1,Ng,8] gt-boxes
   */
  virtual inline int ExactNumBottomBlobs() const { return 4; }

  /**
   * top[0]: loss
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Loc损失层
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  // 损失类型：SMOOTHL1 or L2
  LocLossType loc_loss_type_;
  // 回归器误差权值
  float loc_weight_;
  // 回归器的估计值
  Blob<Dtype> loc_pred_;
  // 回归器的gt值
  Blob<Dtype> loc_gt_;
  // 回归器的Loss
  Blob<Dtype> loc_loss_;
  // 回归器的bottom/top－blobs指针列表
  vector<Blob<Dtype>*> loc_bottom_vec_;
  vector<Blob<Dtype>*> loc_top_vec_;

  // 分类器
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  // 分类器类型：softmax or logistic
  ConfLossType conf_loss_type_;
  // 分类器误差权值
  float conf_weight_;
  // 分类估计值
  Blob<Dtype> conf_pred_;
  // 分类器gt值
  Blob<Dtype> conf_gt_;
  // 分类器loss值
  Blob<Dtype> conf_loss_;
  // 分类器的bottom/top-blobs指针
  vector<Blob<Dtype>*> conf_bottom_vec_;
  vector<Blob<Dtype>*> conf_top_vec_;

  // 参数定义
  // 类别数
  int num_classes_;
  // deprecated.　默认为TRUE
  bool share_location_;
  // 匹配方法:BIPARTITE或者PER_PREDICTION
  // 默认为PER_PREDICTION
  MatchType match_type_;
  // 正例的IOU阈值，超过则为正例
  float overlap_threshold_;
  // 默认为TRUE
  bool use_prior_for_matching_;
  // deprecated,默认为０
  int background_label_id_;
  // 是否使用标记为diff的gt-boxes，默认为FALSE
  bool use_difficult_gt_;
  // 默认为TRUE, deprecated
  bool do_neg_mining_;
  // 使用Hard Negative Mining的反例与正例的数量之比
  float neg_pos_ratio_;
  // 反例的IOU上限，低于该值则为反例
  float neg_overlap_;
  // box的类型，默认为CENTER_SIZE
  CodeType code_type_;
  // 默认为FALSE, deprecated
  bool encode_variance_in_target_;
  // 默认为FALSE,为TRUE则为前景／背景二分类， deprecated
  bool map_object_to_agnostic_;
  // 默认为１. deprecated
  int loc_classes_;
  // gt数量
  int num_gt_;          //used for LOC
  // 样本数
  int num_;
  // prior boxes数量
  int num_priors_;
  // 正例数量
  int num_pos_;         //num of matches
  // 反例数量
  int num_neg_;    //num of negs
  // 用于分类器的样本数，为正例和反例之和
  int num_conf_;        //num of conf

  // 整个batch样本的匹配列表
  vector<map<int, int> > all_pos_indices_;
  // 整个batch样本的反例列表（Hard Negative Mining）
  vector<vector<int> > all_neg_indices_;

  // 损失的正则化模式:FULL/VALID/BATCHSIZE/NONE四种方法
  // 默认为VALID
  LossParameter_NormalizationMode normalization_;

  // 类别名称
  // rsvd.
  string name_to_label_file_;
  map<string, int> name_labels_;
};

}

#endif
