#ifndef CAFFE_MULTIMCBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIMCBOX_LOSS_LAYER_HPP_

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
 * 该层为YOLO/SSD混合检测器的损失模型。
 * 注意：该层使用了YOLO的分类器，采用了SSD的匹配方法和回归器。
 * 请在了解源码基础上使用。
 * 不建议使用该层。
 */

template <typename Dtype>
class MultiMcBoxLossLayer : public LossLayer<Dtype> {
 public:
  //  使用损失层的构造函数
  explicit MultiMcBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiMcBoxLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> conf_diff_;
  // 内建Loc损失层
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  LocLossType loc_loss_type_;
  float loc_weight_;
  Blob<Dtype> loc_pred_;
  Blob<Dtype> loc_gt_;
  Blob<Dtype> loc_loss_;
  vector<Blob<Dtype>*> loc_bottom_vec_;
  vector<Blob<Dtype>*> loc_top_vec_;
  // 内建body-分类损失层
  ConfLossType conf_loss_type_;
  float conf_weight_;
  float conf_loss_;


  // 参数定义
  int num_classes_;
  // deprecated.
  bool share_location_;
  // 匹配方法:BIPARTITE或者PER_PREDICTION
  MatchType match_type_;
  float overlap_threshold_;
  bool use_prior_for_matching_;
  // Note: must be zero, deprecated
  int background_label_id_;
  bool use_difficult_gt_;
  // Note: must be true, deprecated
  bool do_neg_mining_;
  float neg_pos_ratio_;
  // Note: always = overlap_threshold_
  float neg_overlap_;
  // CENTOR or CORNER
  CodeType code_type_;
  // Note: must be false, deprecated
  bool encode_variance_in_target_;
  // Note: must be false, deprecated
  bool map_object_to_agnostic_;
  //Note: must be 1. deprecated
  int loc_classes_;
  // gt数量
  int num_gt_;          //used for LOC
  // 样本数
  int num_;
  int num_priors_;
  // 正例数量
  int num_pos_;         //num of matches
  int num_neg_;    //num of negs
  int num_conf_;        //num of conf

  vector<map<int, int> > all_pos_indices_;
  vector<vector<int> > all_neg_indices_;

  // 损失的正则化模式:FULL/VALID/BATCHSIZE/NONE四种方法
  LossParameter_NormalizationMode normalization_;

  //name
  string name_to_label_file_;
  map<string, int> name_labels_;

  bool rescore_;

  Dtype object_scale_;
  Dtype noobject_scale_;
  Dtype class_scale_;
  Dtype loc_scale_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIMCBOX_LOSS_LAYER_HPP_
