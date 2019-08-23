#ifndef CAFFE_MASK_BBOX_LOSS_LAYER_HPP_
#define CAFFE_MASK_BBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * 检测器损失层，内建一个坐标回归层和一个分类层
 */

template <typename Dtype>
class BBoxLossLayer : public LossLayer<Dtype> {
 public:
  explicit BBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BBoxLoss"; }

  // bottom[0] stores the location predictions.
  // bottom[1] stores the confidence predictions.
  // bottom[2] stores the prior bounding boxes.
  // bottom[3] stores the ground truth bounding boxes.
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 4; }
  virtual inline int MaxBottomBlobs() const { return 5; }

  // top[0] -> loss [1]
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /**
   * loc_loss_layer_: -> 坐标回归层
   * loc_loss_type_: -> 回归损失，默认L1
   * loc_weight_: 误差增益
   * loc_pred_:　坐标估计的Blob
   * loc_gt_: 坐标GT的Blob
   * loc_loss_: 输出损失Blob (1)
   * loc_bottom_vec_: 该层输入Bottom列表
   * loc_top_vec_:　该层输出Top列表
   */
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  LocLossType loc_loss_type_;
  Dtype loc_weight_;
  Blob<Dtype> loc_pred_;
  Blob<Dtype> loc_gt_;
  Blob<Dtype> loc_loss_;
  vector<Blob<Dtype>*> loc_bottom_vec_;
  vector<Blob<Dtype>*> loc_top_vec_;

  /**
   * conf_loss_layer_:分类损失层
   * conf_loss_type_:分类损失类型，默认是softmax　
   * conf_weight_:误差增益
   * conf_pred_:置信度估计Blob
   * conf_gt_:置信度GT-Blob
   * conf_loss_:置信度损失　(1)
   * conf_bottom_vec_:该层输入Bottom列表　
   * conf_top_vec_:该层输出Top列表
   */
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  ConfLossType conf_loss_type_;
  Dtype conf_weight_;
  Blob<Dtype> conf_pred_;
  Blob<Dtype> conf_gt_;
  Blob<Dtype> conf_loss_;
  vector<Blob<Dtype>*> conf_bottom_vec_;
  vector<Blob<Dtype>*> conf_top_vec_;

  // 参数定义
  // 总类别：N+1, N是实际检测的有效类别数
  int num_classes_;

  vector<int> gt_labels_;
  map<int,int> target_labels_;

  // deprecated.　默认是TRUE
  bool share_location_;

  // 匹配方法:BIPARTITE或者PER_PREDICTION，默认是是PER_PREDICTION
  // PER_PREDICTION -> 耗尽型匹配，每个Prior都需要遍历所有GT进行匹配，以确定它是正例还是反例
  // BIPARTITE -> 1v1匹配，每个GT找到一个最佳Prior进行匹配即可
  MatchType match_type_;

  // 正例匹配IOU阈值：下限
  Dtype overlap_threshold_;

  // 匹配是否使用prior_bboxes，默认是TRUE
  bool use_prior_for_matching_;

  // 是否使用标记为diff的GT-Box，默认是None
  bool use_difficult_gt_;

  // 是否使用Hard Negative Mining来限制反例的数量，默认是TRUE
  bool do_neg_mining_;

  // 如果使用Hard Negative Mining，定义反例和正例的数量之比
  Dtype neg_pos_ratio_;

  // 反例的IOU阈值：上限
  Dtype neg_overlap_;

  // Boxes的定义方式：CORNER / CENTER，默认是CENTER
  CodeType code_type_;

  // 编码增益是否已经在目标中实现，默认是False，我们在程序中额外实现
  bool encode_variance_in_target_;

  // 是否做二分类问题：默认是False
  bool map_object_to_agnostic_;

  // 默认是１，不要修改
  int loc_classes_;

  // alias_id
  int alias_id_;

  // gt数量
  int num_gt_;
  // 样本数
  int num_;
  // 每个样本的prior_bboxes数量
  int num_priors_;

  // 正例数量
  int num_pos_;
  // 反例数量
  int num_neg_;
  // 分类器的数量：为正例与反例之和
  int num_conf_;
    // whether to use the images those have no person as background
  bool flag_noperson_;

  /**
   * 正例匹配列表：
   * map<int=prior-id, int=gt-id>
   */
  vector<map<int, int> > all_pos_indices_;
  /**
   * 反例匹配列表
   * vector<int=prior-id>
   */
  vector<vector<int> > all_neg_indices_;

  // NORMALIZE模式设置：VALID
  LossParameter_NormalizationMode normalization_;

  // size thre
  Dtype size_threshold_;

  // FocusLoss
  bool using_focus_loss_;  // default: False
  float gama_;               // if use FocusLoss, define the gama-parameter
  float alpha_;               // if use FocusLoss, define the gama-parameter
  vector<bool> flags_of_anchor_;
  bool flag_checkanchor_;
  bool ShowAnchorNumVsScale_;  
  vector<int> gt_anchornum_scale_total_;
  vector<int> gt_total_;
};

}  // namespace caffe

#endif
