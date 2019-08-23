#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/layers/multimc_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void MultiMcBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 使用损失层的设置函数
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // 重新设置输入的反向计算
  // bot[0] -> loc 需要反向
  // bot[1] -> conf 需要反向
  // bot[2]/[3] -> prior/gt不需要反向
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  // 获取multibox的损失参数
  const MultiMcBoxLossParameter& multimcbox_loss_param =
      this->layer_param_.multimcbox_loss_param();
  // 定义样本数
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();

  CHECK(multimcbox_loss_param.has_num_classes());
  num_classes_ = multimcbox_loss_param.num_classes() - 1;

  share_location_ = multimcbox_loss_param.share_location();  // deprecated!
  CHECK(share_location_) << "Only share_location is supportted in current version.";
  if (share_location_) {
    loc_classes_ = 1;
  } else {
    loc_classes_ = multimcbox_loss_param.loc_class();   // deprecated!
    CHECK_EQ(loc_classes_, 1) << "loc_classes must be 1 in current version.";
  }

  match_type_ = multimcbox_loss_param.match_type();
  overlap_threshold_ = multimcbox_loss_param.overlap_threshold();
  use_prior_for_matching_ = multimcbox_loss_param.use_prior_for_matching();
  //use_prior_for_init_ = multimcbox_loss_param.use_prior_for_init();
  background_label_id_ = multimcbox_loss_param.background_label_id();
  CHECK_EQ(background_label_id_, 0) << "background_label_id must be zero.";
  use_difficult_gt_ = multimcbox_loss_param.use_difficult_gt();
  do_neg_mining_ = multimcbox_loss_param.do_neg_mining();
  CHECK(do_neg_mining_) << "negative mining must be enabled.";
  neg_pos_ratio_ = multimcbox_loss_param.neg_pos_ratio();
  //CHECK_GE(neg_pos_ratio_, 1) << "neg_pos_ratio must ge greater than 1.";
  neg_overlap_ = multimcbox_loss_param.neg_overlap();
  code_type_ = multimcbox_loss_param.code_type();
  encode_variance_in_target_ = multimcbox_loss_param.encode_variance_in_target();

  rescore_ = multimcbox_loss_param.rescore();
  //code_loc_type
  //code_loc_type_ = multimcbox_loss_param.code_loc_type();
  // background_box_loc_scale_
  //background_box_loc_scale_ = multimcbox_loss_param.background_box_loc_scale();
  // loss param
  object_scale_ = multimcbox_loss_param.object_scale();
  noobject_scale_ = multimcbox_loss_param.noobject_scale();
  class_scale_ = multimcbox_loss_param.class_scale();
  loc_scale_ = multimcbox_loss_param.loc_scale();

  map_object_to_agnostic_ = multimcbox_loss_param.map_object_to_agnostic();
  CHECK(!map_object_to_agnostic_) << "Map-object-to-agnostic is not supportted in current version.";

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  /**
   * 构建LOC损失层
   */
  vector<int> loss_shape(1, 1);
  // loc的权重系数
  loc_weight_ = multimcbox_loss_param.loc_weight();
  loc_loss_type_ = multimcbox_loss_param.loc_loss_type();
  vector<int> loc_shape(1, 1);
  loc_shape.push_back(4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
    layer_param.set_type("SmoothL1Loss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }
}

template <typename Dtype>
void MultiMcBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //add loss
  vector<int> top_shape(4,1);
  // top_shape[3] = 3;
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();
  // [n,h,w,num_priors_*4]
  // [n,h,w,num_priors_*(num_classes_+1)]
  CHECK_EQ(num_priors_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * (num_classes_+1), bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // diff_
  //loc_diff_.ReshapeLike(*bottom[0]);
  conf_diff_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void MultiMcBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();
  // LOG(INFO) << "[Batch ID  ] -> " << *(gt_data+2);
  // LOG(INFO) << "[Loss Layer] -> batchsize =  " << num_;
  // LOG(INFO) << "[Loss Layer] -> num_priors = " << num_priors_;
  // LOG(INFO) << "[Loss Layer] -> num_gts =    " << num_gt_;
  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  // int -> 样本号
  // vector<> -> 每个样本对应的gt-boxes列表
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);
  // print INFO of gt
  // for (map<int, vector<NormalizedBBox> >::iterator it = all_gt_bboxes.begin();
  //      it != all_gt_bboxes.end(); ++it) {
  //     int item = it->first;
  //     const vector<NormalizedBBox>& gt_boxes = it->second;
  //     LOG(INFO) << "[Ground Truth] Item " << item
  //               << " has found " << gt_boxes.size() << " persons.";
  // }

  /**************************************************************************#
  Step-0: Logistic and Softmax
  #***************************************************************************/
  Dtype* mutable_conf_data = bottom[1]->mutable_cpu_data();
  // 处理object分类 以及 softmax
  for (int item_id = 0; item_id < num_; ++item_id) {
    for (int n = 0; n < num_priors_; ++n) {
      int idx_conf = item_id * num_priors_ * (num_classes_ + 1) + n * (num_classes_ + 1);
      // object分类：logistic
      mutable_conf_data[idx_conf] = logistic(conf_data[idx_conf]);
      // conf分类：softmax
      Softmax(conf_data + idx_conf + 1, num_classes_, mutable_conf_data + idx_conf + 1);
    }
  }
  Dtype* conf_diff = conf_diff_.mutable_cpu_data();
  //caffe_set(loc_diff_.count(), Dtype(0), loc_diff);
  caffe_set(conf_diff_.count(), Dtype(0), conf_diff);

  //Dtype loss = 0;
  //Dtype loc_loss = 0;
  //Dtype conf_loss = 0;
  //Dtype softmax_loss = 0.0;

  /**************************************************************************#
  获取所有的prior-boxes
  #***************************************************************************/
  // 获取所有priorboxes的坐标信息
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  /**************************************************************************#
  获取所有的Box坐标估计
  #***************************************************************************/
  // 获取LOC的估计信息
  // all_loc_preds[item_id][prior_bbox_id] = NormalizedBBox();
  vector<vector<NormalizedBBox> > all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, &all_loc_preds);

  /**************************************************************************#
  获取分类任务的最大置信度信息:
  #***************************************************************************/
  vector<vector<float> > max_scores;
  conf_data = bottom[1]->cpu_data();
  GetMaxConfidenceScores_yolo(conf_data, num_, num_priors_, num_classes_,
                         background_label_id_, 
                         1, num_classes_, &max_scores);
  num_pos_ = 0;
  num_neg_ = 0;
  /**************************************************************************#
  完成所有任务的匹配统计: 正例统计/反例统计
  #***************************************************************************/
  for (int i = 0; i < num_; ++i) {
    /**************************************************************************#
    单个样本的匹配列表
    #***************************************************************************/
    // 匹配映射表
    map<int, int > match_indices;
    // 未匹配索引
    vector<int> unmatch_indices;
    vector<int> neg_indices;
    // 所有prior-boxes的最大交集列表: 每个prior-box与所有gtbox之间的iou最大值
    vector<float> match_overlaps;
    // 如果当前样本没有gtbox,直接返回
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      all_pos_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
      continue;
    }
    // 获得该样本的gt-box列表
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    // 使用实际的检测box进行匹配
    if (!use_prior_for_matching_) {
      vector<NormalizedBBox> loc_bboxes;
      // 先解码检测box
      DecodeBBoxes(prior_bboxes, prior_variances,
                   code_type_, encode_variance_in_target_,
                   all_loc_preds[i], &loc_bboxes);
      // 完成匹配
      MatchBBox(gt_bboxes, loc_bboxes, match_type_, overlap_threshold_,
                &match_overlaps, &match_indices, &unmatch_indices);

    } else {
      // 使用prior进行匹配,并完成匹配
      MatchBBox(gt_bboxes, prior_bboxes, match_type_, overlap_threshold_,
                &match_overlaps, &match_indices, &unmatch_indices);
    }
    /**************************************************************************#
    匹配结束,统计当前样本的各项匹配列表
    #***************************************************************************/
    int num_pos = match_indices.size();
    num_pos_ += num_pos;
    /**************************************************************************#
    统计反例列表
    #***************************************************************************/
    vector<pair<float, int> > scores_indices;
    int num_neg = 0;
    for (int j = 0; j < unmatch_indices.size(); ++j) {
      if (match_overlaps[unmatch_indices[j]] < neg_overlap_) {
        num_neg++;
        scores_indices.push_back(std::make_pair(
                      max_scores[i][unmatch_indices[j]],
                      unmatch_indices[j]));
      }
    }
    if(neg_pos_ratio_ > 0){//-1 all
      num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
      std::sort(scores_indices.begin(), scores_indices.end(),
                SortScorePairDescend<int>);
    }
    for (int n = 0; n < num_neg; ++n) {
      neg_indices.push_back(scores_indices[n].second);
    }
    num_neg_ += num_neg;

    /**************************************************************************#
    将正例和反例列表保存到batch的索引记录之中
    #***************************************************************************/
    all_pos_indices_.push_back(match_indices);
    all_neg_indices_.push_back(neg_indices);
  }//所有样本的匹配过程结束

  /**************************************************************************#
  用于置信度计算的数量
  #***************************************************************************/
  num_conf_ = num_pos_ + num_neg_;
  /**************************************************************************#
  LOC损失计算
  #***************************************************************************/
  if (num_pos_ >= 1) {
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_pos_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      if (all_pos_indices_[i].size() == 0) continue;
      for (map<int,int>::iterator it = all_pos_indices_[i].begin();
           it != all_pos_indices_[i].end(); ++it) {
        const int prior_id = it->first;
        const int gt_id = it->second;
        CHECK_LT(prior_id, num_priors_);
        CHECK_LT(gt_id, all_gt_bboxes[i].size());
        const vector<NormalizedBBox>& loc_pred = all_loc_preds[i];
        CHECK_LT(prior_id, loc_pred.size());
        // 复制数据
        loc_pred_data[count * 4] = loc_pred[prior_id].xmin();
        loc_pred_data[count * 4 + 1] = loc_pred[prior_id].ymin();
        loc_pred_data[count * 4 + 2] = loc_pred[prior_id].xmax();
        loc_pred_data[count * 4 + 3] = loc_pred[prior_id].ymax();
        const NormalizedBBox& gt_bbox = all_gt_bboxes[i][gt_id];
        NormalizedBBox gt_encode;
        EncodeBBox(prior_bboxes[prior_id], prior_variances[prior_id], code_type_,
                   encode_variance_in_target_, gt_bbox, &gt_encode);
        // 复制label
        loc_gt_data[count * 4] = gt_encode.xmin();
        loc_gt_data[count * 4 + 1] = gt_encode.ymin();
        loc_gt_data[count * 4 + 2] = gt_encode.xmax();
        loc_gt_data[count * 4 + 3] = gt_encode.ymax();
        if (encode_variance_in_target_) {
          for (int k = 0; k < 4; ++k) {
            CHECK_GT(prior_variances[prior_id][k], 0);
            loc_pred_data[count * 4 + k] /= prior_variances[prior_id][k];
            loc_gt_data[count * 4 + k] /= prior_variances[prior_id][k];
          }
        }
        ++count;
      }
    }
    CHECK_EQ(count, num_pos_) << "Unmatch numbers of Loc-Layer.";
    //loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  } else {
    loc_loss_.mutable_cpu_data()[0] = 0;
  }

  /**************************************************************************#
  conf损失计算
  #***************************************************************************/
  if (num_conf_ >= 1) {
    //Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    //Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    //caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      // 正例
      if (all_pos_indices_[i].size() == 0) continue;
      const map<int, int>& match_indices = all_pos_indices_[i];
      for (map<int, int>::const_iterator it =
           match_indices.begin(); it != match_indices.end(); ++it) {
        const int prior_id = it->first;
        CHECK_LT(prior_id, num_priors_);
        const int gt_id = it->second;
        CHECK_LT(gt_id, all_gt_bboxes[i].size());
        int gt_label = all_gt_bboxes[i][gt_id].label();
        CHECK_GT(gt_label, 0) << "Label error in gt_boxes.";
        int conf_id = i * num_priors_ * (num_classes_ + 1) + prior_id * (num_classes_ + 1);
        if (rescore_){
          float iou = JaccardOverlap(all_loc_preds[i][prior_id], all_gt_bboxes[i][gt_id]);
          conf_diff[conf_id] =
            object_scale_ * (conf_data[conf_id] - iou) * logistic_gradient(conf_data[conf_id]);
        }else{
          conf_diff[conf_id] =
            object_scale_ * (conf_data[conf_id] - 1.0) * logistic_gradient(conf_data[conf_id]);          
        }
        for (int c = 1; c < num_classes_ + 1; ++c) {
          conf_diff[conf_id + c] =
            class_scale_ * (conf_data[conf_id + c] - ((c == gt_label)? 1 : 0));
            //softmax_loss = softmax_loss + conf_diff[best_conf_idx + c] * conf_diff[best_conf_idx + c];
        }        
        // switch (conf_loss_type_) {
        //   case MultiBoxLossParameter_ConfLossType_SOFTMAX:
        //     conf_gt_data[count] = gt_label;
        //     break;
        //   case MultiBoxLossParameter_ConfLossType_LOGISTIC:
        //     conf_gt_data[count * num_classes_ + gt_label] = 1;
        //     break;
        //   default:
        //     LOG(FATAL) << "Unknown conf loss type.";
        // }
        // caffe_copy<Dtype>(num_classes_, conf_data + prior_id * num_classes_,
        //                   conf_pred_data + count * num_classes_);
        ++count;
      }
      // 反例
      const vector<int>& neg_indices = all_neg_indices_[i];
      if (neg_indices.size() == 0) continue;
      for (int j = 0; j < neg_indices.size(); ++j) {
        const int prior_id = neg_indices[j];
        CHECK_LT(prior_id, num_priors_);
        int conf_id = i * num_priors_ * (num_classes_ + 1) + prior_id * (num_classes_ + 1);
        conf_diff[conf_id] =
            noobject_scale_ * (conf_data[conf_id] - 0) * logistic_gradient(conf_data[conf_id]);
        // 复制data数据
        // caffe_copy<Dtype>(num_classes_, conf_data + prior_id * num_classes_,
        //                   conf_pred_data + count * num_classes_);
        // 复制gt数据
        // switch (conf_loss_type_) {
        //   case MultiBoxLossParameter_ConfLossType_SOFTMAX:
        //     conf_gt_data[count] = background_label_id_;
        //     break;
        //   case MultiBoxLossParameter_ConfLossType_LOGISTIC:
        //     conf_gt_data[count * num_classes_ + background_label_id_] = 1;
        //     break;
        //   default:
        //     LOG(FATAL) << "Unknown conf loss type.";
        // }
        ++count;
      }
      // 指向下一个样本
      //conf_data += bottom[1]->offset(1);
    }
    CHECK_EQ(count, num_conf_) << "Unmatch numbers of conf-Layer.";
  } else {
    conf_loss_ = 0;
  }

  /**************************************************************************#
  损失累加：由于损失层的NORMALIZE设置为NONE（1），因此需要再此做归一化
  后面的反向传播计算中的SCALE操作也是因为此.
  #***************************************************************************/
  top[0]->mutable_cpu_data()[0] = 0;
  conf_loss_ = conf_diff_.sumsq_data();
  // LOC损失
  if (this->layer_param_.propagate_down(0)) {
    // FULL：num_ x num_priors_
    // BATCHSIZE：num_
    // VALID：num_matches_
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_pos_);
    //YJH
    // LOG(INFO) << "loc match" << num_pos_loc_;
    //LOG(INFO) << loc_loss_.cpu_data()[0];
    //LOG(INFO) << "loc match" << num_pos_ << "loc loss" << loc_weight_ * loc_loss_.cpu_data()[0] / normalizer << "norm " << normalizer;
    top[0]->mutable_cpu_data()[0] +=
        loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
  }
  // conf损失
  //LOG(INFO) << conf_loss_;
  if (this->layer_param_.propagate_down(1)) {
    // conf
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_conf_);
    //YJH
    // LOG(INFO) << "conf match" << num_conf_body_;
    //LOG(INFO) << "conf match " << num_conf_ << "conf loss " << conf_weight_ * conf_loss_ / normalizer << "norm " << normalizer;
    top[0]->mutable_cpu_data()[0] +=
        conf_weight_ * conf_loss_ / normalizer;
  }
}


template <typename Dtype>
void MultiMcBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  /**************************************************************************#
  LOC反向传播
  #***************************************************************************/
  if (propagate_down[0]) {
    // 获取LOC的误差指针
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
    // 初始误差0
    caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
    if (num_pos_ >= 1) {
      vector<bool> loc_propagate_down;
      loc_propagate_down.push_back(true);
      loc_propagate_down.push_back(false);
      loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                                loc_bottom_vec_);
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_pos_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      loss_weight *= loc_weight_;
      caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
      const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        if (all_pos_indices_[i].size() == 0) continue;
        const map<int, int>& match_indices = all_pos_indices_[i];
        for (map<int, int>::const_iterator it = match_indices.begin();
             it != match_indices.end(); ++it) {
          const int prior_id = it->first;
          CHECK_LT(prior_id, num_priors_);
          caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
                            loc_bottom_diff + prior_id * 4);
          ++count;
        }
        loc_bottom_diff += bottom[0]->offset(1);
      }
      CHECK_EQ(count, num_pos_);
    }
  }
  /**************************************************************************#
  Conf反向传播
  #***************************************************************************/
  if (propagate_down[1]) {
    const Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
       normalization_, num_, num_priors_, num_conf_);
    const Dtype alpha = top[0]->cpu_diff()[0] / normalizer;
    const Dtype* diff_data = NULL;
    diff_data = conf_diff_.cpu_data();
    caffe_cpu_axpby(
        bottom[1]->count(),
        alpha,
        diff_data,
        Dtype(0),
        bottom[1]->mutable_cpu_diff());
  }
  /**************************************************************************#
  清除所有匹配列表
  #***************************************************************************/
  all_pos_indices_.clear();
  all_neg_indices_.clear();
}

INSTANTIATE_CLASS(MultiMcBoxLossLayer);
REGISTER_LAYER_CLASS(MultiMcBoxLoss);

}  // namespace caffe
