#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/mask/bbox_loss_layer_odm.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BBoxLossODMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const BBoxLossParameter& bbox_loss_param =
    this->layer_param_.bbox_loss_param();
  // check parameters
  CHECK(bbox_loss_param.has_num_classes());
  num_classes_ = bbox_loss_param.num_classes();
  share_location_ = bbox_loss_param.share_location();
  CHECK(share_location_) << "[deprecated] Only share_location is supportted in current version.";
  if (share_location_) {
    loc_classes_ = 1;
  } else {
    loc_classes_ = bbox_loss_param.loc_class();
    CHECK_EQ(loc_classes_, 1) << "[deprecated] loc_classes must be 1 in current version.";
  }
  // whether to use the images those have no person as background
  if (bbox_loss_param.has_flag_noperson()) {
    if (bbox_loss_param.flag_noperson()) {
      flag_noperson_ = true;
    } else {
      flag_noperson_ = false;
    }
  } else {
    flag_noperson_ = false;
  }
  flag_checkanchor_ = bbox_loss_param.flag_checkanchor();
  // use PER_PREDICTION
  match_type_ = bbox_loss_param.match_type();
  overlap_threshold_ = bbox_loss_param.overlap_threshold();
  use_prior_for_matching_ = bbox_loss_param.use_prior_for_matching();
  CHECK_EQ(use_prior_for_matching_, true) << "please use prior for matching.";
  use_difficult_gt_ = bbox_loss_param.use_difficult_gt();
  CHECK_EQ(use_difficult_gt_, false) << "please do not use difficult bboxes for matching.";
  do_neg_mining_ = bbox_loss_param.do_neg_mining();
  do_pos_mining_ = bbox_loss_param.do_pos_mining();
  pos_objectness_score_ = bbox_loss_param.pos_objectness_score();
  neg_pos_ratio_ = bbox_loss_param.neg_pos_ratio();
  CHECK_GE(neg_pos_ratio_, 1) << "neg_pos_ratio must ge greater than 1.";
  neg_overlap_ = bbox_loss_param.neg_overlap();
  alias_id_ = bbox_loss_param.alias_id();
  code_type_ = bbox_loss_param.code_type();
  encode_variance_in_target_ = bbox_loss_param.encode_variance_in_target();
  CHECK_EQ(encode_variance_in_target_, false) << "encode_variance_in_target should be false.";
  map_object_to_agnostic_ = bbox_loss_param.map_object_to_agnostic();
  CHECK_EQ(map_object_to_agnostic_, false) << "map_object_to_agnostic should be false.";
  size_threshold_ = bbox_loss_param.size_threshold();
  // gama & FocusLoss
  using_focus_loss_ = bbox_loss_param.using_focus_loss();
  gama_ = bbox_loss_param.gama();
  alpha_ = bbox_loss_param.alpha();
  // gt labels & target labels
  CHECK_EQ(bbox_loss_param.gt_labels_size(), bbox_loss_param.target_labels_size());
  for (int i = 0; i < bbox_loss_param.gt_labels_size(); ++i) {
    gt_labels_.push_back(bbox_loss_param.gt_labels(i));
    target_labels_[bbox_loss_param.gt_labels(i)] = bbox_loss_param.target_labels(i);
  }
  // check loss parameters
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  // create LocLoss layers
  vector<int> loss_shape(1, 1);
  loc_weight_ = bbox_loss_param.loc_weight();
  loc_loss_type_ = bbox_loss_param.loc_loss_type();
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
  // create ConfLoss layer
  conf_weight_ = bbox_loss_param.conf_weight();
  conf_loss_type_ = bbox_loss_param.conf_loss_type();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    if (!using_focus_loss_) {
      LayerParameter layer_param;
      layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
      layer_param.set_type("SoftmaxWithLoss");
      layer_param.add_loss_weight(conf_weight_);
      layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
      SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
      softmax_param->set_axis(1);
      vector<int> conf_shape(1, 1);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
      conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
      conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    } else {
      LayerParameter layer_param;
      layer_param.set_name(this->layer_param_.name() + "_softmax_fl_conf");
      layer_param.set_type("SoftmaxWithFocusLoss");
      layer_param.add_loss_weight(conf_weight_);
      layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
      SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
      softmax_param->set_axis(1);
      layer_param.mutable_focus_loss_param()->set_gama(gama_);
      layer_param.mutable_focus_loss_param()->set_alpha(alpha_);
      vector<int> conf_shape(1, 1);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
      conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
      conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    }
  } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    if (!using_focus_loss_) {
      LayerParameter layer_param;
      layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
      layer_param.set_type("SigmoidCrossEntropyLoss");
      layer_param.add_loss_weight(conf_weight_);
      vector<int> conf_shape(1, 1);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
      conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
      conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    } else {
      LayerParameter layer_param;
      layer_param.set_name(this->layer_param_.name() + "_logistic_fl_conf");
      layer_param.set_type("SigmoidCrossEntropyFocusLoss");
      layer_param.add_loss_weight(conf_weight_);
      layer_param.mutable_focus_loss_param()->set_gama(gama_);
      layer_param.mutable_focus_loss_param()->set_alpha(alpha_);
      vector<int> conf_shape(1, 1);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
      conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
      conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
    }
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
  ShowAnchorNumVsScale_ = bbox_loss_param.showanchornumvsscale();
  for (int i = 0; i < 8; i++) {
    gt_anchornum_scale_total_.push_back(0);
    gt_total_.push_back(0);
  }

  // set internal sigmoid layer
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[5]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  objectness_score_ = bbox_loss_param.objectness_score();
}

template <typename Dtype>
void BBoxLossODMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> top_shape(1, 1);
  top[0]->Reshape(top_shape);
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(num_priors_ * 4 * (num_classes_ - 1), bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";

  // set internal sigmoid layer
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}


template <typename Dtype>
void BBoxLossODMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const BBoxLossParameter& bbox_loss_param = this->layer_param_.bbox_loss_param();
  all_pos_indices_.clear();
  all_neg_indices_.clear();
  decode_prior_bboxes_.clear();
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();
  const Dtype* arm_loc_data = bottom[4]->cpu_data();
  const Dtype* arm_conf_data = bottom[5]->cpu_data();
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();

  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  int ndim_label = bbox_loss_param.ndim_label();
  vector<Dtype> img_xmins;
  vector<Dtype> img_ymins;
  vector<Dtype> img_xmaxs;
  vector<Dtype> img_ymaxs;
  img_xmins.resize(num_);
  img_ymins.resize(num_);
  img_xmaxs.resize(num_);
  img_ymaxs.resize(num_);
  vector<bool> flags_needcheck(num_, false);
  if (flag_checkanchor_) {
    for (int i = 0; i < num_gt_; i++) {
      int bind = gt_data[i * ndim_label];
      img_xmins[bind] = gt_data[i * ndim_label + 9];
      img_ymins[bind] = gt_data[i * ndim_label + 10];
      img_xmaxs[bind] = gt_data[i * ndim_label + 11];
      img_ymaxs[bind] = gt_data[i * ndim_label + 12];
      flags_needcheck[bind] = true;
    }
  }
  map<int, vector<LabeledBBox<Dtype> > > all_gt_bboxes;
  GetGTBBoxes(gt_data, num_gt_, use_difficult_gt_, gt_labels_, size_threshold_, &all_gt_bboxes, ndim_label);
  int cnt = 0;
  if (flag_noperson_) {
    vector<LabeledBBox<Dtype> >  gt_bbox;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
        all_gt_bboxes[i] = gt_bbox; // write an empty vector box
        cnt += 1;
      }
    }
  }
//  LOG(INFO)<<cnt<< " images for batchsizie "<<num_<<" In bbox_loss_layer";
  /**************************************************************************#
  获取所有的prior-boxes
  #***************************************************************************/
  vector<LabeledBBox<Dtype> > prior_bboxes;
  vector<vector<Dtype> > prior_variances;
  GetAnchorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);
  flags_of_anchor_.clear();
  for (int i = 0; i < prior_bboxes.size(); i++) {
    flags_of_anchor_.push_back(true);
  }

  /**************************************************************************#
  获取所有的Box坐标估计
  #***************************************************************************/
  // 获取LOC的估计信息
  vector<vector<LabeledBBox<Dtype> > > all_loc_preds;
  GetLocPreds(loc_data, num_, num_priors_, &all_loc_preds);

  /**************************************************************************#
  获取所有一阶段的Box坐标估计
  #***************************************************************************/
  // 获取一阶段的LOC的估计信息
  vector<vector<LabeledBBox<Dtype> > > all_arm_loc_preds;
  GetLocPreds(arm_loc_data, num_, num_priors_, &all_arm_loc_preds);

  /**************************************************************************#
  获取分类任务的最大置信度信息:
  #***************************************************************************/
  vector<vector<Dtype> > max_scores;
  const int loss_type = (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) ? 0 : 1;
  const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
  const int match_type = (match_type_ == MultiBoxLossParameter_MatchType_PER_PREDICTION) ? 0 : 1;
  GetHDMScores(conf_data, num_, num_priors_, num_classes_, loss_type, &max_scores);

  /**************************************************************************#
  获取一阶段置信度信息:
  #***************************************************************************/
  sigmoid_bottom_vec_[0] = bottom[5];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  vector<vector<Dtype> > neg_scores;
  GetARMScores(sigmoid_output_data, num_, num_priors_, num_classes_, loss_type, &neg_scores);

  /**************************************************************************#
  统计各个任务的样本数: batch计算前初始化为0
  #***************************************************************************/
  num_pos_ = 0;
  num_neg_ = 0;
  /**************************************************************************#
  完成所有任务的匹配统计: 正例统计/反例统计
  #***************************************************************************/
  vector<pair<Dtype, pair<int, int> > > scores_indices; // pair <score, pari<prior-id, batch-id> >
  for (int i = 0; i < num_; ++i) {
    if (flag_checkanchor_) {
      int image_width = bottom[4]->width();
      int image_height = bottom[4]->height();
      if (flags_needcheck[i]) {
        for (int ianchor = 0; ianchor < prior_bboxes.size(); ianchor++) {
          int x1 = prior_bboxes[ianchor].bbox.x1_ * image_width;
          int y1 = prior_bboxes[ianchor].bbox.y1_ * image_height;
          int x2 = prior_bboxes[ianchor].bbox.x2_ * image_width;
          int y2 = prior_bboxes[ianchor].bbox.y2_ * image_height;

          if (x1 >= img_xmins[i] && y1 >= img_ymins[i] && x2 <= img_xmaxs[i] && y2 <= img_ymaxs[i]) {
            flags_of_anchor_[ianchor] = true;
          } else {
            flags_of_anchor_[ianchor] = false;
          }
        }
      } else {
        for (int ianchor = 0; ianchor < prior_bboxes.size(); ianchor++) {
          flags_of_anchor_[ianchor] = true;
        }
      }

    }

    /**************************************************************************#
    单个样本的匹配列表
    #***************************************************************************/
    map<int, int > match_indices;
    vector<int> unmatch_indices;
    vector<int> neg_indices;
    vector<Dtype> match_overlaps;
    // 如果当前样本没有gtbox,直接返回
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      all_pos_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
      continue;
    }
    bool flag_mtanchorgt_allneg = bbox_loss_param.flag_mtanchorgt_allneg();
    bool flag_withotherpositive = true;//default true;
    // 获得该样本的gt-box列表
    const vector<LabeledBBox<Dtype> >& gt_bboxes = all_gt_bboxes.find(i)->second;
    // 使用实际的检测box进行匹配
    // if (!use_prior_for_matching_) {
    vector<LabeledBBox<Dtype> > loc_bboxes;
    //   if (code_type == 0) {
    DecodeBBoxes_Center(prior_bboxes, prior_variances, encode_variance_in_target_,
                        all_arm_loc_preds[i], &loc_bboxes);
    decode_prior_bboxes_.push_back(loc_bboxes);
    // } else {
    //   DecodeBBoxes_Corner(prior_bboxes,prior_variances,encode_variance_in_target_,
    //                       all_loc_preds[i],&loc_bboxes);
    // }
    MatchAnchorsAndGTs(gt_bboxes, loc_bboxes, match_type, overlap_threshold_, neg_overlap_,
                       &match_overlaps, &match_indices, &unmatch_indices, flags_of_anchor_, flag_noperson_, flag_withotherpositive, flag_mtanchorgt_allneg);
    // } else {
    //   MatchAnchorsAndGTs(gt_bboxes, prior_bboxes, match_type, overlap_threshold_, neg_overlap_,
    //                     &match_overlaps, &match_indices, &unmatch_indices,flags_of_anchor_,flag_noperson_,flag_withotherpositive,flag_mtanchorgt_allneg);
    // }

    if (ShowAnchorNumVsScale_)
    {
      std::vector<int> gt_real_vec;
      for (map<int, int>::iterator it = match_indices.begin(); it != match_indices.end(); ++it) {
        const int gt_id = it->second;
        bool found = false;
        for (std::vector<int>::iterator i = gt_real_vec.begin(); i != gt_real_vec.end(); ++i) {
          if ( gt_id == *i) {
            found = true;
            break;
          }
        }
        if (!found) {
          gt_real_vec.push_back(gt_id);
        }
      }
      for (std::vector<int>::iterator it = gt_real_vec.begin(); it != gt_real_vec.end(); ++it) {
        float gt_bboxes_scale = gt_bboxes[*it].bbox.compute_area();
        int level = -1;
        if (gt_bboxes_scale > 0)
          level ++;
        if (gt_bboxes_scale > 0.001)
          level ++;
        if (gt_bboxes_scale > 0.01)
          level ++;
        if (gt_bboxes_scale > 0.1)
          level ++;
        if (gt_bboxes_scale > 0.15)
          level ++;
        if (gt_bboxes_scale > 0.3)
          level ++;
        if (gt_bboxes_scale > 0.6)
          level ++;
        if (gt_bboxes_scale > 0.85)
          level ++;
        gt_total_[level] ++;
      }
      for (map<int, int>::iterator it = match_indices.begin(); it != match_indices.end(); ++it) {
        const int prior_id = it->first;
        const int gt_id = it->second;
        float gt_bboxes_scale = gt_bboxes[gt_id].bbox.compute_area();
        int level = -1;
        if (gt_bboxes_scale > 0)
          level ++;
        if (gt_bboxes_scale > 0.001)
          level ++;
        if (gt_bboxes_scale > 0.01)
          level ++;
        if (gt_bboxes_scale > 0.1)
          level ++;
        if (gt_bboxes_scale > 0.15)
          level ++;
        if (gt_bboxes_scale > 0.3)
          level ++;
        if (gt_bboxes_scale > 0.6)
          level ++;
        if (gt_bboxes_scale > 0.85)
          level ++;
        gt_anchornum_scale_total_[level] ++;
      }
    }

    // void MatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
    //                     const vector<LabeledBBox<Dtype> > &pred_bboxes,
    //                     const int match_type,
    //                     const Dtype overlap_threshold,
    //                     const Dtype negative_threshold,
    //                     vector<Dtype> *match_overlaps,
    //                     map<int, int> *match_indices,
    //                     vector<int> *neg_indices,
    //                     vector<bool> flags_of_anchor,
    //                     bool flag_noperson,
    //                     bool flag_withotherpositive,
    //                     bool flag_matchallneg)
    /**************************************************************************#
    滤除标记为iscrowd [bbox]的匹配对象, 可以跳过　【GetGTBBoxes已经滤出了iscrowd对象】
    RemoveCrowds(gt_bboxes,match_indices, &new_indices);
    #***************************************************************************/
    int num_pos = match_indices.size();
    num_pos_ += num_pos;
    /**************************************************************************#
    统计反例列表
    #***************************************************************************/
    if (flag_noperson_) {
      for (int j = 0; j < unmatch_indices.size(); ++j) {
        if (neg_scores[i][unmatch_indices[j]] <= objectness_score_) {
          scores_indices.push_back(std::make_pair(max_scores[i][unmatch_indices[j]], std::make_pair(unmatch_indices[j], i)));
        }
      }
      if(do_pos_mining_){
        int erase_pos = 0;
        for (map<int, int>::iterator it = match_indices.begin(); it != match_indices.end(); ++it){
          //LOG(INFO)<<neg_scores[i][it->first];
         if(neg_scores[i][it->first] > pos_objectness_score_){
            match_indices.erase(it);
            erase_pos++;
          }
         }
         num_pos_ -= erase_pos;
       }
      all_pos_indices_.push_back(match_indices);
    } else {
      if (do_neg_mining_) {
        scores_indices.clear();
        int num_neg = unmatch_indices.size();
        for (int j = 0; j < unmatch_indices.size(); ++j) {
          if (neg_scores[i][unmatch_indices[j]] <= objectness_score_) {
            scores_indices.push_back(std::make_pair(max_scores[i][unmatch_indices[j]], std::make_pair(unmatch_indices[j], i)));
          }
        }
        num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
        std::sort(scores_indices.begin(), scores_indices.end(), PairDescend<pair<int, int>, Dtype>);
        for (int n = 0; n < num_neg; ++n) {
          neg_indices.push_back(scores_indices[n].second.first);
        }
        num_neg_ += num_neg;
      } else {
        for (int n = 0; n < unmatch_indices.size(); ++n) {
          neg_indices.push_back(unmatch_indices[n]);
        }
        num_neg_ += unmatch_indices.size();
      }
      
      //正样本筛选:去除正样本的背景预测大于阈值的正样本,即去除超级困难正样本
      if(do_pos_mining_){
        for (map<int, int>::iterator it = match_indices.begin(); it != match_indices.end(); ++it){
          //LOG(INFO)<<neg_scores[i][it->first];
          if(neg_scores[i][it->first] > pos_objectness_score_){
            match_indices.erase(it);
          }
        }
      }

      /**************************************************************************#
       将正例和反例列表保存到batch的索引记录之中
       #***************************************************************************/
      all_pos_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
    }
  }
//LOG(INFO)<<"All "<<num_neg_<<" negative instances;"<<num_pos_<<" positive instances bbox_loss_layer.";
  bool flag_show = false;
  if (flag_show) {
    vector<Dtype> priors_steps_featmap;
    priors_steps_featmap.push_back(46080);//pool1: 512*288/4/4 = 9216; 9216*5=46080;
    priors_steps_featmap.push_back(9216);// featuremap 512*288/8/8 = 2304;2304*4=9216
    priors_steps_featmap.push_back(5184);// featuremap 512*288/16/16 = 576;576*9=5184
    priors_steps_featmap.push_back(1584);// featuremap 512*288/32/32 = 144;144*11=1584
    for (int i = 1; i < priors_steps_featmap.size(); i++) {
      priors_steps_featmap[i] += priors_steps_featmap[i - 1];
    }
    CHECK_EQ(priors_steps_featmap[3], num_priors_) << priors_steps_featmap[3] << " " << num_priors_;
    vector<Dtype> bbox_sizes_featmap;
    bbox_sizes_featmap.push_back(0.03);
    bbox_sizes_featmap.push_back(0.1);
    bbox_sizes_featmap.push_back(0.35);
    bbox_sizes_featmap.push_back(0.95);
    int match_hand = 0;
    vector<Dtype>  count_match_per_featmap_hand;//cid=1
    vector<Dtype>  count_match_per_featmap_head;//cid=2
    vector<Dtype>  count_match_per_featmap_face;//cid=3
    vector<Dtype> num_gt_percls(3, 0);
    vector<Dtype> num_match_percls(3, 0);
    vector<Dtype> num_gt_perscale_hand(bbox_sizes_featmap.size(), 0);
    vector<Dtype> num_gt_perscale_head(bbox_sizes_featmap.size(), 0);
    vector<Dtype> num_gt_perscale_face(bbox_sizes_featmap.size(), 0);
    for (typename std::map<int, std::vector<LabeledBBox<Dtype> > > ::const_iterator it = all_gt_bboxes.begin(); it != all_gt_bboxes.end(); ++it) {
      // num_gt_box += it->second.size();
      const vector<LabeledBBox<Dtype> >& gt_bboxes = it->second;
      for (int i = 0; i < gt_bboxes.size(); i++) {
        LabeledBBox<Dtype> gt = gt_bboxes[i];
        if (gt.cid == 1 || gt.cid == 2 || gt.cid == 3) {
          num_gt_percls[gt.cid - 1] += 1;
        }
        float area = (gt.bbox.x2_ - gt.bbox.x1_) * (gt.bbox.y2_ - gt.bbox.y1_);
        float bbox_size = std::sqrt(area);
        int level = 0;
        for (int k = 0; k < bbox_sizes_featmap.size(); k++) {
          if (bbox_size < bbox_sizes_featmap[k]) {
            continue;
          } else {
            level += 1;
          }
        }
        if (gt.cid == 1) {
          num_gt_perscale_hand[level] += 1;
        }
        if (gt.cid == 2) {
          num_gt_perscale_head[level] += 1;
        }
        if (gt.cid == 3) {
          num_gt_perscale_face[level] += 1;
        }
      }
    }

    for (int i = 0; i < priors_steps_featmap.size(); i++) {
      count_match_per_featmap_hand.push_back(0);
      count_match_per_featmap_head.push_back(0);
      count_match_per_featmap_face.push_back(0);
    }
    for (int i = 0; i < num_; i++) {
      std::map<int, int> match_indices = all_pos_indices_[i];
      if (match_indices.size() > 0) {
        if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
          continue;
        }
        const std::vector<LabeledBBox<Dtype> >& gt_bboxes = all_gt_bboxes.find(i)->second;
        for (map<int, int>::iterator it = match_indices.begin();
             it != match_indices.end(); ++it) {
          int idx_prior = it->first;
          int level = 0;
          for (int k = 0; k < priors_steps_featmap.size(); k++) {
            if (idx_prior < priors_steps_featmap[k]) {
              continue;
            } else {
              level += 1;
            }
          }
          if (gt_bboxes[it->second].cid == 1) {
            count_match_per_featmap_hand[level] += 1;
            num_match_percls[0] += 1;
          }
          if (gt_bboxes[it->second].cid == 2) {
            count_match_per_featmap_head[level] += 1;
            num_match_percls[1] += 1;
          }

          if (gt_bboxes[it->second].cid == 3) {
            count_match_per_featmap_face[level] += 1;
            num_match_percls[2] += 1;
          }
        }

      }
    }
    LOG(INFO) << "Hand Match " << count_match_per_featmap_hand[0]
              << " " << count_match_per_featmap_hand[1]
              << " " << count_match_per_featmap_hand[2]
              << " " << count_match_per_featmap_hand[3];
    LOG(INFO) << "Head Match " << count_match_per_featmap_head[0]
              << " " << count_match_per_featmap_head[1]
              << " " << count_match_per_featmap_head[2]
              << " " << count_match_per_featmap_head[3];
    LOG(INFO) << "Face Match " << count_match_per_featmap_face[0]
              << " " << count_match_per_featmap_face[1]
              << " " << count_match_per_featmap_face[2]
              << " " << count_match_per_featmap_face[3];
    LOG(INFO) << "NumGT_Hand: " << num_gt_percls[0] << "; NumMatch_Hand: " << num_match_percls[0] << "."
              << "NumGT_Head: " << num_gt_percls[1] << "; NumMatch_Head: " << num_match_percls[1] << "."
              << "NumGT_Face: " << num_gt_percls[2] << "; NumMatch_Face: " << num_match_percls[2] << ".";
    LOG(INFO) << "NumGT_PerScale_Hand " << num_gt_perscale_hand[0]
              << " " << num_gt_perscale_hand[1]
              << " " << num_gt_perscale_hand[2]
              << " " << num_gt_perscale_hand[3];
    LOG(INFO) << "NumGT_PerScale_Head " << num_gt_perscale_head[0]
              << " " << num_gt_perscale_head[1]
              << " " << num_gt_perscale_head[2]
              << " " << num_gt_perscale_head[3];
    LOG(INFO) << "NumGT_PerScale_Face " << num_gt_perscale_face[0]
              << " " << num_gt_perscale_face[1]
              << " " << num_gt_perscale_face[2]
              << " " << num_gt_perscale_face[3];
  }

  if (ShowAnchorNumVsScale_)
  {
    LOG(INFO) << "total: <0.001: " << gt_anchornum_scale_total_[0] << " <0.01: " << gt_anchornum_scale_total_[1]
              << " <0.1: " << gt_anchornum_scale_total_[2] << " <0.15: " << gt_anchornum_scale_total_[3]
              << " <0.3: " << gt_anchornum_scale_total_[4] << " <0.6: " << gt_anchornum_scale_total_[5]
              << " <0.85: " << gt_anchornum_scale_total_[6] << " <1.0: " << gt_anchornum_scale_total_[7];
    LOG(INFO) << "ave: <0.001: " << (gt_total_[0] > 0 ? gt_anchornum_scale_total_[0] / (Dtype)gt_total_[0] : 0)
              << " <0.01: " << (gt_total_[1] > 0 ? gt_anchornum_scale_total_[1] / (Dtype)gt_total_[1] : 0)
              << " <0.1: " << (gt_total_[2] > 0 ? gt_anchornum_scale_total_[2] / (Dtype)gt_total_[2] : 0)
              << " <0.15: " << (gt_total_[3] > 0 ? gt_anchornum_scale_total_[3] / (Dtype)gt_total_[3] : 0)
              << " <0.3: " << (gt_total_[4] > 0 ? gt_anchornum_scale_total_[4] / (Dtype)gt_total_[4] : 0)
              << " <0.6: " << (gt_total_[5] > 0 ? gt_anchornum_scale_total_[5] / (Dtype)gt_total_[5] : 0)
              << " <0.85: " << (gt_total_[6] > 0 ? gt_anchornum_scale_total_[6] / (Dtype)gt_total_[6] : 0)
              << " <1.0: " << (gt_total_[7] > 0 ? gt_anchornum_scale_total_[7] / (Dtype)gt_total_[7] : 0);
  }

  /**************************************************************************#
    find negative instances from mini-batch (not from each image)
    #***************************************************************************/
  if (flag_noperson_) {
    int num_neg_all = scores_indices.size();
    // LOG(INFO)<<"All "<<num_neg_all<<" negative instances;"<<num_pos_<<" positive instances bbox_loss_layer.";
    int num_neg = num_neg_all;
    if (!using_focus_loss_) {
      num_neg = std::min(static_cast<int>(num_pos_ * neg_pos_ratio_), num_neg_all);
    }
    std::sort(scores_indices.begin(), scores_indices.end(), PairDescend<pair<int, int>, Dtype>);
    for (int i = 0; i < num_; ++i) {
      vector<int> neg_indices;
      for (int n = 0; n < num_neg; ++n) {
        if (scores_indices[n].second.second == i) {
          neg_indices.push_back(scores_indices[n].second.first);
        }
      }
      all_neg_indices_.push_back(neg_indices);
      num_neg_ += neg_indices.size();
    }
  }
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
      for (map<int, int>::iterator it = all_pos_indices_[i].begin();
           it != all_pos_indices_[i].end(); ++it) {
        const int prior_id = it->first;
        const int gt_id = it->second;
        CHECK_LT(prior_id, num_priors_);
        CHECK_LT(gt_id, all_gt_bboxes[i].size());
        const vector<LabeledBBox<Dtype> >& loc_pred = all_loc_preds[i];
        CHECK_LT(prior_id, loc_pred.size());
        // 复制数据
        loc_pred_data[count * 4]     = loc_pred[prior_id].bbox.x1_;
        loc_pred_data[count * 4 + 1] = loc_pred[prior_id].bbox.y1_;
        loc_pred_data[count * 4 + 2] = loc_pred[prior_id].bbox.x2_;
        loc_pred_data[count * 4 + 3] = loc_pred[prior_id].bbox.y2_;
        const LabeledBBox<Dtype>& gt_bbox = all_gt_bboxes[i][gt_id];
        LabeledBBox<Dtype> gt_encode = LabeledBBox_Copy(gt_bbox);
        if (code_type == 0) {
          EncodeBBox_Center(decode_prior_bboxes_[i][prior_id].bbox, prior_variances[prior_id],
                            encode_variance_in_target_, gt_bbox.bbox,
                            &gt_encode.bbox);
        } else {
          EncodeBBox_Corner(decode_prior_bboxes_[i][prior_id].bbox, prior_variances[prior_id],
                            encode_variance_in_target_, gt_bbox.bbox,
                            &gt_encode.bbox);
        }
        // 复制label
        loc_gt_data[count * 4]     = gt_encode.bbox.x1_;
        loc_gt_data[count * 4 + 1] = gt_encode.bbox.y1_;
        loc_gt_data[count * 4 + 2] = gt_encode.bbox.x2_;
        loc_gt_data[count * 4 + 3] = gt_encode.bbox.y2_;
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
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  } else {
    loc_loss_.mutable_cpu_data()[0] = 0;
  }
  /**************************************************************************#
  conf损失计算
  #***************************************************************************/
  if (num_conf_ >= 1) {
    vector<int> conf_shape;
    if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      conf_shape.push_back(num_conf_);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
    } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      conf_shape.push_back(1);
      conf_shape.push_back(num_conf_);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    conf_data = bottom[1]->cpu_data();
    caffe_set(conf_gt_.count(), Dtype(0), conf_gt_data);
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      const map<int, int>& match_indices = all_pos_indices_[i];
      for (map<int, int>::const_iterator it =
             match_indices.begin(); it != match_indices.end(); ++it) {
        const int prior_id = it->first;
        CHECK_LT(prior_id, num_priors_);
        const int gt_id = it->second;
        CHECK_LT(gt_id, all_gt_bboxes[i].size());
        // get target_label
        int gt_label;
        if (target_labels_.size() == 0) {
          gt_label = all_gt_bboxes[i][gt_id].cid + 1 - alias_id_;
        } else {
          gt_label = target_labels_[all_gt_bboxes[i][gt_id].cid];
        }
        CHECK_GT(gt_label, 0) << "Found negative gt-labels.";
        CHECK_LT(gt_label, num_classes_);
        switch (conf_loss_type_) {
        case MultiBoxLossParameter_ConfLossType_SOFTMAX:
          conf_gt_data[count] = gt_label;
          break;
        case MultiBoxLossParameter_ConfLossType_LOGISTIC:
          conf_gt_data[count * num_classes_ + gt_label] = 1;
          break;
        default:
          LOG(FATAL) << "Unknown conf loss type.";
        }
        caffe_copy<Dtype>(num_classes_, conf_data + prior_id * num_classes_,
                          conf_pred_data + count * num_classes_);
        ++count;
      }
      // 反例
      const vector<int>& neg_indices = all_neg_indices_[i];
      for (int j = 0; j < neg_indices.size(); ++j) {
        const int prior_id = neg_indices[j];
        CHECK_LT(prior_id, num_priors_);
        // 复制data数据
        caffe_copy<Dtype>(num_classes_, conf_data + prior_id * num_classes_,
                          conf_pred_data + count * num_classes_);
        // 复制gt数据
        switch (conf_loss_type_) {
        case MultiBoxLossParameter_ConfLossType_SOFTMAX:
          conf_gt_data[count] = 0;
          break;
        case MultiBoxLossParameter_ConfLossType_LOGISTIC:
          conf_gt_data[count * num_classes_] = 1;
          break;
        default:
          LOG(FATAL) << "Unknown conf loss type.";
        }
        ++count;
      }
      // 指向下一个样本
      conf_data += bottom[1]->offset(1);
    }
    CHECK_EQ(count, num_conf_) << "Unmatch numbers of conf-Layer.";
    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
  } else {
    conf_loss_.mutable_cpu_data()[0] = 0;
  }

  /**************************************************************************#
  损失累加：由于损失层的NORMALIZE设置为NONE（1），因此需要再此做归一化
  后面的反向传播计算中的SCALE操作也是因为此.
  #***************************************************************************/
  top[0]->mutable_cpu_data()[0] = 0;
  // LOC损失
  if (this->layer_param_.propagate_down(0)) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
                         normalization_, num_, num_priors_, num_pos_);
    top[0]->mutable_cpu_data()[0] +=
      loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
  }
  // conf损失
  if (this->layer_param_.propagate_down(1)) {
    // conf
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
                         normalization_, num_, num_priors_, num_conf_);
    top[0]->mutable_cpu_data()[0] +=
      conf_weight_ * conf_loss_.cpu_data()[0] / normalizer;
  }
}

template <typename Dtype>
void BBoxLossODMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  /**************************************************************************#
  LOC反向传播
  #***************************************************************************/
  if (propagate_down[0]) {
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
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
        if (all_pos_indices_[i].size() > 0) {
          const map<int, int>& match_indices = all_pos_indices_[i];
          for (map<int, int>::const_iterator it = match_indices.begin();
               it != match_indices.end(); ++it) {
            const int prior_id = it->first;
            CHECK_LT(prior_id, num_priors_);
            caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
                              loc_bottom_diff + prior_id * 4);
            ++count;
          }
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
    // 获取误差指针
    Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
    /**************************************************************************#
    conf反向传播
    #***************************************************************************/
    if (num_conf_ >= 1) {
      vector<bool> conf_propagate_down;
      conf_propagate_down.push_back(true);
      conf_propagate_down.push_back(false);
      conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                                 conf_bottom_vec_);
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
                           normalization_, num_, num_priors_, num_conf_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      loss_weight *= conf_weight_;
      caffe_scal(conf_pred_.count(), loss_weight,
                 conf_pred_.mutable_cpu_diff());
      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        const map<int, int>& match_indices = all_pos_indices_[i];
        for (map<int, int>::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
          const int prior_id = it->first;
          CHECK_LT(prior_id, num_priors_);
          caffe_copy<Dtype>(num_classes_,
                            conf_pred_diff + count * num_classes_,
                            conf_bottom_diff + prior_id * num_classes_);
          ++count;
        }
        const vector<int>& neg_indices = all_neg_indices_[i];
        for (int n = 0; n < neg_indices.size(); ++n) {
          const int prior_id = neg_indices[n];
          CHECK_LT(prior_id, num_priors_);
          caffe_copy<Dtype>(num_classes_,
                            conf_pred_diff + count * num_classes_,
                            conf_bottom_diff + prior_id * num_classes_);
          ++count;
        }
        conf_bottom_diff += bottom[1]->offset(1);
      }
      CHECK_EQ(count, num_conf_);
    }
  }
  /**************************************************************************#
  清除所有匹配列表
  #***************************************************************************/
  all_pos_indices_.clear();
  all_neg_indices_.clear();
}

INSTANTIATE_CLASS(BBoxLossODMLayer);
REGISTER_LAYER_CLASS(BBoxLossODM);

}  // namespace caffe
