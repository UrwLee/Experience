#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/mask/dense_bbox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DenseBBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const DenseBBoxLossParameter& dense_bbox_loss_param =
      this->layer_param_.dense_bbox_loss_param();
  // whether to use the images those have no person as background
  flag_noperson_ = dense_bbox_loss_param.flag_noperson();
  flag_checkanchor_ = dense_bbox_loss_param.flag_checkanchor();
  // check parameters
  CHECK(dense_bbox_loss_param.has_num_classes());
  // NOTE: num_classes_ shoule be N+1
  num_classes_ = dense_bbox_loss_param.num_classes();
  overlap_threshold_ = dense_bbox_loss_param.overlap_threshold();
  use_prior_for_matching_ = dense_bbox_loss_param.use_prior_for_matching();
  CHECK_EQ(use_prior_for_matching_, true) << "please use prior for matching.";
  use_difficult_gt_ = dense_bbox_loss_param.use_difficult_gt();
  CHECK_EQ(use_difficult_gt_, false) << "please do not use difficult bboxes for matching.";
  do_neg_mining_ = dense_bbox_loss_param.do_neg_mining();
  neg_pos_ratio_ = dense_bbox_loss_param.neg_pos_ratio();
  CHECK_GE(neg_pos_ratio_, 1) << "neg_pos_ratio must be greater than 1.";
  neg_overlap_ = dense_bbox_loss_param.neg_overlap();
  alias_id_ = dense_bbox_loss_param.alias_id();
  code_type_ = dense_bbox_loss_param.code_type();
  encode_variance_in_target_ = dense_bbox_loss_param.encode_variance_in_target();
  CHECK_EQ(encode_variance_in_target_, false) << "encode_variance_in_target should be false.";

  flag_showdebug_ = dense_bbox_loss_param.flag_showdebug();
  flag_forcematchallgt_ = dense_bbox_loss_param.flag_forcematchallgt();
  flag_areamaxcheckinmatch_ = dense_bbox_loss_param.flag_areamaxcheckinmatch();
  size_threshold_.push_back(dense_bbox_loss_param.size_threshold());
  size_threshold_.push_back(dense_bbox_loss_param.size_threshold_max());
  // gt labels & target labels
  CHECK_EQ(dense_bbox_loss_param.gt_labels_size(), dense_bbox_loss_param.target_labels_size());
  for (int i = 0; i < dense_bbox_loss_param.gt_labels_size(); ++i) {
    gt_labels_.push_back(dense_bbox_loss_param.gt_labels(i));
    target_labels_[dense_bbox_loss_param.gt_labels(i)] = dense_bbox_loss_param.target_labels(i);
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
  loc_weight_ = dense_bbox_loss_param.loc_weight();
  loc_loss_type_ = dense_bbox_loss_param.loc_loss_type();
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
  using_focus_loss_ = dense_bbox_loss_param.using_focus_loss();
  gama_ = dense_bbox_loss_param.gama();
  alpha_ = dense_bbox_loss_param.alpha();
  conf_weight_ = dense_bbox_loss_param.conf_weight();
  conf_loss_type_ = dense_bbox_loss_param.conf_loss_type();
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
  other_value_ = 50;
  other_value_2_ = -50;
}

template <typename Dtype>
void DenseBBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> top_shape(1,1);
  top[0]->Reshape(top_shape);
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // NOTE: Np*4*(num_classes_-1) -> LOC
  // NOTE: Np*(num_classes_) -> CONF
  CHECK_EQ(num_priors_ * 4 * (num_classes_ - 1), bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
}

template <typename Dtype>
void DenseBBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   const DenseBBoxLossParameter& dense_bbox_loss_param = this->layer_param_.dense_bbox_loss_param();
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* gt_data = bottom[3]->cpu_data();
  num_ = bottom[0]->num();
  num_priors_ = bottom[2]->height() / 4;
  num_gt_ = bottom[3]->height();
 
  

  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  // map<int, vector<LabeledBBox<Dtype> > > all_gt_bboxes;
  // size_threshold_[0]: min area;size_threshold_[1]: max area;
  int ndim_label = dense_bbox_loss_param.ndim_label();
  if(flag_areamaxcheckinmatch_){ // not check area in getting gt
    GetGTBBoxes(gt_data, num_gt_, use_difficult_gt_, gt_labels_, size_threshold_[0], &all_gt_bboxes_,ndim_label);
  } else{
    GetGTBBoxes(gt_data, num_gt_, use_difficult_gt_, gt_labels_, size_threshold_, &all_gt_bboxes_,ndim_label);
  }
  
  // LOG(INFO)<<"all_gt_bboxes_ "<<all_gt_bboxes_.size();
  // for(int i=0;i<all_gt_bboxes_.size();i++){
  //   LOG(INFO)<<
  // }
  int cnt = 0; // Nouse; Only for Debugging.
  if (flag_noperson_){
    vector<LabeledBBox<Dtype> >  gt_bbox;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes_.find(i) == all_gt_bboxes_.end()) {
          all_gt_bboxes_[i] = gt_bbox; // write an empty vector box
          cnt += 1;
      }
    }
  }
  // LOG(INFO)<<cnt<< " images for batchsizie "<<num_<<" In dense_bbox_loss_layer";
  /**************************************************************************#
  获取所有的prior-boxes
  #***************************************************************************/
  vector<LabeledBBox<Dtype> > prior_bboxes;
  vector<vector<Dtype> > prior_variances;
  GetAnchorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);
  flags_of_anchor_.clear();
  for(int i=0;i<prior_bboxes.size();i++){
    flags_of_anchor_.push_back(true);
  }
 
  /**************************************************************************#
  获取所有的Box坐标估计
  #***************************************************************************/
  // 获取LOC的估计信息
  vector<vector<LabeledBBox<Dtype> > > all_loc_preds;
  GetLocPreds(loc_data, num_, num_priors_ * (num_classes_ - 1), &all_loc_preds);
  /**************************************************************************#
  获取分类任务的最大置信度信息:
  #***************************************************************************/
  vector<vector<Dtype> > max_scores;
  conf_data = bottom[1]->cpu_data();
  const int loss_type = (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) ? 0 : 1;
  const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
  GetHDMScores(conf_data, num_, num_priors_, num_classes_, loss_type, &max_scores);
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
    /**************************************************************************#
    单个样本的匹配列表
    #***************************************************************************/
    vector<pair<int, int > > match_indices; //[id_priorbox,id_gt]
    vector<int> unmatch_indices;
    vector<int> neg_indices;
    // 如果当前样本没有gtbox,直接返回
    // NOTE:如果using_focus_loss_，是否需要使用背景样本？
    if (all_gt_bboxes_.find(i) == all_gt_bboxes_.end()) {
      all_pos_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
      continue;
    }
    // 获得该样本的gt-box列表
    const vector<LabeledBBox<Dtype> >& gt_bboxes = all_gt_bboxes_.find(i)->second;
    // 匹配
    if(flag_areamaxcheckinmatch_){
      ExhaustMatchAnchorsAndGTs(gt_bboxes, prior_bboxes, overlap_threshold_, neg_overlap_,
                      &match_indices, &unmatch_indices, flags_of_anchor_,flag_noperson_,flag_forcematchallgt_,size_threshold_[1]);
    } else{
      ExhaustMatchAnchorsAndGTs(gt_bboxes, prior_bboxes, overlap_threshold_, neg_overlap_,
                      &match_indices, &unmatch_indices, flags_of_anchor_,flag_noperson_,flag_forcematchallgt_);
    }
    
   //################################################################Draw Matched Priorbox Start
    // if(false){
    //   static int counter = 0;
    //   int width =  bottom[4]->width();
    //   int height = bottom[4]->height();
      
    //   const int src_offset2 = height * width;
    //   const int src_offset3 = src_offset2 * 3;
      
    //   for(int i_mat=0;i_mat<match_indices.size();i_mat++){
    //     const Dtype* image_data = bottom[4]->cpu_data();
    //     LabeledBBox<Dtype> gt = gt_bboxes[match_indices[i_mat].second];
    //     int bindex = gt.bindex;
    //     cv::Mat img(height, width, CV_8UC3);
    //     for (int y = 0; y < height; y++){
    //     for (int x = 0; x < width; x++){
    //       cv::Vec3b & color = img.at<cv::Vec3b>(y,x);
    //         color[0] = (image_data[bindex*src_offset3 + 0*src_offset2 + y*width + x] + 104);
    //         color[1] = (image_data[bindex*src_offset3 + 1*src_offset2 + y*width + x] + 117);
    //         color[2] = (image_data[bindex*src_offset3 + 2*src_offset2 + y*width + x] + 123);
    //       }
    //     }
    //     LabeledBBox<Dtype> bbox = prior_bboxes[match_indices[i_mat].first];
    //     int xmin = int(bbox.bbox.x1_*width);
    //     int ymin = int(bbox.bbox.y1_*height);
    //     int xmax = int(bbox.bbox.x2_*width);
    //     int ymax = int(bbox.bbox.y2_*height);
    //     cv::rectangle(img,cvPoint(xmin,ymin),cvPoint(xmax,ymax),cv::Scalar(255,0,0),1,1,0);
    //     xmin = int(gt.bbox.x1_*width);
    //     ymin = int(gt.bbox.y1_*height);
    //     xmax = int(gt.bbox.x2_*width);
    //     ymax = int(gt.bbox.y2_*height);
    //     cv::rectangle(img,cvPoint(xmin,ymin),cvPoint(xmax,ymax),cv::Scalar(0,0,255),1,1,0);
    //     char imagename [256];
    //     sprintf(imagename, "/home/zhangming/vis_match/augment_%06d.jpg", counter);
    //     cv::imwrite(imagename, img);
    //     counter++;
    //   }
    // }

    //################################################################Draw Matched Priorbox End
    /****************************************************************************/
    int num_pos = match_indices.size();
    num_pos_ += num_pos;
    /**************************************************************************#
    统计反例列表
    #***************************************************************************/
    if (flag_noperson_){
      for (int j = 0; j < unmatch_indices.size(); ++j) {
        scores_indices.push_back(std::make_pair(max_scores[i][unmatch_indices[j]], std::make_pair(unmatch_indices[j], i)));
      }
      all_pos_indices_.push_back(match_indices);
    } else{
      if (do_neg_mining_) {
        scores_indices.clear();
        int num_neg = unmatch_indices.size();
        for (int j = 0; j < unmatch_indices.size(); ++j) {
          scores_indices.push_back(std::make_pair(max_scores[i][unmatch_indices[j]], std::make_pair(unmatch_indices[j], i)));
        }
        num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
        std::sort(scores_indices.begin(), scores_indices.end(), PairDescend<pair<int, int>,Dtype>); // find negative from each image
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
      /**************************************************************************#
      将正例和反例列表保存到batch的索引记录之中
      #***************************************************************************/
      all_pos_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
  }
}


if (flag_showdebug_){
  vector<Dtype> priors_steps_featmap;
  priors_steps_featmap.push_back(18432);//pool1: 512*288/4/4 = 9216; 9216*5=46080;
  // priors_steps_featmap.push_back(4608);// featuremap 512*288/8/8 = 2304;2304*4=9216
  // priors_steps_featmap.push_back(5184);// featuremap 512*288/16/16 = 576;576*9=5184
  // priors_steps_featmap.push_back(1584);// featuremap 512*288/32/32 = 144;144*11=1584
  for(int i = 1;i<priors_steps_featmap.size();i++){
    priors_steps_featmap[i] += priors_steps_featmap[i-1];
  }
  CHECK_EQ(priors_steps_featmap[priors_steps_featmap.size()-1],num_priors_)<<priors_steps_featmap[priors_steps_featmap.size()-1]<<" "<<num_priors_;
  vector<Dtype> bbox_sizes_featmap;
  bbox_sizes_featmap.push_back(0.03);
  bbox_sizes_featmap.push_back(0.1);
  bbox_sizes_featmap.push_back(0.35);
  bbox_sizes_featmap.push_back(0.95);
  int match_hand = 0;
  vector<Dtype>  count_match_per_featmap_hand;//cid=1
  vector<Dtype>  count_match_per_featmap_head;//cid=2
  vector<Dtype>  count_match_per_featmap_face;//cid=3
  vector<Dtype> num_gt_percls(3,0);
  vector<Dtype> num_match_percls(3,0);
  vector<Dtype> num_gt_perscale_hand(bbox_sizes_featmap.size(),0);
  vector<Dtype> num_gt_perscale_head(bbox_sizes_featmap.size(),0);
  vector<Dtype> num_gt_perscale_face(bbox_sizes_featmap.size(),0);
  for (typename std::map<int, std::vector<LabeledBBox<Dtype> > > ::const_iterator it = all_gt_bboxes_.begin(); it != all_gt_bboxes_.end(); ++it){
    // num_gt_box += it->second.size();
    const vector<LabeledBBox<Dtype> >& gt_bboxes = it->second;
    for (int i=0;i<gt_bboxes.size();i++){
      LabeledBBox<Dtype> gt = gt_bboxes[i];
      if(gt.cid == 1 || gt.cid == 2 || gt.cid == 3){
        num_gt_percls[gt.cid-1] +=1;
      }
      float area = (gt.bbox.x2_-gt.bbox.x1_)*(gt.bbox.y2_-gt.bbox.y1_);
      float bbox_size = std::sqrt(area);
      int level = 0;
      for(int k=0;k<bbox_sizes_featmap.size();k++){
        if(bbox_size<bbox_sizes_featmap[k]){
          continue;
        }else{
          level += 1;
        }
      }
      if(gt.cid == 1){
        num_gt_perscale_hand[level] += 1;
      }
      if(gt.cid == 2){
        num_gt_perscale_head[level] += 1;
      }
      if(gt.cid == 3){
        num_gt_perscale_face[level] += 1;
      }  
    }
  }

  for(int i=0;i<priors_steps_featmap.size();i++){
    count_match_per_featmap_hand.push_back(0);
    count_match_per_featmap_head.push_back(0);
    count_match_per_featmap_face.push_back(0);
  }
  for(int i=0;i<num_;i++){
    std::vector<pair<int, int > > match_indices = all_pos_indices_[i];
    if(match_indices.size()>0){
      if (all_gt_bboxes_.find(i) == all_gt_bboxes_.end()) {
        continue;
      }
      const std::vector<LabeledBBox<Dtype> >& gt_bboxes = all_gt_bboxes_.find(i)->second;
      for (int j=0;j<match_indices.size();j++){
        int idx_prior = match_indices[j].first;
        int level = 0;
        for(int k=0;k<priors_steps_featmap.size();k++){
          if(idx_prior<priors_steps_featmap[k]){
            continue;
          }else{
            level += 1;
          }
        }
        if(gt_bboxes[match_indices[j].second].cid==1){
           count_match_per_featmap_hand[level] += 1;
           num_match_percls[0] += 1;
        }
        if(gt_bboxes[match_indices[j].second].cid==2){
          count_match_per_featmap_head[level] += 1;
          num_match_percls[1] += 1;
        }
           
        if(gt_bboxes[match_indices[j].second].cid==3){
          count_match_per_featmap_face[level] += 1; 
          num_match_percls[2] += 1;
        }  
      }

    }
  }
  LOG(INFO)<<"Hand Match "<<count_match_per_featmap_hand[0]
           <<" "<<count_match_per_featmap_hand[1]
           <<" "<<count_match_per_featmap_hand[2]
           <<" "<<count_match_per_featmap_hand[3];
  LOG(INFO)<<"Head Match "<<count_match_per_featmap_head[0]
           <<" "<<count_match_per_featmap_head[1]
           <<" "<<count_match_per_featmap_head[2]
           <<" "<<count_match_per_featmap_head[3];
  LOG(INFO)<<"Face Match "<<count_match_per_featmap_face[0]
           <<" "<<count_match_per_featmap_face[1]
           <<" "<<count_match_per_featmap_face[2]
           <<" "<<count_match_per_featmap_face[3];
  LOG(INFO)<<"NumGT_Hand: "<<num_gt_percls[0]<< "; NumMatch_Hand: "<<num_match_percls[0]<<"."
           <<"NumGT_Head: "<<num_gt_percls[1]<< "; NumMatch_Head: "<<num_match_percls[1]<<"."
           <<"NumGT_Face: "<<num_gt_percls[2]<< "; NumMatch_Face: "<<num_match_percls[2]<<".";
  LOG(INFO)<<"NumGT_PerScale_Hand "<<num_gt_perscale_hand[0]
           <<" "<<num_gt_perscale_hand[1]
           <<" "<<num_gt_perscale_hand[2]
           <<" "<<num_gt_perscale_hand[3];
  LOG(INFO)<<"NumGT_PerScale_Head "<<num_gt_perscale_head[0]
           <<" "<<num_gt_perscale_head[1]
           <<" "<<num_gt_perscale_head[2]
           <<" "<<num_gt_perscale_head[3];
  LOG(INFO)<<"NumGT_PerScale_Face "<<num_gt_perscale_face[0]
           <<" "<<num_gt_perscale_face[1]
           <<" "<<num_gt_perscale_face[2]
           <<" "<<num_gt_perscale_face[3];
}
         
// int num_gt_box = 0;
// for (typename std::map<int, std::vector<LabeledBBox<Dtype> > > ::const_iterator it = all_gt_bboxes_.begin(); it != all_gt_bboxes_.end(); ++it){
//   // num_gt_box += it->second.size();
//   const vector<LabeledBBox<Dtype> >& gt_bboxes = it->second;
//   for (int i=0;i<gt_bboxes.size();i++){
//     LabeledBBox<Dtype> gt = gt_bboxes[i];
//     if(gt.cid==1){
//       float area = (gt.bbox.x2_-gt.bbox.x1_)*(gt.bbox.y2_-gt.bbox.y1_);
//       float bbox_size = std::sqrt(area);
//       if (bbox_size<other_value_){
//         other_value_ = bbox_size;
//         LOG(INFO)<<"current minimum bboxsize "<<other_value_;
//       }
//       if(bbox_size>other_value_2_){
//         other_value_2_ = bbox_size;
//         LOG(INFO)<<"current maximum bboxsize "<<other_value_2_;

//       }
      
//       num_gt_box += 1;
//     }
    
//   }
// }
// LOG(INFO)<<"num_gt_box hand "<<num_gt_box<<" num match pos "<<num_pos_;
/**************************************************************************#
find negative instances from mini-batch (not from each image)
#***************************************************************************/
if(flag_noperson_){
  int num_neg_all = scores_indices.size();
  // LOG(INFO)<<"All "<<num_neg_all<<" negative instances;"<<num_pos_<<" positive instances dense_bbox_loss_layer.";
  int num_neg = std::min(static_cast<int>(num_pos_ * neg_pos_ratio_), num_neg_all);
  std::sort(scores_indices.begin(), scores_indices.end(), PairDescend<pair<int, int>,Dtype>);
  for (int i = 0; i < num_; ++i) {
    vector<int> neg_indices;
    for (int n=0; n<num_neg;++n){
      if (scores_indices[n].second.second == i){
        neg_indices.push_back(scores_indices[n].second.first);
      }
    }
    all_neg_indices_.push_back(neg_indices);
    num_neg_ += neg_indices.size();
  } 
}
// LOG(INFO)<<"Now have  "<<num_neg_all<<" negative instances;"<<num_pos_<<" positive instances.";
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
      for (int k = 0; k < all_pos_indices_[i].size(); ++k) {
        const int prior_id = all_pos_indices_[i][k].first;
        const int gt_id = all_pos_indices_[i][k].second;
        CHECK_LT(prior_id, num_priors_);
        CHECK_LT(gt_id, all_gt_bboxes_[i].size());
        const vector<LabeledBBox<Dtype> >& loc_pred = all_loc_preds[i];
        CHECK_LT(prior_id, loc_pred.size() / (num_classes_ - 1));
        const LabeledBBox<Dtype>& gt_bbox = all_gt_bboxes_[i][gt_id];
        LabeledBBox<Dtype> gt_encode = LabeledBBox_Copy(gt_bbox);
        // copy label
        if (code_type == 0) {
          EncodeBBox_Center(prior_bboxes[prior_id].bbox, prior_variances[prior_id],
                            false, gt_bbox.bbox,
                            &gt_encode.bbox);
        } else {
          EncodeBBox_Corner(prior_bboxes[prior_id].bbox, prior_variances[prior_id],
                            false, gt_bbox.bbox,
                            &gt_encode.bbox);
        }
        loc_gt_data[count * 4]     = gt_encode.bbox.x1_;
        loc_gt_data[count * 4 + 1] = gt_encode.bbox.y1_;
        loc_gt_data[count * 4 + 2] = gt_encode.bbox.x2_;
        loc_gt_data[count * 4 + 3] = gt_encode.bbox.y2_;
        // copy pred
        // NOTE: it's different from classification. the start point is 0 while cls is 1.
        int label;
        if (target_labels_.size() == 0) {
          label = gt_bbox.cid - alias_id_;
        } else {
          label = target_labels_[gt_bbox.cid] - 1;
        }
        CHECK_GE(label,0);
        CHECK_LT(label,num_classes_-1);
        loc_pred_data[count * 4]     = loc_pred[prior_id * (num_classes_ - 1) + label].bbox.x1_;
        loc_pred_data[count * 4 + 1] = loc_pred[prior_id * (num_classes_ - 1) + label].bbox.y1_;
        loc_pred_data[count * 4 + 2] = loc_pred[prior_id * (num_classes_ - 1) + label].bbox.x2_;
        loc_pred_data[count * 4 + 3] = loc_pred[prior_id * (num_classes_ - 1) + label].bbox.y2_;
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
      for (int k = 0; k < all_pos_indices_[i].size(); ++k) {
        const int prior_id = all_pos_indices_[i][k].first;
        const int gt_id = all_pos_indices_[i][k].second;
        CHECK_LT(prior_id, num_priors_);
        CHECK_LT(gt_id, all_gt_bboxes_[i].size());
        int gt_label;
        if (target_labels_.size() == 0) {
          gt_label = all_gt_bboxes_[i][gt_id].cid + 1 - alias_id_;
        } else {
          gt_label = target_labels_[all_gt_bboxes_[i][gt_id].cid];
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
        caffe_copy<Dtype>(num_classes_, conf_data + prior_id * num_classes_,
                          conf_pred_data + count * num_classes_);
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
        normalization_, num_, num_priors_, num_pos_);
    top[0]->mutable_cpu_data()[0] +=
        conf_weight_ * conf_loss_.cpu_data()[0] / normalizer;
  }
  // LOG(INFO) <<"loc_loss_ "<<loc_loss_.cpu_data()[0]<<" conf_loss_ "<<conf_loss_.cpu_data()[0];
}

template <typename Dtype>
void DenseBBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
          for (int k = 0; k < all_pos_indices_[i].size(); ++k) {
            const int prior_id = all_pos_indices_[i][k].first;
            const int gt_id = all_pos_indices_[i][k].second;
            CHECK_LT(prior_id, num_priors_);
            CHECK_LT(gt_id, all_gt_bboxes_[i].size());
            int label;
            if (target_labels_.size() == 0) {
              label = all_gt_bboxes_[i][gt_id].cid - alias_id_;
            } else {
              label = target_labels_[all_gt_bboxes_[i][gt_id].cid] - 1;
            }
            caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
              loc_bottom_diff + (prior_id * (num_classes_ - 1) + label) * 4);
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
          normalization_, num_, num_priors_, num_pos_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      loss_weight *= conf_weight_;
      caffe_scal(conf_pred_.count(), loss_weight,
                 conf_pred_.mutable_cpu_diff());
      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        if (all_pos_indices_[i].size() > 0) {
          for (int k = 0; k < all_pos_indices_[i].size(); ++k) {
            const int prior_id = all_pos_indices_[i][k].first;
            CHECK_LT(prior_id, num_priors_);
            caffe_copy<Dtype>(num_classes_, conf_pred_diff + count * num_classes_,
              conf_bottom_diff + prior_id * num_classes_);
              ++count;
          }
        }
        if (all_neg_indices_[i].size() > 0) {
          const vector<int>& neg_indices = all_neg_indices_[i];
          for (int n = 0; n < neg_indices.size(); ++n) {
            const int prior_id = neg_indices[n];
            CHECK_LT(prior_id, num_priors_);
            caffe_copy<Dtype>(num_classes_, conf_pred_diff + count * num_classes_,
              conf_bottom_diff + prior_id * num_classes_);
              ++count;
            }
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
  all_gt_bboxes_.clear();
}

INSTANTIATE_CLASS(DenseBBoxLossLayer);
REGISTER_LAYER_CLASS(DenseBBoxLoss);

}  // namespace caffe
