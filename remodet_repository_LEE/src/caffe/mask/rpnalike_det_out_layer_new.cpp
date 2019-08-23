#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"
#include "caffe/mask/rpnalike_det_out_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void RPNAlikeDetOutLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes.";
  num_classes_ = detection_output_param.num_classes();
  CHECK_EQ(num_classes_, 2) << "num_classes_ must be 2.";
  background_label_id_ = detection_output_param.background_label_id();
  CHECK_EQ(background_label_id_, 0) << "background_label_id must be 0.";
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  CHECK_EQ(variance_encoded_in_target_, false);
  alias_id_ = detection_output_param.alias_id();
  // output labels
  if (detection_output_param.target_labels_size() > 0) {
    CHECK_EQ(detection_output_param.target_labels_size(), num_classes_ - 1);
    for (int i = 0; i < num_classes_ - 1; ++i) {
      target_labels_.push_back(detection_output_param.target_labels(i));
    }
  }
  // nms-threshold
  // size threshold
  CHECK(detection_output_param.has_size_threshold());
  size_threshold_ = detection_output_param.size_threshold();
  CHECK_GE(size_threshold_, 0);
  code_type_ = detection_output_param.code_type();
  conf_loss_type_ = detection_output_param.conf_loss_type();
  out_label_type_ = detection_output_param.out_label_type();
  match_type_ = detection_output_param.match_type();
  for (int i = 0; i < detection_output_param.gt_labels_size(); ++i) {
    gt_labels_.push_back(detection_output_param.gt_labels(i));
  }
  size_threshold_ = detection_output_param.size_threshold();
  flag_noperson_ = detection_output_param.flag_noperson();
  overlap_threshold_ = detection_output_param.overlap_threshold();
  neg_overlap_ = detection_output_param.neg_overlap();
  neg_pos_ratio_ = detection_output_param.neg_pos_ratio();
  use_difficult_gt_ = detection_output_param.use_difficult_gt();
  img_w_ = detection_output_param.img_w();
  img_h_ = detection_output_param.img_h();
  iter_ = 0;
  conf_threshold_ = detection_output_param.conf_threshold();
  nms_threshold_  = detection_output_param.nms_threshold();
  top_k_  = detection_output_param.top_k();

}

template <typename Dtype>
void RPNAlikeDetOutLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_ = bottom[0]->num();
  //定义box的数量，每个box由四个参数定义
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(num_priors_ * 4 * (num_classes_-1), bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  //top[0]: rois (input of roi_pooling_layer)
  //top[1]: label of each roi
  vector<int> top_shape(1, 1);
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
  vector<int> top_shape1(1, 1);
  const int label_type = (out_label_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) ? 0 : 1;
  if(label_type){// for logistic
    top_shape1.push_back(num_classes_);
  } else{// for softmax
    top_shape1.push_back(1);
  }
  top[1]->Reshape(top_shape1);
}

template <typename Dtype>
void RPNAlikeDetOutLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *loc_data = bottom[0]->cpu_data();
  const Dtype *conf_data = bottom[1]->cpu_data();
  const Dtype *prior_data = bottom[2]->cpu_data();
  const Dtype *gt_data = bottom[3]->cpu_data();
  const int num = bottom[0]->num();
  const int num_priors = bottom[2]->height() / 4;
  const int num_gt = bottom[3]->height();
  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  map<int, vector<LabeledBBox<Dtype> > > all_gt_bboxes;
  GetGTBBoxes(gt_data, num_gt, use_difficult_gt_, gt_labels_, size_threshold_, &all_gt_bboxes);
  int cnt = 0;
  if (flag_noperson_){
    vector<LabeledBBox<Dtype> >  gt_bbox;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
          all_gt_bboxes[i] = gt_bbox; // write an empty vector box
          cnt += 1;
      }
    }
  }
  /**************************************************************************#
  获取LOC的估计信息
  #***************************************************************************/
  vector<vector<LabeledBBox<Dtype> > > all_loc_preds;//[id_batch][id_prior]
  GetLocPreds(loc_data, num, num_priors, &all_loc_preds);
  /**************************************************************************#
  获取每个部位的置信度信息
  #***************************************************************************/
  vector<vector<vector<Dtype> > > all_conf_scores(num); //[id_batch][id_class][id_prior]
  // 统计所有类别的置信度信息　[包含bkg]
  GetConfScores(conf_data, num, num_priors, num_classes_, &all_conf_scores);
  /**************************************************************************#
  获取prior-boxes信息
  #***************************************************************************/
  vector<LabeledBBox<Dtype> > prior_bboxes;
  vector<vector<Dtype> > prior_variances;
  GetAnchorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);
  /**************************************************************************#
  获取实际的估计box位置
  #***************************************************************************/
  vector<vector<LabeledBBox<Dtype> > > all_decode_bboxes;
  const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
  DecodeBBoxes(all_loc_preds, prior_bboxes, prior_variances, num, code_type,
               variance_encoded_in_target_, &all_decode_bboxes);
  /**************************************************************************#
  获取分类任务的最大置信度信息:
  #***************************************************************************/
  vector<vector<Dtype> > max_scores;
  conf_data = bottom[1]->cpu_data();
  const int loss_type = (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) ? 0 : 1;
  const int match_type = (match_type_ == MultiBoxLossParameter_MatchType_PER_PREDICTION) ? 0 : 1;
  GetHDMScores(conf_data, num_, num_priors_, num_classes_, loss_type, &max_scores);
  num_pos_ = 0;
  num_neg_ = 0;
  vector<pair<Dtype, pair<int, int> > > scores_indices; // pair <score, pari<prior-id, batch-id> >
  vector<vector<LabeledBBox<Dtype> > > pos_loc_boxes, neg_loc_boxes;
  for (int i = 0; i < num; ++i) {
    vector<LabeledBBox<Dtype> > pos_loc_boxes_batch, neg_loc_boxes_batch;
    /**************************************************************************#
    单个样本的匹配列表
    #***************************************************************************/
    map<int, int > match_indices;//[id_pred]-->[id_gt]
    vector<int> match_indices_vec;
    vector<int> unmatch_indices;
    vector<int> neg_indices;
    vector<Dtype> match_overlaps;
    const vector<LabeledBBox<Dtype> >& gt_bboxes = all_gt_bboxes.find(i)->second;
    vector<LabeledBBox<Dtype> > loc_bboxes = all_decode_bboxes[i];
    vector<Dtype> max_scores_filtersize;
    vector<LabeledBBox<Dtype> > loc_bboxes_filtersize;
    for(int j=0;j<loc_bboxes.size();j++){
      if(loc_bboxes[j].bbox.compute_area()>size_threshold_){
        loc_bboxes_filtersize.push_back(loc_bboxes[j]);
        max_scores_filtersize.push_back(max_scores[i][j]);
      }
    }
    const Dtype voting_thre = 0.75;
    vector<int> indices;
    NmsFastWithVoting(&loc_bboxes_filtersize, max_scores_filtersize,conf_threshold_,nms_threshold_, top_k_, voting_thre, &indices);
    vector<LabeledBBox<Dtype> > loc_bboxes_nms;
    for(int j=0;j<indices.size();j++){
      loc_bboxes_nms.push_back(loc_bboxes_filtersize[indices[j]]);
    }
	//LOG(INFO)<<" loc_bboxes "<<loc_bboxes.size()<<" loc_bboxes_nms "<<loc_bboxes_nms.size();
    // match the output of the first detection after nms (loc_bboxes_nms) with gt_bboxes
    MatchAnchorsAndGTs(gt_bboxes, loc_bboxes_nms, match_type, overlap_threshold_, neg_overlap_,
                        &match_overlaps, &match_indices, &unmatch_indices,flag_noperson_);
    // get postive loc_boxes
    for (typename map<int,int >::const_iterator it = match_indices.begin(); it != match_indices.end(); ++it){
        match_indices_vec.push_back(it->first);
    }
	//LOG(INFO)<<" match_indices_vec "<<match_indices_vec.size()<<" gt_bboxes "<<gt_bboxes.size()<<" unmatch_indices "<<unmatch_indices.size();
    int num_pos = match_indices_vec.size();
    // if(match_indices_vec.size()>3*gt_bboxes.size()){
    //   num_pos = 3*gt_bboxes.size();
    //   random_shuffle(match_indices_vec.begin(),match_indices_vec.end());
    // } else{
    //   num_pos = match_indices_vec.size();
    // }
    for (int j=0;j<num_pos;j++){
        pos_loc_boxes_batch.push_back(loc_bboxes_nms[match_indices_vec[j]]);
    }
    num_pos_ += num_pos;
    // get negative loc_boxes
    std::sort(unmatch_indices.begin(),unmatch_indices.end());
    int num_neg = unmatch_indices.size();
    if (num_pos==0){// for images with no person, num_pos==0; we still choose negative instances from these images  
      num_neg = std::min(10,num_neg);
    }else{
      num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
    }
    for(int j=0;j<num_neg;j++){
      neg_loc_boxes_batch.push_back(loc_bboxes_nms[unmatch_indices[j]]);
    } 
    num_neg_ += num_neg; 
    pos_loc_boxes.push_back(pos_loc_boxes_batch);
    neg_loc_boxes.push_back(neg_loc_boxes_batch);
	//LOG(INFO)<<" num_neg "<<num_neg<<" num_pos "<<num_pos<<" num_gt "<<gt_bboxes.size();
   
  }
  for (typename std::map<int, std::vector<LabeledBBox<Dtype> > > ::const_iterator it = all_gt_bboxes.begin(); it != all_gt_bboxes.end(); ++it){
    // num_gt_box += it->second.size();
    const vector<LabeledBBox<Dtype> >& gt_bboxes = it->second;
    int ibatch = it->first;
    for(int i=0;i<gt_bboxes.size();i++){
      pos_loc_boxes[ibatch].push_back(gt_bboxes[i]);
      num_pos_++;
    }
  }
  // LOG(INFO) << "Found " << num_det << " examples.";
  /**************************************************************************#
  Output to Top[0]
  #***************************************************************************/
  if (iter_%20==0){
	  LOG(INFO)<<"num_pos_ "<<num_pos_<<";num_neg_ "<<num_neg_;
  }
  int num_all = num_pos_ + num_neg_;
  vector<int> top_shape0;
  top_shape0.push_back(num_all);
  top_shape0.push_back(5);
  vector<int> top_shape1;
  top_shape1.push_back(num_all);
  const int label_type = (out_label_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) ? 0 : 1;
  if(label_type){// for logistic
    top_shape1.push_back(num_classes_);
  } else{// for softmax
    top_shape1.push_back(1);
  }
  if (num_all == 0) {
    top_shape0[0] = 1;
    top[0]->Reshape(top_shape0);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
    top_shape1[0] = 1;
    top[1]->Reshape(top_shape1);
    caffe_set<Dtype>(top[1]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(top_shape0);
    top[1]->Reshape(top_shape1);
    Dtype *top_data0 = top[0]->mutable_cpu_data();
    Dtype *top_data1 = top[1]->mutable_cpu_data();
    int cnt0 = 0;
    int cnt1 = 0;
	//LOG(INFO)<<"num "<<num;
	//LOG(INFO)<<"num "<<num;
    for(int ibatch=0;ibatch<num;ibatch++){
      for(int ipos =0;ipos<pos_loc_boxes[ibatch].size();ipos++){
		  //LOG(INFO)<<"ibatch "<<ibatch;
        top_data0[cnt0++] = ibatch;
        top_data0[cnt0++] = pos_loc_boxes[ibatch][ipos].bbox.x1_*img_w_;
        top_data0[cnt0++] = pos_loc_boxes[ibatch][ipos].bbox.y1_*img_h_;
        top_data0[cnt0++] = pos_loc_boxes[ibatch][ipos].bbox.x2_*img_w_;
        top_data0[cnt0++] = pos_loc_boxes[ibatch][ipos].bbox.y2_*img_h_;
        if (label_type){// for logistic
          top_data1[cnt1++] = 0.0;
          top_data1[cnt1++] = 1.0;
        } else{// for softmax
          top_data1[cnt1++] = 1.0;
        }
      }
      for(int ineg =0;ineg<neg_loc_boxes[ibatch].size();ineg++){
		  //LOG(INFO)<<"ibatch "<<ibatch;
        top_data0[cnt0++] = ibatch;
        top_data0[cnt0++] = neg_loc_boxes[ibatch][ineg].bbox.x1_*img_w_;
        top_data0[cnt0++] = neg_loc_boxes[ibatch][ineg].bbox.y1_*img_h_;
        top_data0[cnt0++] = neg_loc_boxes[ibatch][ineg].bbox.x2_*img_w_;
        top_data0[cnt0++] = neg_loc_boxes[ibatch][ineg].bbox.y2_*img_h_;
        if (label_type){// for logistic
          top_data1[cnt1++] = 1.0;
          top_data1[cnt1++] = 0.0;
        } else{
          top_data1[cnt1++] = 0;
        }
      }

    }
  }
  iter_ ++;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RPNAlikeDetOutLayer, Forward);
#endif

INSTANTIATE_CLASS(RPNAlikeDetOutLayer);
REGISTER_LAYER_CLASS(RPNAlikeDetOut);

} // namespace caffe
