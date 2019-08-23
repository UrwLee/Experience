#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"
#include "caffe/mask/det_out_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void DetOutLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
    this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes.";
  num_classes_ = detection_output_param.num_classes();
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
  // Conf-threshold
  CHECK(detection_output_param.has_conf_threshold());
  conf_threshold_ = detection_output_param.conf_threshold();
  CHECK_GE(conf_threshold_, 0);
  // nms-threshold
  CHECK(detection_output_param.has_nms_threshold());
  nms_threshold_ = detection_output_param.nms_threshold();
  CHECK_GE(nms_threshold_, 0);
  // size threshold
  CHECK(detection_output_param.has_size_threshold());
  size_threshold_ = detection_output_param.size_threshold();
  CHECK_GE(size_threshold_, 0);
  // nms keep maximum
  CHECK(detection_output_param.has_top_k());
  top_k_ = detection_output_param.top_k();
  CHECK_GT(top_k_, 0);
  // vote or not
  vote_or_not_ = detection_output_param.vote_or_not();
  LOG(INFO) << "Pre:" << vote_or_not_;
  soft_type_ = detection_output_param.soft_type();

  if (bottom.size() == 5) {
    // set internal sigmoid layer
    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[4]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  }
  objectness_score_ = detection_output_param.objectness_score();
}

template <typename Dtype>
void DetOutLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_ = bottom[0]->num();
  //定义box的数量，每个box由四个参数定义
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(num_priors_ * 4 * (num_classes_ - 1), bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  vector<int> top_shape(3, 1);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);

  if (bottom.size() == 5) {
    // set internal sigmoid layer
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  }
}

template <typename Dtype>
void DetOutLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
    this->layer_param_.detection_output_param();
  const Dtype *loc_data = bottom[0]->cpu_data();
  const Dtype *conf_data = bottom[1]->cpu_data();
  const Dtype *prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int num_priors = bottom[2]->height() / 4;
  if (bottom.size() == 5) {
    const Dtype* arm_loc_data = bottom[3]->cpu_data();
    const Dtype* arm_conf_data = bottom[4]->cpu_data();

    all_arm_loc_preds_.clear();

    GetLocPreds(arm_loc_data, num_, num_priors_, &all_arm_loc_preds_);

    sigmoid_bottom_vec_[0] = bottom[4];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    neg_scores_.resize(num);
    neg_scores_.clear();
    // 统计所有类别的置信度信息　[包含bkg]
    GetConfScores(sigmoid_output_data, num, num_priors, num_classes_, &neg_scores_);
  }
  /**************************************************************************#
  获取LOC的估计信息
  #***************************************************************************/
  vector<vector<LabeledBBox<Dtype> > > all_loc_preds;
  GetLocPreds(loc_data, num, num_priors, &all_loc_preds);
  /**************************************************************************#
  获取每个部位的置信度信息
  #***************************************************************************/
  vector<vector<vector<Dtype> > > all_conf_scores(num);
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
  if (bottom.size() == 5) {
    vector<vector<LabeledBBox<Dtype> > > decode_prior_bboxes_;
    const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
    DecodeBBoxes(all_arm_loc_preds_, prior_bboxes, prior_variances, num, code_type,
                 variance_encoded_in_target_, &decode_prior_bboxes_);

    for (int i = 0; i < num; ++i)
    {
      vector<LabeledBBox<Dtype> > loc_bboxes;
      if (code_type == 0) {
        DecodeBBoxes_Center(decode_prior_bboxes_[i], prior_variances, variance_encoded_in_target_,
                            all_loc_preds[i], &loc_bboxes);
      } else {
        DecodeBBoxes_Corner(decode_prior_bboxes_[i], prior_variances, variance_encoded_in_target_,
                            all_loc_preds[i], &loc_bboxes);
      }
      all_decode_bboxes.push_back(loc_bboxes);
    }
  } else {
    const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
    DecodeBBoxes(all_loc_preds, prior_bboxes, prior_variances, num, code_type,
                 variance_encoded_in_target_, &all_decode_bboxes);
  }
  /**************************************************************************#
  NMS
  #***************************************************************************/
  int num_det = 0;
  vector<vector<int> > prior_ids;
  prior_ids.resize(num);
  vector<vector<vector<LabeledBBox<Dtype> > > > all_dets(num);

  CPUTimer img_timer;
  for (int i = 0; i < num; ++i) {
    for (int j = 1; j < num_classes_; ++j) {
      vector<int> indices;
      img_timer.Start();
      //LOG(INFO)<<vote_or_not_;
      if (vote_or_not_ == DetectionOutputParameter_NmsType_Ori) {
        NmsOri(all_decode_bboxes[i], all_conf_scores[i][j],
               conf_threshold_, nms_threshold_, top_k_, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_Fast) {
        NmsFast(all_decode_bboxes[i], all_conf_scores[i][j],
                conf_threshold_, nms_threshold_, top_k_, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_FastVote) {
        const Dtype voting_thre = 0.75;
        NmsFastWithVoting(&all_decode_bboxes[i], all_conf_scores[i][j], conf_threshold_,
                          nms_threshold_, top_k_, voting_thre, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_Power2) {
        NmsOriSoft(all_decode_bboxes[i], all_conf_scores[i][j],
                   conf_threshold_, nms_threshold_, top_k_, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_Power3) {
        NmsOriSoftThree(all_decode_bboxes[i], all_conf_scores[i][j],
                        conf_threshold_, nms_threshold_, top_k_, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_weight04) {
        NmsOriSoftweight04(all_decode_bboxes[i], all_conf_scores[i][j],
                           conf_threshold_, nms_threshold_, top_k_, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_weight04vote) {
        const Dtype voting_thre = 0.75;
        NmsOriSoftweight04WithVoting(all_decode_bboxes[i], all_conf_scores[i][j],
                                     conf_threshold_, nms_threshold_, top_k_, voting_thre, &indices);
      } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_Power2vote) {
        const Dtype voting_thre = 0.75;
        NmsOriSoftPower2WithVoting(all_decode_bboxes[i], all_conf_scores[i][j],
                                   conf_threshold_, nms_threshold_, top_k_, voting_thre, &indices);
      }
      // } else if (vote_or_not_ == DetectionOutputParameter_NmsType_OriSoft && soft_type_ == DetectionOutputParameter_SoftType_Power2vote) {
      //   NmsOriSoft(all_decode_bboxes[i], all_conf_scores[i][j],
      //           conf_threshold_, nms_threshold_, top_k_, &indices);
      // }
      img_timer.Stop();
      // LOG(INFO)<<"top_k_ :"<<top_k_<<";indices: "<<indices.size();
      vector<LabeledBBox<Dtype> > dets;
      for (int p = 0; p < indices.size(); ++p) {
        int prior_id = indices[p];
        if (bottom.size() == 5) {
          if (neg_scores_[i][0][prior_id] > objectness_score_) {
            continue;
          }
        }
        LabeledBBox<Dtype> tbbox;
        tbbox.bindex = i;
        tbbox.cid = j - 1;
        tbbox.pid = p;
        tbbox.score    = all_conf_scores[i][j][prior_id];
        tbbox.bbox.x1_ = all_decode_bboxes[i][prior_id].bbox.x1_;
        tbbox.bbox.y1_ = all_decode_bboxes[i][prior_id].bbox.y1_;
        tbbox.bbox.x2_ = all_decode_bboxes[i][prior_id].bbox.x2_;
        tbbox.bbox.y2_ = all_decode_bboxes[i][prior_id].bbox.y2_;
        tbbox.bbox.clip();
        if (tbbox.bbox.compute_area() < size_threshold_) {
          // LOG(INFO) << "Small boxes, passed. size less than " << size_threshold_;
          continue;
        }
        dets.push_back(tbbox);
        prior_ids[i].push_back(prior_id);
        num_det++;
      }
      all_dets[i].push_back(dets);
    }
  }
  // LOG(INFO) << "Found " << num_det << " examples.";
  /**************************************************************************#
  Output to Top[0]
  #***************************************************************************/
  vector<int> top_shape(2, 1);
  int ndim_detout = detection_output_param.ndim_detout();
  top_shape.push_back(num_det);
  top_shape.push_back(ndim_detout);
  if (num_det == 0) {
    top_shape[2] = 1;
    top[0]->Reshape(top_shape);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(top_shape);
    Dtype *top_data = top[0]->mutable_cpu_data();
    int count = 0;
    for (int i = 0; i < num; ++i) {
      int cnt_prior = 0;
      for (int j = 0; j < num_classes_ - 1; ++j) {
        if (all_dets[i][j].size() == 0) continue;
        vector<LabeledBBox<Dtype> > &dets = all_dets[i][j];
        for (int p = 0; p < dets.size(); ++p) {
          LabeledBBox<Dtype>& tbbox = dets[p];
          top_data[count++] = tbbox.bindex;
          int out_label;
          if (target_labels_.size() == 0) {
            out_label = tbbox.cid + alias_id_;
          } else {
            out_label = target_labels_[tbbox.cid];
          }
          top_data[count++] = out_label;
          top_data[count++] = tbbox.score;
          top_data[count++] = tbbox.bbox.x1_;
          top_data[count++] = tbbox.bbox.y1_;
          top_data[count++] = tbbox.bbox.x2_;
          top_data[count++] = tbbox.bbox.y2_;
          if (ndim_detout == 8) top_data[count++] = prior_ids[i][cnt_prior++];
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetOutLayer, Forward);
#endif

INSTANTIATE_CLASS(DetOutLayer);
REGISTER_LAYER_CLASS(DetOut);

} // namespace caffe
