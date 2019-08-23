#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"
#include "caffe/mask/dense_det_out_2nms_layer.hpp"
#include "caffe/util/myimg_proc.hpp"
 
namespace caffe {

template <typename Dtype>
void DenseDetOut2nmsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes.";
  num_classes_ = detection_output_param.num_classes();
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
}

template <typename Dtype>
void DenseDetOut2nmsLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
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
}

template <typename Dtype>
void DenseDetOut2nmsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionOutputParameter &detection_output_param =
      this->layer_param_.detection_output_param();
  const Dtype *loc_data = bottom[0]->cpu_data();
  const Dtype *conf_data = bottom[1]->cpu_data();
  const Dtype *prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int num_priors = bottom[2]->height() / 4;
  /**************************************************************************#
  获取LOC的估计信息
  #***************************************************************************/
  vector<vector<LabeledBBox<Dtype> > > all_loc_preds;
  GetLocPreds(loc_data, num, num_priors * (num_classes_ - 1), &all_loc_preds);
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
  // [num,class,prior_id]
  vector<vector<vector<LabeledBBox<Dtype> > > > all_decode_bboxes; // shape:
  const int code_type = (code_type_ == PriorBoxParameter_CodeType_CENTER_SIZE) ? 0 : 1;
  DecodeDenseBBoxes(all_loc_preds,prior_bboxes,prior_variances,num,
                    num_classes_, code_type, variance_encoded_in_target_,
                    &all_decode_bboxes);
  /**************************************************************************#
  NMS
  #***************************************************************************/
  int num_det = 0;
  vector<vector<vector<LabeledBBox<Dtype> > > > all_dets(num);
  vector<vector<int> > prior_ids;
  prior_ids.resize(num);

  vector<vector<LabeledBBox<Dtype> > > all_decode_bboxes_2nms(num); // shape: [num][(c-1)*Np]
  vector<vector<Dtype> > all_conf2nms(num);
 
  const bool use_voting = true;
  for (int i = 0; i < num; ++i) {
    for (int j = 1; j < num_classes_; ++j) {
      vector<int> indices;
      if (use_voting) {
        const Dtype voting_thre = 0.75;
        if (false){
            NmsFastWithVotingChangeScore(&all_decode_bboxes[i][j-1], &all_conf_scores[i][j], conf_threshold_,
                          nms_threshold_, top_k_, voting_thre, &indices);
          //   for (vector<int>::iterator it = indices.begin(); it != indices.end();) {
          //     if (all_conf_scores[i][j][*it] < 0.8) {
          // // LOG(INFO) << scores[*it];
          //     indices.erase(it);
          //   }
          // } // sort
        }
        else {
            NmsFastWithVoting(&all_decode_bboxes[i][j-1], all_conf_scores[i][j], conf_threshold_,
                          nms_threshold_, top_k_, voting_thre, &indices);
        }
      } else {
        NmsFast(all_decode_bboxes[i][j-1], all_conf_scores[i][j],
                conf_threshold_, nms_threshold_, top_k_, &indices);
      }
      


      vector<LabeledBBox<Dtype> > dets;
      for (int p = 0; p < indices.size(); ++p) {
        int prior_id = indices[p];
        LabeledBBox<Dtype> tbbox;
        tbbox.bindex = i;
        // LOG(INFO)<<"hzw tbbox.bindex"<<tbbox.bindex;
        tbbox.cid = j-1;
        tbbox.pid = p;
        tbbox.score    = all_conf_scores[i][j][prior_id];
        tbbox.bbox.x1_ = all_decode_bboxes[i][j-1][prior_id].bbox.x1_;
        tbbox.bbox.y1_ = all_decode_bboxes[i][j-1][prior_id].bbox.y1_;
        tbbox.bbox.x2_ = all_decode_bboxes[i][j-1][prior_id].bbox.x2_;
        tbbox.bbox.y2_ = all_decode_bboxes[i][j-1][prior_id].bbox.y2_;
        tbbox.bbox.clip();
        if (tbbox.bbox.compute_area() < size_threshold_) continue;
        dets.push_back(tbbox);
        prior_ids[i].push_back(prior_id);
        num_det++;
        // 2nms det, conf 
        all_decode_bboxes_2nms[i].push_back(tbbox); // 把 (c-1) * Np 放入
        all_conf2nms[i].push_back(all_conf_scores[i][j][prior_id]); 
      }
      all_dets[i].push_back(dets); // [num]
    }
  }

  /**************************************************************************#
     second NMS
  #***************************************************************************/
  Dtype conf_thre_2nms = conf_threshold_; // 2次nms的conf 和 一次相同,
  Dtype nms_thre_2 = 0.7;
  Dtype top_k_2nms = 200; 
  Dtype voting_thre_2nms = 0.75;

  vector<vector<LabeledBBox<Dtype>  > > all_dets2nms(num);
  for (int i = 0; i < num; ++i) {
    vector<int> indices_2nms;
    // LOG(INFO) << "second nms num:" << all_decode_bboxes_2nms[i].size();
    NmsFastWithVoting(&all_decode_bboxes_2nms[i], all_conf2nms[i], conf_thre_2nms,
                          nms_thre_2, top_k_2nms, voting_thre_2nms, &indices_2nms);
    for (int p = 0; p < indices_2nms.size(); ++p){
      int prior_id_2nms = indices_2nms[p];
      all_dets2nms[i].push_back(all_decode_bboxes_2nms[i][prior_id_2nms]); 
    }
  }


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
      
      if (all_dets2nms[i].size() == 0) continue;
      vector<LabeledBBox<Dtype> > &dets = all_dets2nms[i];
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
        if(ndim_detout==8) top_data[count++] = prior_ids[i][cnt_prior++];
      }
      
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DenseDetOut2nmsLayer, Forward);
#endif

INSTANTIATE_CLASS(DenseDetOut2nmsLayer);
REGISTER_LAYER_CLASS(DenseDetOut2nms);

} // namespace caffe
