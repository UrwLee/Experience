#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/mask/matching_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BoxMatchingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BoxMatchingParameter& box_matching_param =
      this->layer_param_.box_matching_param();
  // check parameters
  overlap_threshold_ = box_matching_param.overlap_threshold();
  use_difficult_gt_ = box_matching_param.use_difficult_gt();
  CHECK_EQ(use_difficult_gt_, false) << "please do not use difficult bboxes for matching.";
  size_threshold_ = box_matching_param.size_threshold();
  top_k_ = box_matching_param.top_k();
  CHECK_GT(top_k_, 0);
}

template <typename Dtype>
void BoxMatchingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1,1,1,7);
  CHECK_EQ(bottom[1]->width(),9);
}

template <typename Dtype>
void BoxMatchingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* prior_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  num_priors_ = bottom[0]->height() / 4;
  num_gt_ = bottom[1]->height();
  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  map<int, vector<LabeledBBox<Dtype> > > all_gt_bboxes;
  // const int keep_id = 0;
  const vector<int> ids;
  GetGTBBoxes(gt_data, num_gt_, use_difficult_gt_, ids, size_threshold_, &all_gt_bboxes);
  /**************************************************************************#
  获取所有的prior-boxes
  #***************************************************************************/
  vector<LabeledBBox<Dtype> > prior_bboxes;
  vector<vector<Dtype> > prior_variances;
  GetAnchorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);
  /**************************************************************************#
  完成所有任务的匹配统计: 正例统计
  #***************************************************************************/
  match_rois_.clear();
  for (typename map<int, vector<LabeledBBox<Dtype> > >::iterator it = all_gt_bboxes.begin(); it != all_gt_bboxes.end(); ++it) {
    // int bindex = it->first;
    vector<LabeledBBox<Dtype> >& gt_bboxes = it->second;
    map<int, int > match_indices;
    vector<int> unmatch_indices;
    vector<Dtype> match_overlaps;
    MatchAnchorsAndGTs(gt_bboxes, prior_bboxes, 0, overlap_threshold_, overlap_threshold_,
                      &match_overlaps, &match_indices, &unmatch_indices);
    for (map<int, int>::iterator iit = match_indices.begin(); iit != match_indices.end(); ++iit) {
      int prior_id = iit->first;
      int gt_id = iit->second;
      vector<Dtype> roi;
      roi.push_back(match_overlaps[prior_id]);
      roi.push_back(gt_bboxes[gt_id].bindex);
      roi.push_back(gt_bboxes[gt_id].cid);
      roi.push_back(gt_bboxes[gt_id].pid);
      roi.push_back(prior_bboxes[prior_id].bbox.x1_);
      roi.push_back(prior_bboxes[prior_id].bbox.y1_);
      roi.push_back(prior_bboxes[prior_id].bbox.x2_);
      roi.push_back(prior_bboxes[prior_id].bbox.y2_);
      match_rois_.push_back(roi);
    }
  }
  /**************************************************************************#
  排序，选取置信度最高的前TopK个ROI输出
  #***************************************************************************/
  std::stable_sort(match_rois_.begin(), match_rois_.end(), VectorDescend<Dtype>);
  int use_rois = match_rois_.size() > top_k_ ? top_k_ : match_rois_.size();
  match_rois_.resize(use_rois);
  /**************************************************************************#
  输出
  #***************************************************************************/
  if (use_rois == 0) {
    LOG(FATAL) << "Found No GT-Boxes.";
  }
  top[0]->Reshape(1,1,use_rois,7);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < use_rois; ++i) {
    top_data[7*i] = match_rois_[i][1];
    top_data[7*i+1] = match_rois_[i][2];
    top_data[7*i+2] = match_rois_[i][3];
    top_data[7*i+3] = match_rois_[i][4];
    top_data[7*i+4] = match_rois_[i][5];
    top_data[7*i+5] = match_rois_[i][6];
    top_data[7*i+6] = match_rois_[i][7];
  }
}

template <typename Dtype>
void BoxMatchingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_CLASS(BoxMatchingLayer);
REGISTER_LAYER_CLASS(BoxMatching);

}  // namespace caffe
