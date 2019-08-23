#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/mask/det_eval_layer.hpp"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

template <typename Dtype>
void DetEvalLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  // background_label_id_ = detection_evaluate_param.background_label_id();
  // CHECK_EQ(background_label_id_, 0);
  evaluate_difficult_gt_ = detection_evaluate_param.evaluate_difficult_gt();
  CHECK(detection_evaluate_param.has_num_classes())
       << "num_classes must be specified.";
  num_classes_ = detection_evaluate_param.num_classes();
  CHECK_EQ(detection_evaluate_param.boxsize_threshold_size(), 7) <<
     "7 levels should be defined.";
  size_thre_[0] = detection_evaluate_param.boxsize_threshold(0);
  size_thre_[1] = detection_evaluate_param.boxsize_threshold(1);
  size_thre_[2] = detection_evaluate_param.boxsize_threshold(2);
  size_thre_[3] = detection_evaluate_param.boxsize_threshold(3);
  size_thre_[4] = detection_evaluate_param.boxsize_threshold(4);
  size_thre_[5] = detection_evaluate_param.boxsize_threshold(5);
  size_thre_[6] = detection_evaluate_param.boxsize_threshold(6);
  for(int i=0;i<7;i++){
    level_cnt_.push_back(0);
  }
  // 获取iou_thre_
  CHECK_EQ(detection_evaluate_param.iou_threshold_size(), 3) <<
     "3 diff_levels should be defined.";
  iou_thre_[0] = detection_evaluate_param.iou_threshold(0);
  iou_thre_[1] = detection_evaluate_param.iou_threshold(1);
  iou_thre_[2] = detection_evaluate_param.iou_threshold(2);
  // active labels
  for (int i = 0; i < detection_evaluate_param.gt_labels_size(); ++i) {
    gt_labels_.push_back(detection_evaluate_param.gt_labels(i));
  }
}

template <typename Dtype>
void DetEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 7);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  // [bindex, cid, pid, is_diff, iscrowd, xmin, ymin, xmax, ymax]
  CHECK_EQ(bottom[1]->width(), 9);
  vector<int> top_shape(4, 1);
  top_shape[0] = 1;   //
  top_shape[1] = 1;   //diff: @IOU
  top_shape[2] = 1;   //level: @size
  top_shape[3] = 7;   //(num_classes_ - 1 + num_dets) * 5
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetEvalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  CHECK_EQ(bottom[0]->width(), 7) << "detection results-width must be 7.";
  CHECK_EQ(bottom[1]->width(), 9) << "ground-truth width must be 9.";
  // ===========================================================================
  // Get Det results
  map<int, map<int, vector<LabeledBBox<Dtype> > > > all_detections; //  bindex, cid , box
  GetDetections(det_data, bottom[0]->height(), &all_detections);
  // ===========================================================================
  // Get GT results
  map<int, map<int, vector<LabeledBBox<Dtype> > > > all_gt_bboxes;
  GetGTBBoxes(gt_data, bottom[1]->height(), evaluate_difficult_gt_, gt_labels_, &all_gt_bboxes);
  // ===========================================================================
  // Get leveld GT results
  vector<map<int, map<int, vector<LabeledBBox<Dtype> > > > > leveld_gtboxes;
  leveld_gtboxes.resize(7);
  GetLeveledGTBBoxes(size_thre_,all_gt_bboxes, &leveld_gtboxes); // func: 计算box的 面积大小跑
  // for(int i=0;i<7;i++){
  //   level_cnt_[i]+=leveld_gtboxes[i].size();
  // }
  // LOG(INFO)<<level_cnt_[0]<<" "
  //            <<level_cnt_[1]<<" "
  //            <<level_cnt_[2]<<" "
  //            <<level_cnt_[3]<<" "
  //            <<level_cnt_[4]<<" "
  //            <<level_cnt_[5]<<" "
  //            <<level_cnt_[6];
  // ===========================================================================
  // all res [diff][level]
  map<int, map<int, vector<vector<Dtype> > > > all_res;
  for (int d = 0; d < 3; ++d) {
    for (int l = 0; l < 7; ++l) {
      GetLeveledEvalDetections(leveld_gtboxes[l], all_detections, size_thre_[l], iou_thre_[d],
                               num_classes_, l, d, gt_labels_, &all_res[d][l]);
    }
  }
  // output Top[0]
  int num_out = 0;
  for (int d = 0; d < 3; ++d) {
    for (int l = 0; l < 7; ++l) {
      num_out += all_res[d][l].size();
    }
  }
  // LOG(INFO)<<"hzw num_out"<<num_out;
  vector<int> top_shape(4, 1);
  top_shape[0] = 1;
  top_shape[1] = 1;
  top_shape[2] = num_out;
  top_shape[3] = 7;
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();

  int count = 0;
  for (int d = 0; d < 3; ++d) {
    for (int l = 0; l < 7; ++l) {
      vector<vector<Dtype> >& l_res = all_res[d][l];
      for (int i = 0; i < l_res.size(); ++i) {
        CHECK_EQ(l_res[i].size(), 7);
        for (int j = 0; j < l_res[i].size(); ++j) {
          top_data[count++] = l_res[i][j];
        }
      }
    }
  }
  CHECK_EQ(count, 7 * num_out);
}

INSTANTIATE_CLASS(DetEvalLayer);
REGISTER_LAYER_CLASS(DetEval);

}
