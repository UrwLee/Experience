#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::LayerSetUp(
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
  CHECK(detection_output_param.has_visual_param()) << "Must specify visual_param.";
  visual_param_ = detection_output_param.visual_param();
  CHECK(visual_param_.has_visualize()) << "visualize must be specified.";
  // visual_param
  visualize_ = visual_param_.visualize();

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
void DetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_ = bottom[0]->num();
  //定义box的数量，每个box由四个参数定义
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(num_priors_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  vector<int> top_shape(3, 1);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *loc_data = bottom[0]->cpu_data();
  const Dtype *conf_data = bottom[1]->cpu_data();
  const Dtype *prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int num_priors = bottom[2]->height() / 4;
  /**************************************************************************#
  获取LOC的估计信息
  #***************************************************************************/
  //LOG(INFO) << "haha";
  vector<vector<NormalizedBBox> > all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors, &all_loc_preds);
  /**************************************************************************#
  获取每个部位的置信度信息:1
  #***************************************************************************/
  vector<vector<float> > all_conf_scores(num);  //该类下每个box的置信度
  conf_data = bottom[1]->cpu_data();
  GetConfidenceScores(conf_data, num, num_priors, num_classes_,
                      1, &all_conf_scores);
  CHECK_EQ(all_conf_scores.size(), num);
  CHECK_EQ(all_conf_scores[0].size(), num_priors);
  /**************************************************************************#
  获取prior-boxes信息
  #***************************************************************************/
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);
  /**************************************************************************#
  获取实际的估计box位置
  #***************************************************************************/
  vector<vector<NormalizedBBox> > all_decode_bboxes;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                  code_type_, variance_encoded_in_target_, &all_decode_bboxes);
  /**************************************************************************#
  NMS
  #***************************************************************************/
  int num_det = 0;
  vector<map<int, vector<NormalizedBBox> > > all_part;
  for (int i = 0; i < num; ++i) {
    vector<int> indices;
    ApplyNMSFastUnit(&all_decode_bboxes[i], all_conf_scores[i],
                      conf_threshold_, nms_threshold_, top_k_,
                      &indices);
    map<int, vector<NormalizedBBox> > parts;
    AddParts(all_decode_bboxes[i], all_conf_scores[i],
             indices, &num_det, &parts);
    all_part.push_back(parts);
  }

  /**************************************************************************#
  统计检测数
  #***************************************************************************/
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_det);
  top_shape.push_back(7);
  if (num_det == 0) {
    // LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = 1;
    top[0]->Reshape(top_shape);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(top_shape);
    Dtype *top_data = top[0]->mutable_cpu_data();
    // 复制到Top[0]中
    int count = 0;
    for (int i = 0; i < num; ++i) {
      if (all_part[i].size() == 0) continue;
      map<int, vector<NormalizedBBox> > &parts = all_part[i];
      for (map<int, vector<NormalizedBBox> >::iterator it = parts.begin();
           it != parts.end(); ++it){
        const int label = it->first;
        const vector<NormalizedBBox> &det_bboxes = it->second;
        if (det_bboxes.size() == 0) continue;
        for (int j = 0; j < det_bboxes.size(); ++j){
          top_data[count++] = i;
          top_data[count++] = label;
          top_data[count++] = det_bboxes[j].score();
          top_data[count++] = det_bboxes[j].xmin();
          top_data[count++] = det_bboxes[j].ymin();
          top_data[count++] = det_bboxes[j].xmax();
          top_data[count++] = det_bboxes[j].ymax();
        }
      }
    }
  }
  /**************************************************************************#
  可视化输出.
  #***************************************************************************/
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    const int channels = bottom[3]->channels();
    const int height = bottom[3]->height();
    const int width = bottom[3]->width();
    const int nums = bottom[3]->num();
    CHECK_GE(nums, 1);
    CHECK_EQ(nums,num);
    const Dtype* img_data = bottom[3]->cpu_data();
    // 将图片数据转换为cvimg
    for(int i = 0; i < num; ++i){
      cv::Mat input_img(height, width, CV_8UC3, cv::Scalar(0,0,0));
      blobTocvImage(img_data, height, width, channels, &input_img);
      cv_imgs.push_back(input_img);
      img_data += bottom[3]->offset(1);
    }
    // 可视化结果输出
    VisualizeBBox(cv_imgs, all_part, visual_param_);
#endif
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectionOutputLayer);
REGISTER_LAYER_CLASS(DetectionOutput);

} // namespace caffe
