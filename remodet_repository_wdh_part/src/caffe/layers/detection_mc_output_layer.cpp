#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "caffe/layers/detection_mc_output_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void DetectionMcOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const DetectionMcOutputParameter &detection_mc_output_param =
      this->layer_param_.detection_mc_output_param();
  CHECK(detection_mc_output_param.has_num_classes()) << "Must specify num_classes.";
  num_classes_ = detection_mc_output_param.num_classes();
  CHECK_GE(num_classes_, 1);
  CHECK(detection_mc_output_param.has_visual_param()) << "Must specify visual_param.";
  visual_param_ = detection_mc_output_param.visual_param();
  CHECK(visual_param_.has_visualize()) << "visualize must be specified.";
  visualize_ = visual_param_.visualize();

  // Conf-threshold
  CHECK(detection_mc_output_param.has_conf_threshold());
  conf_threshold_ = detection_mc_output_param.conf_threshold();
  CHECK_GE(conf_threshold_, 0);

  // nms-threshold
  CHECK(detection_mc_output_param.has_nms_threshold());
  nms_threshold_ = detection_mc_output_param.nms_threshold();
  CHECK_GE(nms_threshold_, 0);

  // size threshold
  CHECK(detection_mc_output_param.has_boxsize_threshold());
  boxsize_threshold_ = detection_mc_output_param.boxsize_threshold();
  CHECK_GE(boxsize_threshold_, 0);

  // nms keep maximum
  CHECK(detection_mc_output_param.has_top_k());
  top_k_ = detection_mc_output_param.top_k();
  CHECK_GT(top_k_, 0);

  clip_ = detection_mc_output_param.clip();
  //code_loc_type
  code_loc_type_ = detection_mc_output_param.code_loc_type();

  // priors
  prior_width_.clear();
  prior_height_.clear();
  prior_width_.push_back(Dtype(0.95));
  prior_height_.push_back(Dtype(0.95));
  if (detection_mc_output_param.boxsize_size() > 0 && detection_mc_output_param.pwidth_size() > 0) {
    LOG(FATAL) << "boxsize and pwidth/height could not be provided at the same time.";
  } else if (detection_mc_output_param.boxsize_size() == 0 && detection_mc_output_param.pwidth_size() == 0) {
    LOG(FATAL) << "Must provide boxsize or pwidth.";
  }

  if (detection_mc_output_param.boxsize_size() > 0) {
    CHECK_GT(detection_mc_output_param.aspect_ratio_size(), 0);
    for (int i = 0; i < detection_mc_output_param.boxsize_size(); ++i) {
      CHECK_GT(detection_mc_output_param.boxsize(i), 0);
      CHECK_LT(detection_mc_output_param.boxsize(i), 1);
      Dtype base_size = detection_mc_output_param.boxsize(i);
      for (int j = 0; j < detection_mc_output_param.aspect_ratio_size(); ++j) {
        Dtype ratio = detection_mc_output_param.aspect_ratio(j);
        Dtype w = base_size * sqrt(ratio);
        if (clip_) w = std::min(w, Dtype(1));
        Dtype h = base_size / sqrt(ratio);
        if (clip_) h = std::min(h, Dtype(1));
        prior_width_.push_back(w);
        prior_height_.push_back(h);
      }
    }
    num_priors_ = prior_width_.size();
  } else {
    CHECK_EQ(detection_mc_output_param.pwidth_size(), detection_mc_output_param.pheight_size());
    for (int i = 0; i < detection_mc_output_param.pwidth_size(); ++i) {
      CHECK_GT(detection_mc_output_param.pwidth(i), 0);
      CHECK_LE(detection_mc_output_param.pwidth(i), 1);
      CHECK_GT(detection_mc_output_param.pheight(i), 0);
      CHECK_LE(detection_mc_output_param.pheight(i), 1);
      prior_width_.push_back(detection_mc_output_param.pwidth(i));
      prior_height_.push_back(detection_mc_output_param.pheight(i));
    }
    num_priors_ = prior_width_.size();
  }
}

template <typename Dtype>
void DetectionMcOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_ = bottom[0]->num();
  CHECK_EQ(num_priors_ * 4, bottom[0]->width());
  CHECK_EQ(num_priors_ * (num_classes_ + 1), bottom[1]->width());
  vector<int> top_shape(3, 1);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionMcOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *loc_data = bottom[0]->cpu_data();
  const Dtype *conf_data = bottom[1]->cpu_data();
  const int num = bottom[0]->num();

  int layer_height = bottom[0]->channels();
  int layer_width = bottom[0]->height();
  CHECK_EQ(layer_height, bottom[1]->channels());
  CHECK_EQ(layer_width, bottom[1]->height());

  /**************************************************************************#
  获取所有boxes和评分信息
  #***************************************************************************/
  vector<vector<NormalizedBBox> > all_decode_bboxes(num);
  vector<map<int, vector<float> > > all_conf_scores(num);
  all_decode_bboxes.clear();
  all_conf_scores.clear();
  int outsize_loc = layer_width*layer_height*num_priors_*4;
  int outsize_conf = layer_width*layer_height*num_priors_*(1 + num_classes_);
  for (int item_id = 0; item_id < num; ++item_id) {
    vector<NormalizedBBox> decode_bboxes;
    map<int, vector<float> > conf_scores;
    for (int i = 0; i < layer_height; ++i) {
      for (int j = 0; j < layer_width; ++j) {
        for (int n = 0; n < num_priors_; ++n) {
          int idx_loc = item_id * outsize_loc + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * 4;
          int idx_conf = item_id * outsize_conf + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * (num_classes_ + 1);
          // 获取坐标信息
          NormalizedBBox pred_bbox = get_NormalizedBBoxbyLoc(
                                      loc_data,prior_width_,prior_height_,
                                      n,idx_loc,j,i,layer_width,layer_height,code_loc_type_);
          NormalizedBBox corner_pred_box;
          CenterToCorner(pred_bbox, &corner_pred_box);
          ClipBBox(corner_pred_box,&corner_pred_box);
          decode_bboxes.push_back(corner_pred_box);
          // 获取评分信息
          float object_score = conf_data[idx_conf];
          for (int c = 1; c < num_classes_ + 1; ++c) {
            conf_scores[c].push_back(object_score * conf_data[idx_conf + c]);
          }
        }
      }
    }
    all_decode_bboxes.push_back(decode_bboxes);
    all_conf_scores.push_back(conf_scores);
  }
  /**************************************************************************#
  定义输出结果
  #***************************************************************************/
  int num_det = 0;
  vector<map<int, vector<NormalizedBBox> > > all_dets(num);
  vector<map<int, vector<int> > > all_indices(num);
  all_dets.clear();
  all_indices.clear();
  /**************************************************************************#
  NMS: 获取检测索引列表
  #***************************************************************************/
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    for (int c = 1; c < num_classes_ + 1; ++c) {
      const vector<float>& indices_c = all_conf_scores[i][c];
      ApplyNMSFast(all_decode_bboxes[i],indices_c,
                  conf_threshold_, nms_threshold_, top_k_, &indices[c]);
    }
    all_indices.push_back(indices);
  }
  /**************************************************************************#
  获取检测结果
  #***************************************************************************/
  for (int item_id = 0; item_id < num; ++item_id) {
    map<int, vector<NormalizedBBox> > item_dets;
    const map<int, vector<int> >& indices = all_indices[item_id];
    for (map<int, vector<int> >::const_iterator it = indices.begin();
          it != indices.end(); ++it) {
      int label = it->first;
      if (it->second.size() == 0) continue;
      for (int i = 0; i < it->second.size(); ++i) {
        int idx = it->second[i];
        CHECK_LT(idx, all_decode_bboxes[item_id].size());
        // 获取检测结果
        NormalizedBBox det_bbox;
        // 获取坐标
        det_bbox.CopyFrom(all_decode_bboxes[item_id][idx]);
        float box_size = BBoxSize(det_bbox);
        if (box_size < boxsize_threshold_) continue;
        // 获取评分
        float score = all_conf_scores[item_id][label][idx];
        det_bbox.set_score(score);
        // 送入样本结果之中
        item_dets[label].push_back(det_bbox);
        num_det ++;
      }
    }
    all_dets.push_back(item_dets);
  }
  /**************************************************************************#
  检测结果输出 -> Eval
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
    int count = 0;
    for (int item_id = 0; item_id < num; ++item_id) {
      if (all_dets[item_id].size() == 0) continue;
      map<int, vector<NormalizedBBox> > &item_dets = all_dets[item_id];
      for (map<int, vector<NormalizedBBox> >::iterator it = item_dets.begin();
           it != item_dets.end(); ++it){
        const int label = it->first;
        const vector<NormalizedBBox> &det_bboxes = it->second;
        if (det_bboxes.size() == 0) continue;
        for (int i = 0; i < det_bboxes.size(); ++i){
          top_data[count++] = item_id;
          top_data[count++] = label;
          top_data[count++] = det_bboxes[i].score();
          top_data[count++] = det_bboxes[i].xmin();
          top_data[count++] = det_bboxes[i].ymin();
          top_data[count++] = det_bboxes[i].xmax();
          top_data[count++] = det_bboxes[i].ymax();
        }
      }
    }
  }
  /**************************************************************************#
  可视化输出
  #***************************************************************************/
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    const int channels = bottom[2]->channels();
    const int height = bottom[2]->height();
    const int width = bottom[2]->width();
    const int nums = bottom[2]->num();
    CHECK_GE(nums, 1);
    CHECK_EQ(nums,num);
    const Dtype* img_data = bottom[2]->cpu_data();
    // 将图片数据转换为cvimg
    for(int i = 0; i < num; ++i){
      cv::Mat input_img(height, width, CV_8UC3, cv::Scalar(0,0,0));
      blobTocvImage(img_data, height, width, channels, &input_img);
      cv_imgs.push_back(input_img);
      img_data += bottom[2]->offset(1);
    }
    // 可视化结果输出
    VisualizeBBox(cv_imgs, all_dets, visual_param_);
#endif
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectionMcOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectionMcOutputLayer);
REGISTER_LAYER_CLASS(DetectionMcOutput);

} // namespace caffe
