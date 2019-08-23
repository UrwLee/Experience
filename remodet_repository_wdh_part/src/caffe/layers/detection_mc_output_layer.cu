#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/detection_mc_output_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void DetectionMcOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const int num = bottom[0]->num();
  const int layer_height = bottom[0]->channels();
  const int layer_width = bottom[0]->height();
  CHECK_EQ(layer_height, bottom[1]->channels());
  CHECK_EQ(layer_width, bottom[1]->height());
  /**************************************************************************#
  GPU：获取boxes
  #***************************************************************************/
  // 获取boxes
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype* bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  Blob<Dtype> prior_wh;
  vector<int> prior_shape(1, 2 * num_priors_);
  prior_wh.Reshape(prior_shape);
  int count = 0;
  Dtype* mutable_prior_data = prior_wh.mutable_cpu_data();
  for (int i = 0; i < num_priors_; ++i) {
    mutable_prior_data[count++] = prior_width_[i];
    mutable_prior_data[count++] = prior_height_[i];
  }
  const Dtype* prior_data = prior_wh.gpu_data();
  DecodeBBoxesByLocGPU<Dtype>(loc_count, loc_data, prior_data,
    layer_width, layer_height, num_priors_, bbox_data);
  const Dtype* bbox_cpu_data = bbox_preds.cpu_data();
  /**************************************************************************#
  GPU：获取Conf
  #***************************************************************************/
  // 获取conf
  Blob<Dtype> bg_conf;        //object
  Blob<Dtype> class_conf;     //class
  vector<int> conf_shape(4,1);
  conf_shape[0] = num;
  conf_shape[1] = layer_height;
  conf_shape[2] = layer_width;
  conf_shape[3] = num_priors_;
  bg_conf.Reshape(conf_shape);
  conf_shape[3] = num_priors_ * num_classes_;
  class_conf.Reshape(conf_shape);
  Dtype* bg_data = bg_conf.mutable_gpu_data();
  Dtype* class_data = class_conf.mutable_gpu_data();
  // PermuteDataGPU
  int conf_count = bottom[1]->count();
  const Dtype* conf_data = bottom[1]->gpu_data();
  PermuteConfDataToBgClassGPU<Dtype>(conf_count,conf_data,
          num_classes_,bg_data,class_data);
  // bg -> logistic, in-place
  int bg_conf_count = bg_conf.count();
  LogisticGPU<Dtype>(bg_conf_count, bg_data);
  // class -> softmax, in-place
  const Dtype* con_class_data = class_conf.gpu_data();
  SoftMaxGPU<Dtype>(con_class_data, bg_conf_count,
      num_classes_, class_data);
  // multiply: class_conf *= objectness
  const Dtype* con_bg_data = bg_conf.gpu_data();
  const int class_conf_count = class_conf.count();
  UpdateConfByObjGPU<Dtype>(class_conf_count, num_classes_,
        con_bg_data, class_data);
  const Dtype* conf_cpu_data = class_conf.cpu_data();
  /**************************************************************************#
  CPU：获取检测结果
  #***************************************************************************/
  vector<vector<NormalizedBBox> > all_decode_bboxes(num);
  vector<map<int, vector<float> > > all_conf_scores(num);
  all_decode_bboxes.clear();
  all_conf_scores.clear();
  int outsize_loc = layer_width*layer_height*num_priors_*4;
  int outsize_conf = layer_width*layer_height*num_priors_*num_classes_;
  for (int item_id = 0; item_id < num; ++item_id) {
    vector<NormalizedBBox> decode_bboxes;
    map<int, vector<float> > conf_scores;
    for (int i = 0; i < layer_height; ++i) {
      for (int j = 0; j < layer_width; ++j) {
        for (int n = 0; n < num_priors_; ++n) {
          int idx_loc = item_id * outsize_loc + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * 4;
          int idx_conf = item_id * outsize_conf + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * num_classes_;
          // 获取坐标信息
          NormalizedBBox pred_bbox;
          pred_bbox.set_xmin(bbox_cpu_data[idx_loc]);
          pred_bbox.set_ymin(bbox_cpu_data[idx_loc + 1]);
          pred_bbox.set_xmax(bbox_cpu_data[idx_loc + 2]);
          pred_bbox.set_ymax(bbox_cpu_data[idx_loc + 3]);
          ClipBBox(pred_bbox,&pred_bbox);
          decode_bboxes.push_back(pred_bbox);
          for (int c = 1; c < num_classes_ + 1; ++c) {
            conf_scores[c].push_back(conf_cpu_data[idx_conf + c - 1]);
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

INSTANTIATE_LAYER_GPU_FUNCS(DetectionMcOutputLayer);

}  // namespace caffe
