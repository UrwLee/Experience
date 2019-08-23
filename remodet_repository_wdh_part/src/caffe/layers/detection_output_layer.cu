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

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 将数据从gpu中读入cpu
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();
  const int num_priors = bottom[2]->height() / 4;
  // LOG(INFO) << "haha";
  // 解码box,读到bbox_preds
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype* bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors, bbox_data);
  // 获取所有的检测结果all_decode_bboxes
  const Dtype* bbox_cpu_data = bbox_preds.cpu_data();
  vector<vector<NormalizedBBox> > all_decode_bboxes;
  GetLocPredictions(bbox_cpu_data, num, num_priors, &all_decode_bboxes);

  // 获取所有的置信度信息, 从gpu读到cpu
  // 注意: permute -> [num_,num_priors_,num_classes_,1] ->
  // [num_,num_classes_,num_priors_,1]
  // 坐标变换为classes在前,所以class_major = true
  const Dtype* conf_data;
  Blob<Dtype> conf_permute;
  conf_permute.ReshapeLike(*(bottom[1]));
  Dtype* conf_permute_data = conf_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(conf_permute.count(), bottom[1]->gpu_data(),
      num_classes_, num_priors, 1, conf_permute_data);
  conf_data = conf_permute.cpu_data();
  const bool class_major = true;
  // 获取所有的部位的置信度信息
  vector<vector<float> > all_conf_scores(num);
  GetConfidenceScores(conf_data, num, num_priors, num_classes_,
                      class_major, 1, &all_conf_scores);
  CHECK_EQ(all_conf_scores.size(), num);
  CHECK_EQ(all_conf_scores[0].size(), num_priors);
  // for (int i = 0; i < 10; i++) {
  //   LOG(INFO) << "aaa" << all_conf_scores[0][i];
  //   LOG(INFO) << "===" << bottom[2]->cpu_data()[num_priors * 4 + i];
  // }
  // for (int i = 0; i < 4; i++) {
  //   LOG(INFO) << "bbb" << all_decode_bboxes[0][i].xmin();
  //   LOG(INFO) << "bbb" << all_decode_bboxes[0][i].ymin();
  //   LOG(INFO) << "bbb" << all_decode_bboxes[0][i].xmax();
  //   LOG(INFO) << "bbb" << all_decode_bboxes[0][i].ymax();
  // }
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
    // LOG(INFO) << "[#] Couldn't find any detections";
    top_shape[2] = 1;
    top[0]->Reshape(top_shape);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(top_shape);
    Dtype *top_data = top[0]->mutable_cpu_data();
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
    // LOG(INFO) << "Visualize the input images.";
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

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
