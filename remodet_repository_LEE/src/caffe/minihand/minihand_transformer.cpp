#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

#include "caffe/minihand/minihand_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {

template <typename Dtype>
bool MinihandTransformer<Dtype>::Transform(HandAnnoData<Dtype>& anno, cv::Mat* image, vector<LabeledBBox<Dtype> >* bboxes) {
  // (0) 如果没有实例，直接返回
  if (anno.hands.size() == 0) return false;
  // (1) 读取图像: origImage
  cv::Mat origImage = cv::imread(anno.image_path.c_str());
  if (! origImage.data) {
    LOG(INFO) << "Open Error when open " << anno.image_path << ", skipped.";
    return false;
  }
  // (2) 获取裁剪框: crop_roi
  BoundingBox<Dtype> crop_roi;
  bool flag = getCropBBox(anno, &crop_roi, phase_);
  if (! flag) {
    LOG(INFO) << "Random Crop failed when process " << anno.image_path << ", skipped.";
    return false;
  }
  // (3) 标注转换 & Normalize
  transCrop(crop_roi, anno);
  if (anno.hands.size() == 0) {
    LOG(INFO) << "No active samples when process " << anno.image_path << ", skipped.";
    return false;
  }
  // 使用try进行保护，因为可能出错
try {
  // (4) 图像裁剪
  // @背景
  cv::Mat bg_image(crop_roi.get_height(), crop_roi.get_width(), CV_8UC3, cv::Scalar(0,0,0));
  // @patch在原图上的区域
  Dtype pxmin = std::max(crop_roi.x1_, (Dtype)0);  // 原图
  Dtype pymin = std::max(crop_roi.y1_, (Dtype)0);
  Dtype pxmax = std::min(crop_roi.x2_, (Dtype)(origImage.cols));
  Dtype pymax = std::min(crop_roi.y2_, (Dtype)(origImage.rows));
  Dtype patch_w = pxmax - pxmin;
  Dtype patch_h = pymax - pymin;
  cv::Rect patch_orig(pxmin, pymin, patch_w, patch_h);
  // @patch在背景上的区域
  Dtype bxmin = pxmin - crop_roi.x1_;
  Dtype bymin = pymin - crop_roi.y1_;
  cv::Rect patch_bg(bxmin, bymin, patch_w, patch_h);
  // @复制
  cv::Mat opatch = origImage(patch_orig);
  cv::Mat bpatch = bg_image(patch_bg);
  opatch.copyTo(bpatch);
  // (5) Flip: 标注翻转 & 图像翻转
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  bool flip_flag = (dice <= param_.flip_prob()) && param_.do_flip() && (phase_ == caffe::TRAIN);
  if (flip_flag) {
    // 图片翻转
    cv::flip(bg_image, bg_image, 1);
    // 所有标注翻转
    for (int i = 0; i < anno.hands.size(); ++i) {
      Dtype temp = anno.hands[i].bbox.x1_;
      anno.hands[i].bbox.x1_ = 1.0 - anno.hands[i].bbox.x2_;
      anno.hands[i].bbox.x2_ = 1.0 - temp;
    }
  }
  //(6) Color Distortion
  if (phase_ == caffe::TRAIN) {
    bg_image = DistortImage(bg_image, param_.dis_param());
  }
  // (7) Resize
  cv::resize(bg_image, bg_image, cv::Size(param_.resized_width(), param_.resized_height()), cv::INTER_LINEAR);
  // (8) 可视化
  if (param_.save()) {
    for (int i = 0; i < anno.hands.size(); ++i) {
      anno.hands[i].bbox.DrawBoundingBoxNorm(&bg_image);
    }
    static int counter = 0;
    char imagename[256];
    sprintf(imagename, "%s/augment_%06d.jpg", param_.save_path().c_str(), counter);
    cv::imwrite(imagename, bg_image);
    counter++;
  }
  // 将结果返回
  *image = bg_image;
  bboxes->clear();
  for (int i = 0; i < anno.hands.size(); ++i) {
    bboxes->push_back(anno.hands[i]);
  }
  return true;
} catch (exception& e) {
  LOG(INFO) << "Augmention Error when processing " << anno.image_path << ", skipped.";
  return false;
}
}

template <typename Dtype>
void MinihandTransformer<Dtype>::transCrop(BoundingBox<Dtype>& crop_roi, HandAnnoData<Dtype>& anno) {
  // 做每个box的裁剪和归一化
  for (int i = 0; i < anno.hands.size(); ++i) {
    BoundingBox<Dtype> proj;
    Dtype cov = anno.hands[i].bbox.project_bbox(crop_roi, &proj);
    if (cov < 0.25) {
      anno.hands[i].score = -1;
    } else {
      // 纠正bbox
      anno.hands[i].bbox = proj;
    }
  }
  // 消除score < 0的box
  typename vector<LabeledBBox<Dtype> >::iterator it;
  for (it = anno.hands.begin(); it != anno.hands.end();) {
    if (it->score < 0) {
      // 删除
      it = anno.hands.erase(it);
    } else {
      ++it;   // 指向下一个单元
    }
  }
  // 修正数量
  anno.num_hands = anno.hands.size();
}

template <typename Dtype>
bool MinihandTransformer<Dtype>::getCropBBox(HandAnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_roi, Phase phase) {
  // 获取区域的范围
  Dtype temp = sqrt((Dtype)anno.image_width * (Dtype)anno.image_height);
  int rw,rh;
  // if (phase == caffe::TRAIN){
    // rw = int((temp * 8.0 / 3.0) / 16) * 16;// scale area 4
  float area_scale;
  if (param_.has_crop_max_area_scale() && param_.has_crop_min_area_scale()){
    float max_area_scale, min_area_scale;
    CHECK_GE(param_.crop_max_area_scale(), param_.crop_min_area_scale());
    max_area_scale = param_.crop_max_area_scale();
    min_area_scale = param_.crop_min_area_scale();
    caffe_rng_uniform(1, min_area_scale, max_area_scale, &area_scale);
  }
  else{
    area_scale = 2.f;
  }
  CHECK_NE(param_.sample_sixteennine(),param_.sample_ninesixteen());
  if(param_.sample_sixteennine()){
    rw = int((temp * area_scale) / 16) * 16; // scale area 2 #16:9
    rh = rw / 16 * 9;//16:9
  }else if(param_.sample_ninesixteen()){
    rw = int((temp * area_scale) / 9) * 9; // scale area 2 #9:16
    rh = rw / 9 * 16;//9:16
  }
    
  // } else {
  //   rw = int((temp * 8.0 / 3.0) / 16) * 16;
  // }
  // int rh = rw / 16 * 9;//16:9
  int dw = anno.image_width - rw;
  int dh = anno.image_height - rh;
  if (phase == caffe::TRAIN) {
    // 裁剪点
    vector<pair<Dtype,Dtype> > crop_left_pos;
    // 开始依次查找
    for (int i = 0; i < param_.cov_limits_size(); ++i) {
      Dtype cov_limit = param_.cov_limits(i);
      int tries = 100;
      bool found = false;
      while(tries--) {
        float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        Dtype xoffs = dice * dw;
        Dtype yoffs = dice * dh;
        for (int j = 0; j < anno.hands.size(); ++j) {
          BoundingBox<Dtype>& box = anno.hands[j].bbox;
          Dtype ccp_xmin = std::max(box.x1_, xoffs);
          Dtype ccp_ymin = std::max(box.y1_, yoffs);
          Dtype ccp_xmax = std::min(box.x2_, xoffs + rw);
          Dtype ccp_ymax = std::min(box.y2_, yoffs + rh);
          if ((ccp_ymax - ccp_ymin > 0) && (ccp_xmax - ccp_xmin > 0) && ((ccp_ymax - ccp_ymin) * (ccp_xmax - ccp_xmin) /  box.compute_area() >= cov_limit)) {
            crop_left_pos.push_back(std::make_pair(xoffs,yoffs));
            found = true;
            break;
          }
        }
        if (found) break;
      }
    }
    // 随机选取一个
    if (crop_left_pos.size() > 0) {
      int dice_int = static_cast<int>(rand()) % crop_left_pos.size();
      Dtype xoffs = crop_left_pos[dice_int].first;
      Dtype yoffs = crop_left_pos[dice_int].second;
      crop_roi->x1_ = xoffs;
      crop_roi->y1_ = yoffs;
      crop_roi->x2_ = xoffs + rw;
      crop_roi->y2_ = yoffs + rh;
      return true;
    } else {
      // 裁剪失败，返回false
      return false;
    }
  } else {
    Dtype xoffs = dw / 2;
    Dtype yoffs = dh / 2;
    crop_roi->x1_ = xoffs;
    crop_roi->y1_ = yoffs;
    crop_roi->x2_ = xoffs + rw;
    crop_roi->y2_ = yoffs + rh;
    return true;
  }
}

INSTANTIATE_CLASS(MinihandTransformer);

}
