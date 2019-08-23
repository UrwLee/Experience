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

#include "caffe/minihand/minihand_sample_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

#define DEBUG_WDH false

namespace caffe {
// func: 
  /*
    1. 裁剪 bg
    2. 放置到 原始图像
    3. resize 512*288
  */
template <typename Dtype>
bool MinihandSampleTransformer<Dtype>::Transform(HandAnnoData<Dtype>& anno, cv::Mat* image, vector<LabeledBBox<Dtype> >* bboxes, cv::Mat& bg_image) {
  // (0) 如果没有实例，直接返回
  if (anno.hands.size() == 0) return false;
  // (1) 读取图像: origImage
  cv::Mat origImage = cv::imread(anno.image_path.c_str());
  // LOG(INFO) << "==================1 ============ origImage " <<  origImage.cols << " * "<< origImage.rows;
  if (! origImage.data) {
    LOG(INFO) << "Open Error when open " << anno.image_path << ", skipped.";
    return false;
  }

  // (1.1)  抠手增广
  // cutGtChangeBg(anno , origImage, bg_image);

  // (2) 获取裁剪框: crop_roi
  BoundingBox<Dtype> crop_roi;
  bool flag = getCropBBox(anno, &crop_roi, phase_); // func: 获得裁剪坐标, return ture 表示有符合要求的裁剪坐标
  if (! flag) {
    LOG(INFO) << "Random Crop failed when process " << anno.image_path << ", skipped.";
    return false;
  }
  // (3) 标注转换 & Normalize
  transCrop(crop_roi, anno); // func: 转换成crop 后的相对坐标
  if (anno.hands.size() == 0) { 
    LOG(INFO) << "No active samples when process " << anno.image_path << ", skipped.";
    return false;
  }
  // 使用try进行保护，因为可能出错
try {
  // (4) 图像裁剪
  // @背景
  cv::Mat bg_image(crop_roi.get_height(), crop_roi.get_width(), CV_8UC3, cv::Scalar(0,0,0)); 
  // crop_roi 坐标为 (-111, -111, 111,111)
  // func: crop 背景框与原始图片的交集
  Dtype pxmin = std::max(crop_roi.x1_, (Dtype)0);  // 原图
  Dtype pymin = std::max(crop_roi.y1_, (Dtype)0);
  Dtype pxmax = std::min(crop_roi.x2_, (Dtype)(origImage.cols)); // 720*1280 
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
  
   // resize 
  // int resize_w;
  // int resize_h;
  // resize_w = (int)(patch_w * 0.333);
  // resize_h = (int)(patch_h * 0.333);
  // LOG(INFO) << "xxxx" << resize_w << "," << resize_h;
  // resize(opatch, opatch, Size(resize_w, resize_h), INTER_LINEAR);  // func: resize (w,h)
  // char imagename2[256];
  // sprintf(imagename2, "%s/augment_%06d.jpg", param_.save_path().c_str(), 1);
  // cv::imwrite(imagename2, opatch);
  
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
void MinihandSampleTransformer<Dtype>::transCrop(BoundingBox<Dtype>& crop_roi, HandAnnoData<Dtype>& anno) {
  // 做每个box的裁剪和归一化
  for (int i = 0; i < anno.hands.size(); ++i) { // func: 
    BoundingBox<Dtype> proj;
    Dtype cov = anno.hands[i].bbox.project_bbox(crop_roi, &proj); // func: 获得以crop_roi为背景的 gt相对坐标(归一化后), 计算gt 的coverage
    if (cov < 0.25) {
      anno.hands[i].score = -1;
    } else {
      // 纠正bbox
      anno.hands[i].bbox = proj; // func: gt 都是以 crop 为原点进行相应修改
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
bool MinihandSampleTransformer<Dtype>::getCropBBox(HandAnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_roi, Phase phase) {
  // 获取区域的范围
  Dtype temp = sqrt((Dtype)anno.image_width * (Dtype)anno.image_height); // func: 
  int rw,rh;
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
  if(param_.sample_sixteennine()){ // func: 面积扩大area_scale倍的　16:9 背景
    rw = int((temp * area_scale) / 16) * 16; // scale area  #16:9
    rh = rw / 16 * 9;//16:9
  }else if(param_.sample_ninesixteen()){
    rw = int((temp * area_scale) / 9) * 9; // scale area  #9:16
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
    for (int i = 0; i < param_.cov_limits_size(); ++i) { // func: 获得符合要求的背景裁剪坐标
      Dtype cov_limit = param_.cov_limits(i); // func: 获得条件
      int tries = 100;
      bool found = false;
      while(tries--) { // func: 
        float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // func: 产生0~1随机数
     

        Dtype xoffs = dice * dw;
        Dtype yoffs = dice * dh;
        for (int j = 0; j < anno.hands.size(); ++j) { // func: 对所有gt(hand face) 进行交集检查,目标要选择框中gt的 背景框
          BoundingBox<Dtype>& box = anno.hands[j].bbox;
          // func: 新背景框 与gt 求交集, 与gt相交部分与gt iou> cov_limit 时 保留此背景框
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
    if (crop_left_pos.size() > 0) { // func: 
      int dice_int = static_cast<int>(rand()) % crop_left_pos.size(); // func: 随机在符合crop 要求的坐标中选取一个
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

// ================================
/*
  对于尺度小于512 288 的图像进行填充
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::borderImage(cv::Mat& image){
  /*
  BORDER_REPLICATE 重复： 就是对边界像素进行复制
  BORDER_REFLECT 反射：对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb 反射
  BORDER_REFLECT_101 反射101： 例子：gfedcb|abcdefgh|gfedcba
  BORDER_WRAP 外包装：cdefgh|abcdefgh|abcdefg
  BORDER_CONSTANT 常量复制：例子：iiiiii|abcdefgh|iiiiiii */

  cv::Size shape_img = image.size();
  // 图片尺寸比512 ,288 小进行填充
  if (shape_img.width < 512 || shape_img.height < 288){
    int add_x = std::max(0, 512 - shape_img.width);
    int add_y = std::max(0, 288 - shape_img.height);
    if(DEBUG_WDH) LOG(INFO) << " 边缘填充: " << add_y << ", " << add_x ;
    cv::copyMakeBorder(image, image, 0, add_x, 0,  add_y, BORDER_WRAP);
  }
}

/*
  获得anno 的实际坐标
  原始图像的 box是归一化的, 进行恢复
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::unNormalize(HandAnnoData<Dtype>& anno) {
 //
  const int image_width  = anno.image_width; // 原始图片尺寸
  const int image_height = anno.image_height;
  //
 
  if(DEBUG_WDH) LOG(INFO) << "anno num "<< anno.hands.size();
  // for (int i = 0; i < anno.hands.size(); ++i) { // func: 对anno 中每个gt进行归一化 
  //   BoundingBox<Dtype>& bbox = anno.hands[i].bbox;
  //   bbox.clip();
  //   bbox.x1_ *= (Dtype)image_width;
  //   bbox.x2_ *= (Dtype)image_width;
  //   bbox.y1_ *= (Dtype)image_height;
  //   bbox.y2_ *= (Dtype)image_height; 
   
  // }

   
  typename vector<LabeledBBox<Dtype> >::iterator it;
  for (it = anno.hands.begin(); it != anno.hands.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    Dtype xmin = (int)gt_bbox.x1_;
    Dtype xmax = (int)gt_bbox.x2_;
    Dtype ymin = (int)gt_bbox.y1_;
    Dtype ymax = (int)gt_bbox.y2_;

    if(( xmin >= xmax ) || ( ymin >= ymax )) {
      if(DEBUG_WDH) LOG(INFO) <<"false " << gt_bbox.x1_ << " , "<< gt_bbox.y1_ << " , "<< gt_bbox.x2_<< " , "<< gt_bbox.y2_;
      it = anno.hands.erase(it);  //以迭代器为参数，删除元素3，并把数据删除后的下一个元素位置返回给迭代器。 此时不执行++it;
      anno.num_hands -= 1;
    } else{ ++it; } // 
  }

  if(DEBUG_WDH)  LOG(INFO) << "anno num " << anno.hands.size();
} 
 
/*
  获得包含所有gt的最小矩形框, 此矩形框的坐标是原图坐标
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::RelatGtBbox(HandAnnoData<Dtype> anno, BoundingBox<Dtype>& relat_gt_bbox){
  Dtype minx=0, miny=0, maxx=0, maxy=0;
  for(int i=0; i < anno.hands.size(); ++i){ // func: find 框选所有gt的最小矩形
    BoundingBox<Dtype> bbox = anno.hands[i].bbox;
    if (minx > bbox.x1_) minx = bbox.x1_; 
    if (miny > bbox.y1_) miny = bbox.y1_;
    if (maxx < bbox.x2_) maxx = bbox.x2_;
    if (maxy < bbox.y2_) maxy = bbox.y2_;
  }
  relat_gt_bbox.x1_ = minx;
  relat_gt_bbox.y1_ = miny;
  relat_gt_bbox.x2_ = maxx;
  relat_gt_bbox.y2_ = maxy;
}

/*
  将原图片中的gt 的坐标, 变换为背景图片中的坐标, 如果原图片中的 gt包围圈 超过了背景图的大小, 将gt包围圈缩小到背景图片中, 
  同时对应的 gt 坐标变为背景坐标, 需要缩小时 进行缩小.
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::changeInstance(HandAnnoData<Dtype> anno, HandAnnoData<Dtype>& temp_anno, 
    Dtype x_relat_bbox, Dtype y_relat_bbox, 
    BoundingBox<Dtype> relat_gt_bbox, Dtype scale){
    for(int i=0; i < temp_anno.hands.size(); ++i){ // func: find 框选所有gt的最小矩形
      BoundingBox<Dtype>& bbox = temp_anno.hands[i].bbox;
      if(DEBUG_WDH) LOG(INFO) << "实际gt位置 num: " << i <<" , gt 坐标 " << bbox.x1_ << " ,"<< bbox.y1_ << " ,"<< bbox.x2_<< " ,"<< bbox.y2_;
      bbox.x1_ = bbox.x1_ - relat_gt_bbox.x1_ + x_relat_bbox;
      bbox.y1_ = bbox.y1_ - relat_gt_bbox.y1_ + y_relat_bbox;
      bbox.x2_ = bbox.x2_ - relat_gt_bbox.x1_ + x_relat_bbox;
      bbox.y2_ = bbox.y2_ - relat_gt_bbox.y1_ + y_relat_bbox;
      if((scale) < Dtype(1)){
        bbox.x1_ *= scale; 
        bbox.y1_ *= scale;
        bbox.x2_ *= scale;
        bbox.y2_ *= scale;
      }
      if(DEBUG_WDH) LOG(INFO) << "背景gt位置 num: " << i <<" , gt 坐标 " << bbox.x1_ << " ,"<< bbox.y1_ << " ,"<< bbox.x2_<< " ,"<< bbox.y2_;
  }
}

/*
  将 原图中的 gt 图片抠出来 resize之后放到背景图上, 
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::cutGtImgChangeBg(cv::Mat raw_image, cv::Mat& bg_img, HandAnnoData<Dtype> anno, HandAnnoData<Dtype> temp_anno){
  for(int i=0; i < anno.hands.size(); ++i){
    BoundingBox<Dtype> raw_box = anno.hands[i].bbox;
    BoundingBox<Dtype>& temp_box = temp_anno.hands[i].bbox;

    if(DEBUG_WDH) LOG(INFO) <<"背景gt  "<< temp_box.x1_ << ", " <<  temp_box.y1_ << " , " << temp_box.x2_ << " ,"<< temp_box.y2_ << ", "<< temp_box.get_width() << ", " <<temp_box.get_height();
    if(DEBUG_WDH) LOG(INFO) <<"原图gt  "<< raw_box.x1_ << ", " <<  raw_box.y1_ << " , " << raw_box.x2_ << " ,"<< raw_box.y2_ << ", " << raw_box.get_width() << ", " <<raw_box.get_height();
    cv::Mat raw_image_gt = raw_image(cv::Rect((int)raw_box.x1_, (int)raw_box.y1_, (int)raw_box.get_width(), (int)raw_box.get_height()));
    cv::Mat bg_img_ROI   = bg_img(cv::Rect((int)temp_box.x1_, (int)temp_box.y1_, (int)temp_box.get_width(), (int)temp_box.get_height()));
    cv::resize(raw_image_gt, raw_image_gt, cv::Size((int)temp_box.get_width(), (int)temp_box.get_height()), CV_INTER_AREA);
    if(DEBUG_WDH) LOG(INFO) <<"原图gt  "<< raw_image_gt.cols << ", " <<  raw_image_gt.rows;
    
    cv::addWeighted(bg_img_ROI, 0.0, raw_image_gt, 1.0, 0.0, bg_img_ROI);
  }
}

/*
  获得 gt包围圈相对于 背景图的坐标, 目的是计算gt 包围圈在背景图中的实际位置
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::normRelatGtBox(BoundingBox<Dtype>& temp_box, BoundingBox<Dtype> box){
  temp_box.x1_ = Dtype(0);
  temp_box.y1_ = Dtype(0);
  temp_box.x2_ = box.x2_ - box.x1_;
  temp_box.y2_ = box.y2_ - box.y1_;
}


/*
  修改temp_anno, 
    1. 将temp_anno.image_width 改为背景图的尺寸
    2. 将
*/
template <typename Dtype>
void MinihandSampleTransformer<Dtype>::instancesFitBg(cv::Size shape_bg_img, BoundingBox<Dtype>& relat_gt_bbox, 
  HandAnnoData<Dtype>& temp_anno , HandAnnoData<Dtype> anno){
  temp_anno = anno ;// 初始化 为相同
  temp_anno.image_width = shape_bg_img.width;
  temp_anno.image_height = shape_bg_img.height;
  if(DEBUG_WDH) LOG(INFO) << "背景图 尺寸 " <<temp_anno.image_width << ", "<< temp_anno.image_height;

  // func: 获得temp box, 
  BoundingBox<Dtype> temp_relat_gt_bbox = relat_gt_bbox;
  normRelatGtBox(temp_relat_gt_bbox, relat_gt_bbox); 
  if(DEBUG_WDH) LOG(INFO) << "归一化 gt包围圈 " << temp_relat_gt_bbox.x1_ <<" , "<< temp_relat_gt_bbox.y1_ <<" , "<< temp_relat_gt_bbox.x2_<<" , "<< temp_relat_gt_bbox.y2_;
  // 缩放比例
  Dtype scale_w = shape_bg_img.width / temp_relat_gt_bbox.get_width();
  Dtype scale_h = shape_bg_img.height / temp_relat_gt_bbox.get_height();
  if(DEBUG_WDH) LOG(INFO) << "w 比例 " << scale_w << ", h 比例 " << scale_h;

  Dtype scale_fit_bg = std::min(scale_w, scale_h); 
  if(scale_fit_bg < Dtype(1)){
    temp_relat_gt_bbox.x1_ *= scale_fit_bg;
    temp_relat_gt_bbox.y1_ *= scale_fit_bg;
    temp_relat_gt_bbox.x2_ *= scale_fit_bg;
    temp_relat_gt_bbox.y2_ *= scale_fit_bg;
  }
  else scale_fit_bg = Dtype(1);

  Dtype x_relat_bbox, y_relat_bbox; // 随机选取 temp_relat_gt_bbox 在背景图上的坐标
  // temp_relat_gt_bbox 在背景图上的 最大位置, 
  Dtype x_bg_max = shape_bg_img.width - temp_relat_gt_bbox.get_width();
  Dtype y_bg_max = shape_bg_img.height - temp_relat_gt_bbox.get_height();
  if(DEBUG_WDH) LOG(INFO) << " gt 包围圈 坐标范围 " << x_bg_max << ", " << y_bg_max << ", " << temp_relat_gt_bbox.get_width() +x_bg_max << " , "<< temp_relat_gt_bbox.get_height() + y_bg_max;
  caffe_rng_uniform(1, std::min(Dtype(0), x_bg_max), std::max(x_bg_max, Dtype(0)), &x_relat_bbox);
  caffe_rng_uniform(1, std::min(Dtype(0), y_bg_max), std::max(y_bg_max, Dtype(0)), &y_relat_bbox);
  if(DEBUG_WDH) LOG(INFO) << " gt 包围圈 随机后位置 " << x_relat_bbox << ", " << y_relat_bbox <<", " << temp_relat_gt_bbox.get_width() + x_relat_bbox << " , "<< temp_relat_gt_bbox.get_height() + y_relat_bbox;

  // 修改gt 参数 fit 背景
  changeInstance(anno, temp_anno, x_relat_bbox, y_relat_bbox, relat_gt_bbox, scale_fit_bg);
}



template <typename Dtype>
void MinihandSampleTransformer<Dtype>::cutGtChangeBg(HandAnnoData<Dtype>& anno , cv::Mat& raw_image, cv::Mat& bg_img){
  // 如果 超过概率
  if (false) { return; }

  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > 0.5) { return; }
  // 找到一张背景图, 获得w,h 
  // cv::Mat bg_img = cv::imread("/home/xjx/awdh_old/my_python/Data_manage/tool/automoldroid/test_augmentation/image4.jpg");  
  // cv::Mat bg_img = cv::imread("/home/xjx/Datasets/ditan/691.jpg");  


  borderImage(bg_img);   

  cv::Size shape_bg_img = bg_img.size();
  BoundingBox<Dtype> relat_gt_bbox;
  unNormalize(anno); // func: 转换为绝对坐标
  RelatGtBbox(anno, relat_gt_bbox);
  if(DEBUG_WDH) LOG(INFO)  << "gt包围圈: " << relat_gt_bbox.x1_ << " , " << relat_gt_bbox.y1_<< " , " << relat_gt_bbox.x2_<< " , " << relat_gt_bbox.y2_;

  // 随机选择 relag_gt_bbox 在新背景上的位置
  HandAnnoData<Dtype> temp_anno ; 
  instancesFitBg(shape_bg_img, relat_gt_bbox, temp_anno, anno);

  // 将raw_image 中的gt iou扣到 背景图的相应位置.
  cutGtImgChangeBg(raw_image, bg_img, anno, temp_anno);
  raw_image = bg_img;
  anno = temp_anno;
  // Normalize(anno);  // 对新anno 进行归一化
}
 // =================================================

INSTANTIATE_CLASS(MinihandSampleTransformer);

}
