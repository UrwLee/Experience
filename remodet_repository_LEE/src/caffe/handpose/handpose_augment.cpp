#include "caffe/handpose/handpose_augment.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
namespace caffe {

HandPoseAugmenter::HandPoseAugmenter(const bool do_flip, const float flip_prob, const int resized_width,
                  const int resized_height,const bool save, const string& save_path,
                  const DistortionParameter& param,const float bbox_extend_min,const float bbox_extend_max, 
                  const float rotate_angle, const bool clip,const bool flag_augIntrain) {
    do_flip_ = do_flip;
    flip_prob_ = flip_prob;
    resized_width_ = resized_width;
    resized_height_ = resized_height;
    save_ = save;
    save_path_ = save_path;
    bbox_extend_min_ = bbox_extend_min;
    bbox_extend_max_ = bbox_extend_max;
    rotate_angle_ = rotate_angle;
    clip_ = clip;
    param_ = param;
    flag_augIntrain_ = flag_augIntrain;

}

void HandPoseAugmenter::aug(HandPoseInstance& anno, cv::Mat* image, int* id, caffe::Phase phase) {
  // 读取图片
  cv::Mat oimage = cv::imread(anno.path_.c_str());
  if (! oimage.data) {
    LOG(FATAL) << "Error when open image: " << anno.path_;
  }
  //NOTE: the GT must satisfy bbox_w == bbox_h;
  int xmin = anno.box_.xmin();
  int ymin = anno.box_.ymin();
  int xmax = anno.box_.xmax();
  int ymax = anno.box_.ymax();
  int bbox_w = xmax - xmin;
  int bbox_h = ymax - ymin; //
  int extend_w = 0;
  int extend_h = 0;
  // LOG(INFO)<<oimage.cols<<" "<<oimage.rows<<" "<<anno.path_.c_str();
  // LOG(INFO)<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<" read "<<anno.path_.c_str();
  if (phase == caffe::TEST) {
    // 直接裁剪
    extend_w = (int)(bbox_w * (bbox_extend_min_ - 1.0)/2.0);
    extend_h = (int)(bbox_h * (bbox_extend_min_ - 1.0)/2.0);
    xmin = max(0, xmin - extend_w);
    ymin = max(0, ymin - extend_h);
    xmax = min(oimage.cols, xmax + extend_w);
    ymax = min(oimage.rows, ymax + extend_h);
    cv::Rect roi(xmin, ymin, xmax-xmin, ymax-ymin);
    cv::Mat crop_patch = oimage(roi);
    // Resize
    cv::resize(crop_patch, *image, cv::Size(resized_width_, resized_height_), cv::INTER_LINEAR);
  } else {
    // (1) 随机裁剪
    int cp_xmin = 0;
    int cp_ymin = 0;
    int cp_xmax = 0;
    int cp_ymax = 0;
    if(flag_augIntrain_) {
      float dice_x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float dice_y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float dice_scale = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float extend_scale = bbox_extend_min_ + (bbox_extend_max_ - bbox_extend_min_)*dice_scale;
      int extend_w = (int)(bbox_w * (extend_scale - 1.0) / 2.0);
      int extend_h = (int)(bbox_h * (extend_scale - 1.0) / 2.0);
      cp_xmin = max(0, xmin - extend_w);
      cp_ymin = max(0, ymin - extend_h);
      cp_xmax = min(oimage.cols, xmax + extend_w);
      cp_ymax = min(oimage.rows, ymax + extend_h);
      // LOG(INFO)<<cp_xmin<<" "<<cp_ymin<<" "<<cp_xmax<<" "<<cp_ymax<<" extend "<<anno.path_.c_str();
      // random drift parameter
      int extend_w_min = (int)(bbox_w * (bbox_extend_min_ - 1.0) / 2.0);
      int extedn_h_min = (int)(bbox_h * (bbox_extend_min_ - 1.0) / 2.0);
      float xmin_extend = max(0, xmin - extend_w_min);
      float ymin_extend = max(0, ymin - extedn_h_min);
      int drift_x = (int)((cp_xmin - xmin_extend) * (2.0 * dice_x - 1.0));
      int drift_y = (int)((cp_ymin - ymin_extend) * (2.0 * dice_y - 1.0));
      cp_xmin = max(0, cp_xmin + drift_x);
      cp_ymin = max(0, cp_ymin + drift_y);
      cp_xmax = min(oimage.cols, cp_xmax + drift_x);
      cp_ymax = min(oimage.rows, cp_ymax + drift_y);
      // LOG(INFO)<<cp_xmin<<" "<<cp_ymin<<" "<<cp_xmax<<" "<<cp_ymax<<" drift "<<anno.path_.c_str();
        
      // LOG(INFO)<<anno.path_<<" cp_xmin "<<cp_xmin<<" cp_xmax "<<cp_xmax<<" cp_ymin "<<cp_ymin<<" cp_ymax "<<cp_ymax<<"oimage.cols "<<oimage.cols<<"oimage.rows "<<oimage.rows;
      // LOG(INFO)<<anno.path_<<" anno.box_.xmin() "<<anno.box_.xmin()<<" anno.box_.xmax "<<anno.box_.xmax()<<" anno.box_.ymin "<<anno.box_.ymin()<<" anno.box_.ymax "<<anno.box_.ymax();
    } else{
      extend_w = (int)(bbox_w * (bbox_extend_min_ - 1.0)/2.0);
      extend_h = (int)(bbox_h * (bbox_extend_min_ - 1.0)/2.0);
      cp_xmin = max(0, xmin - extend_w);
      cp_ymin = max(0, ymin - extend_h);
      cp_xmax = min(oimage.cols, xmax + extend_w);
      cp_ymax = min(oimage.rows, ymax + extend_h);
    }
    // LOG(INFO)<<cp_xmin<<" "<<cp_ymin<<" "<<cp_xmax<<" "<<cp_ymax<<" cp "<<anno.path_.c_str();
    cv::Rect roi(cp_xmin, cp_ymin, cp_xmax-cp_xmin, cp_ymax-cp_ymin);
    cv::Mat crop_patch = oimage(roi);
    // (2) 随机翻转
    float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    bool flip_flag = (dice <= flip_prob_) && do_flip_;
    if (flip_flag) {
      cv::flip(crop_patch, crop_patch, 1);
      int id = anno.id_;
      if(id == 5){
        anno.id_ = 6;
      }
      if(id == 6){
        anno.id_ = 5;
      }
    }
    // (3) 颜色处理
    crop_patch = DistortImage(crop_patch, param_);
    // (4) random rotate
    if(!anno.is_rotated_){
      float dice_rotate = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float degree = (dice_rotate - 0.5) * 2 * rotate_angle_;

      Point2f center(crop_patch.cols/2.0, crop_patch.rows/2.0);
      Mat R = getRotationMatrix2D(center, degree, 1.0);
      Rect bbox = RotatedRect(center, crop_patch.size(), degree).boundingRect();
      // 输出坐标矫正到0,0
      R.at<double>(0,2) += bbox.width/2.0 - center.x;
      R.at<double>(1,2) += bbox.height/2.0 - center.y;
      //多余的部分全部用128/128/128填充
      cv::Mat crop_patch_aug;
      warpAffine(crop_patch, crop_patch_aug, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(104,117,123));
      // (5) Resize
      cv::resize(crop_patch_aug, *image, cv::Size(resized_width_, resized_height_), cv::INTER_LINEAR);
    }else{
      cv::resize(crop_patch, *image, cv::Size(resized_width_, resized_height_), cv::INTER_LINEAR);
    }
  }//end else train
    // 指定其ID
    *id = anno.id_;
    // 保存
    if (save_) {
      static int counter = 0;
      char imagename[256];
      sprintf(imagename, "%s/augment_%06d.jpg", save_path_.c_str(), counter);
      char clsid[10];
      sprintf(clsid,"%d",anno.id_);
      cv::putText(*image, clsid, cv::Point(0,96),cv::FONT_HERSHEY_PLAIN,2,cv::Scalar(0,0,255));
      cv::imwrite(imagename, *image);
      counter++;
    }


}//end aug

}//end caffe
