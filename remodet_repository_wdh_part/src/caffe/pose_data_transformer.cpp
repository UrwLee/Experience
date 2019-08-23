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

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "caffe/pose_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {
using namespace boost::property_tree;
// 读入所有标注文件和数据
template<typename Dtype>
bool PoseDataTransformer<Dtype>::ReadMetaDataFromXml(const string& xml_file, const string& root_dir, MetaData& meta) {
  ptree pt;
  read_xml(xml_file, pt);
  // 读取路径
  string temp_img = pt.get<string>("Annotations.ImagePath");
  // string temp_mask_all = pt.get<string>("Annotations.MaskAllPath");
  string temp_mask_miss = pt.get<string>("Annotations.MaskMissPath");
  meta.img_path = root_dir + temp_img;
  // meta.mask_all_path = root_dir + temp_mask_all;
  meta.mask_miss_path = root_dir + temp_mask_miss;
  // 读取Metadata
  meta.dataset = pt.get<string>("Annotations.MetaData.dataset");
  meta.isValidation = (pt.get<int>("Annotations.MetaData.isValidation") == 0 ? false : true);
  int width = pt.get<int>("Annotations.MetaData.width");
  int height = pt.get<int>("Annotations.MetaData.height");
  meta.img_size = Size(width,height);
  meta.numOtherPeople = pt.get<int>("Annotations.MetaData.numOtherPeople");
  meta.people_index = pt.get<int>("Annotations.MetaData.people_index");
  meta.annolist_index = pt.get<int>("Annotations.MetaData.annolist_index");
  // objpos & scale
  meta.objpos.x = pt.get<float>("Annotations.MetaData.objpos.center_x");
  meta.objpos.y = pt.get<float>("Annotations.MetaData.objpos.center_y");
  meta.objpos -= Point2f(1,1);
  meta.scale_self = pt.get<float>("Annotations.MetaData.scale");
  meta.area = pt.get<float>("Annotations.MetaData.area");
  // joints of self
  if (meta.dataset.find("COCO") != string::npos) {
    np_ = 17;
  } else if (meta.dataset.find("MPII") != string::npos) {
    np_ = 16;
  } else {
    LOG(FATAL) << "Unknown dataset type: " << meta.dataset;
  }
  meta.joint_self.joints.resize(np_);
  meta.joint_self.isVisible.resize(np_);
  for(int i = 0; i < np_; ++i) {
    char temp_x[256], temp_y[256], temp_vis[256];
    sprintf(temp_x, "Annotations.MetaData.joint_self.kp_%d.x", i+1);
    sprintf(temp_y, "Annotations.MetaData.joint_self.kp_%d.y", i+1);
    sprintf(temp_vis, "Annotations.MetaData.joint_self.kp_%d.vis", i+1);
    meta.joint_self.joints[i].x = pt.get<float>(temp_x);
    meta.joint_self.joints[i].y = pt.get<float>(temp_y);
    meta.joint_self.joints[i] -= Point2f(1,1);
    int isVisible = pt.get<int>(temp_vis);
    meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
       meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height) {
      meta.joint_self.isVisible[i] = 2;
    }
  }
  // box
  meta.bbox.x1_ = pt.get<float>("Annotations.MetaData.bbox.xmin");
  meta.bbox.y1_ = pt.get<float>("Annotations.MetaData.bbox.ymin");
  meta.bbox.x2_ = meta.bbox.x1_ + pt.get<float>("Annotations.MetaData.bbox.width");
  meta.bbox.y2_ = meta.bbox.y1_ + pt.get<float>("Annotations.MetaData.bbox.height");
  // other people
  meta.objpos_other.clear();
  meta.scale_other.clear();
  meta.joint_others.clear();
  meta.area_other.clear();
  if (meta.numOtherPeople > 0) {
    meta.objpos_other.resize(meta.numOtherPeople);
    meta.scale_other.resize(meta.numOtherPeople);
    meta.joint_others.resize(meta.numOtherPeople);
    meta.area_other.resize(meta.numOtherPeople);
    for(int p = 0; p < meta.numOtherPeople; p++){
      // ojbpos & scale
      char temp_x[256], temp_y[256], temp_scale[256], temp_area[256];
      sprintf(temp_x, "Annotations.MetaData.objpos_other.objpos_%d.center_x", p+1);
      sprintf(temp_y, "Annotations.MetaData.objpos_other.objpos_%d.center_y", p+1);
      sprintf(temp_scale, "Annotations.MetaData.scale_other.scale_%d", p+1);
      sprintf(temp_area, "Annotations.MetaData.area_other.area_%d", p+1);
      meta.objpos_other[p].x = pt.get<float>(temp_x);
      meta.objpos_other[p].y = pt.get<float>(temp_y);
      meta.objpos_other[p] -= Point2f(1,1);
      meta.scale_other[p] = pt.get<float>(temp_scale);
      meta.area_other[p] = pt.get<float>(temp_area);
      // joints
      meta.joint_others[p].joints.resize(np_);
      meta.joint_others[p].isVisible.resize(np_);
      for(int i = 0; i < np_; i++) {
        char joint_x[256], joint_y[256], joint_vis[256];
        sprintf(joint_x, "Annotations.MetaData.joint_others.joint_%d.kp_%d.x", p+1, i+1);
        sprintf(joint_y, "Annotations.MetaData.joint_others.joint_%d.kp_%d.y", p+1, i+1);
        sprintf(joint_vis, "Annotations.MetaData.joint_others.joint_%d.kp_%d.vis", p+1, i+1);
        meta.joint_others[p].joints[i].x = pt.get<float>(joint_x);
        meta.joint_others[p].joints[i].y = pt.get<float>(joint_y);
        meta.joint_others[p].joints[i] -= Point2f(1,1);
        int isVisible = pt.get<int>(joint_vis);
        meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
        if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
           meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
          meta.joint_others[p].isVisible[i] = 2;
        }
      }
    }
  }
  return true;
}

// 转换joints: COCO from 17 -> 18
template<typename Dtype>
void PoseDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  TransformJoints(meta.joint_self);
  for(int i = 0; i < meta.joint_others.size(); i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::TransformJoints(Joints& j) {
  Joints jo = j;
  // COCO
  if(np_ == 17) {
    int COCO_to_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    // 转换为18个, 包含Neck
    jo.joints.resize(18);
    jo.isVisible.resize(18);
    for(int i = 0; i < 18; i++) {
      jo.joints[i] = (j.joints[COCO_to_1[i]-1] + j.joints[COCO_to_2[i]-1]) * 0.5;
      if(j.isVisible[COCO_to_1[i]-1] >= 2 || j.isVisible[COCO_to_2[i]-1] >= 2) {
        jo.isVisible[i] = 2;
      } else {
        jo.isVisible[i] = j.isVisible[COCO_to_1[i]-1] && j.isVisible[COCO_to_2[i]-1];
      }
    }
  } else if(np_ == 16) {
    // MPII
    int MPI_to_coco[18] = {-1,8,13,12,11,14,15,16,3,2,1,4,5,6,-1,-1,-1,-1};
    jo.joints.resize(18);
    jo.isVisible.resize(18);
    for(int i = 0; i < 18; i++) {
      if (MPI_to_coco[i] > 0) {
        jo.joints[i] = j.joints[MPI_to_coco[i]-1];
        jo.isVisible[i] = j.isVisible[MPI_to_coco[i]-1];
      } else {
        // masked
        jo.joints[i].x = -1;
        jo.joints[i].y = -1;
        jo.isVisible[i] = 2;
      }
    }
  } else {
    LOG(FATAL) << "num_parts should be 17 for COCO and 16 for MPI.";
  }
  j = jo;
}

template<typename Dtype>
PoseDataTransformer<Dtype>::PoseDataTransformer(const PoseDataTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

// 转换顶层API
template<typename Dtype>
void PoseDataTransformer<Dtype>::Transform_nv(const string& xml_file,
                                Dtype* transformed_data,
                                Dtype* transformed_vec_mask,
                                Dtype* transformed_heat_mask,
                                Dtype* transformed_vecmap,
                                Dtype* transformed_heatmap) {
  MetaData meta;
  // 读入参数信息
  if (!ReadMetaDataFromXml(xml_file, param_.root_dir(), meta)) {
    LOG(FATAL) << "Error found in reading from: " << xml_file;
  }
  // 读入图片和mask
  string& img_path = meta.img_path;
  string& mask_miss_path = meta.mask_miss_path;
  CPUTimer process_timer;
  process_timer.Start();
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
  cv::Mat mask_miss = cv::imread(mask_miss_path, CV_LOAD_IMAGE_GRAYSCALE);
  process_timer.Stop();
  LOG(INFO) << "   imread: " << process_timer.MilliSeconds() << " ms.";
  if (!img.data) {
    LOG(FATAL) << "Open error: " << img_path;
  }
  if (!mask_miss.data) {
    LOG(FATAL) << "Open error: " << mask_miss_path;
  }
  if ((img.cols != mask_miss.cols) || (img.rows != mask_miss.rows)) {
    LOG(FATAL) << "Image and Mask size not matched: " << img_path << ", "
               << " image size: " << img.cols << "x" << img.rows << ", "
               << " mask size: " << mask_miss.cols << "x" << mask_miss.rows;
  }
  cv::Mat mask_all;
  if (param_.mode() == 6) {
    // string& mask_all_path = meta.mask_all_path;
    // mask_all = cv::imread(mask_all_path, CV_LOAD_IMAGE_GRAYSCALE);
    LOG(FATAL) << "Error - mode must be 5 in this version.";
  } else {
    mask_all = Mat::zeros(img.rows, img.cols, CV_8UC1);
  }
  // 完成转换
  Transform_nv(img,mask_miss,mask_all,meta,transformed_data,transformed_vec_mask,
               transformed_heat_mask,transformed_vecmap,transformed_heatmap);
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::Transform_nv_out(const string& xml_file,
                                        Dtype* transformed_data,
                                        Dtype* transformed_vec_mask,
                                        Dtype* transformed_heat_mask,
                                        Dtype* transformed_vecmap,
                                        Dtype* transformed_heatmap,
                                        Dtype* transformed_kps) {
  MetaData meta;
  // 读入参数信息
  if (!ReadMetaDataFromXml(xml_file, param_.root_dir(), meta)) {
    LOG(FATAL) << "Error found in reading from: " << xml_file;
  }
  // 读入图片和mask
  string& img_path = meta.img_path;
  string& mask_miss_path = meta.mask_miss_path;
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
  cv::Mat mask_miss = cv::imread(mask_miss_path, CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data) {
    LOG(FATAL) << "Open error: " << img_path;
  }
  if (!mask_miss.data) {
    LOG(FATAL) << "Open error: " << mask_miss_path;
  }
  if ((img.cols != mask_miss.cols) || (img.rows != mask_miss.rows)) {
    LOG(FATAL) << "Image and Mask size not matched: " << img_path << ", "
               << " image size: " << img.cols << "x" << img.rows << ", "
               << " mask size: " << mask_miss.cols << "x" << mask_miss.rows;
  }
  cv::Mat mask_all;
  if (param_.mode() == 6) {
    // string& mask_all_path = meta.mask_all_path;
    // mask_all = cv::imread(mask_all_path, CV_LOAD_IMAGE_GRAYSCALE);
    LOG(FATAL) << "Error - mode must be 5 in this version.";
  } else {
    mask_all = Mat::zeros(img.rows, img.cols, CV_8UC1);
  }
  // 完成转换
  Transform_nv(img,mask_miss,mask_all,meta,transformed_data,transformed_vec_mask,
               transformed_heat_mask,transformed_vecmap,transformed_heatmap);
  // 输出kp
  Output_keypoints(meta, transformed_kps);
}

// 转换调用
template<typename Dtype>
void PoseDataTransformer<Dtype>::Transform_nv(cv::Mat& img,
                                              cv::Mat& mask_miss,
                                              cv::Mat& mask_all,
                                              MetaData& meta,
                                              Dtype* transformed_data,
                                              Dtype* transformed_vec_mask,
                                              Dtype* transformed_heat_mask,
                                              Dtype* transformed_vecmap,
                                              Dtype* transformed_heatmap) {
  // 数据增广参数结果
  AugmentSelection as = {
    false,
    0.0,
    Size(),
    0,
  };
  CPUTimer process_timer;
  int mode = param_.mode();
  CHECK_EQ(mode,5) << "mode must be 5 in current version.";
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();
  int train_resized_width = param_.train_resized_width();
  int train_resized_height = param_.train_resized_height();
  int offset = img.rows * img.cols;

  int stride = param_.stride();

  // 转换joints from 17 -> 18 (COCO)
  // from 16 -> 18 (MPII)
  if(param_.transform_body_joint()) {
    TransformMetaJoints(meta);
  }

  //Start transforming
  Mat img_aug;
  if (param_.crop_using_resize()) {
    img_aug = Mat::zeros(train_resized_height, train_resized_width, CV_8UC3);
  } else {
    img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  }
  Mat mask_miss_aug, mask_all_aug ;
  Mat img_temp, img_temp2, img_temp3;
  if (phase_ == TRAIN) {
    //0: disortion, note: disortion does not affect the masks and meta data
    if (param_.has_dis_param()) {
      process_timer.Start();
      img = DistortImage(img,param_.dis_param());
      process_timer.Stop();
      LOG(INFO) << "   DistortImage: " << process_timer.MilliSeconds() << " ms.";
    }
    //1: scale
    try {
      process_timer.Start();
      as.scale = augmentation_scale(img, img_temp, mask_miss, mask_all, meta, mode);
      LOG(INFO) << "   augmentation_scale: " << process_timer.MilliSeconds() << " ms.";
    } catch (exception& e) {
      LOG(FATAL) << "Scale error: " << meta.img_path << ", "
                << "image: " << img.cols << "x" << img.rows
                << ", image_temp: " << img_temp.cols << "x" << img_temp.rows
                << ", mask: " << mask_miss.cols << "x" << mask_miss.rows;
    }
    //2: rotate
    try {
      process_timer.Start();
      as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, mask_all, meta, mode);
      LOG(INFO) << "   augmentation_rotate: " << process_timer.MilliSeconds() << " ms.";
    } catch (exception& e) {
      LOG(FATAL) << "Rotate error: " << meta.img_path << ", "
                << "image_temp: " << img_temp.cols << "x" << img_temp.rows
                << ", image_temp2: " << img_temp2.cols << "x" << img_temp2.rows
                << ", mask: " << mask_miss.cols << "x" << mask_miss.rows;
    }
    //3: crop
    try {
      process_timer.Start();
      as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, mask_all, mask_all_aug, meta, mode);
      LOG(INFO) << "   augmentation_croppad: " << process_timer.MilliSeconds() << " ms.";
    } catch (exception& e) {
      LOG(FATAL) << "Crop error: " << meta.img_path << ", "
                << "image_temp2: " << img_temp2.cols << "x" << img_temp2.rows
                << ", image_temp3: " << img_temp3.cols << "x" << img_temp3.rows
                << ", mask: " << mask_miss.cols << "x" << mask_miss.rows
                << ", mask_aug: " << mask_miss_aug.cols << "x" << mask_miss_aug.rows;
    }
    //4: flip
    try {
      as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, mask_all_aug, meta, mode);
    } catch (exception& e) {
      LOG(FATAL) << "Flip error: " << meta.img_path << ", "
                << "image_temp3: " << img_temp3.cols << "x" << img_temp3.rows
                << ", image_aug: " << img_aug.cols << "x" << img_aug.rows
                << ", mask_aug: " << mask_miss_aug.cols << "x" << mask_miss_aug.rows;
    }
    if(param_.visualize()) {
      visualize(img_aug, meta);
    }
    // 5: Mask-miss/all resize -> 1/stride
    if (mode > 4) {
      resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
    if (mode > 5) {
      LOG(FATAL) << "Error - mode must be 5 in this version.";
      // resize(mask_all_aug, mask_all_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
  } else {
    // resize
    int resized_width = param_.resized_width();
    int resized_height = param_.resized_height();
    CHECK_GT(resized_height, 0);
    CHECK_GT(resized_width, 0);
    resize(img,img_aug,Size(resized_width,resized_height),INTER_CUBIC);
    if (mode > 4) {
      resize(mask_miss,mask_miss_aug,Size(resized_width,resized_height),INTER_CUBIC);
    }
    if (mode > 5) {
      LOG(FATAL) << "Error - mode must be 5 in this version.";
      // resize(mask_all,mask_all_aug,Size(resized_width,resized_height),INTER_CUBIC);
    }
    // MetaData
    float scale_x = (float)resized_width / img.cols;
    float scale_y = (float)resized_height / img.rows;
    meta.objpos.x *= scale_x;
    meta.objpos.y *= scale_y;
    meta.area *= (scale_x * scale_y);
    for (int i = 0; i < 18; i++){
      meta.joint_self.joints[i].x *= scale_x;
      meta.joint_self.joints[i].y *= scale_y;
    }
    for(int p = 0; p < meta.numOtherPeople; p++){
      meta.objpos_other[p].x *= scale_x;
      meta.objpos_other[p].y *= scale_y;
      meta.area_other[p] *= (scale_x * scale_y);
      for(int i = 0; i < 18; i++){
        meta.joint_others[p].joints[i].x *= scale_x;
        meta.joint_others[p].joints[i].y *= scale_y;
      }
    }
    if(param_.visualize()) {
      visualize(img_aug, meta);
    }
    if (mode > 4) {
      resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
    if (mode > 5) {
      LOG(FATAL) << "Error - mode must be 5 in this version.";
      // resize(mask_all_aug, mask_all_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
  }
  // 图像数据减去均值, 标准为[-0.5~0.5]
  offset = img_aug.rows * img_aug.cols;
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  // data
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      if (param_.normalize()) {
        transformed_data[i * img_aug.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[offset + i * img_aug.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2 * offset + i * img_aug.cols + j] = (rgb[2] - 128)/256.0;
      } else {
        // we use means to substract
        CHECK_EQ(param_.mean_value_size(), 3) << "Must provide mean_values, and mean_value should have length of 3.";
        transformed_data[i * img_aug.cols + j] = rgb[0] - param_.mean_value(0);
        transformed_data[offset + i * img_aug.cols + j] = rgb[1] - param_.mean_value(1);
        transformed_data[2 * offset + i * img_aug.cols + j] = rgb[2] - param_.mean_value(2);
      }
    }
  }
  // 生成mask
  for (int g_y = 0; g_y < grid_y; g_y++) {
    for (int g_x = 0; g_x < grid_x; g_x++) {
      float mask = float(mask_miss_aug.at<uchar>(g_y, g_x)) / 255;
      // heatmask
      for (int i = 0; i < 18; i++) {
        if (np_ == 17) {
          // COCO
          transformed_heat_mask[i * channelOffset + g_y * grid_x + g_x] = mask;
        } else if (np_ == 16) {
          // MPII
          // 0/14-17 -> 0
          if ((i == 0) || (i > 13)) {
            transformed_heat_mask[i * channelOffset + g_y * grid_x + g_x] = 0;
          } else {
            transformed_heat_mask[i * channelOffset + g_y * grid_x + g_x] = mask;
          }
        } else {
          LOG(FATAL) << "Unknown np_ : " << np_;
        }
      }
      // vecmask
      for (int i = 0; i < 17 * 2; ++i) {
        if (np_ == 17) {
          // COCO
          transformed_vec_mask[i * channelOffset + g_y * grid_x + g_x] = mask;
        } else if (np_ == 16) {
          // MPII
          if (i > 23) {
            transformed_vec_mask[i * channelOffset + g_y * grid_x + g_x] = 0;
          } else {
            transformed_vec_mask[i * channelOffset + g_y * grid_x + g_x] = mask;
          }
        } else {
          LOG(FATAL) << "Unknown np_ : " << np_;
        }
      }
    }
  }
  // 生成label-map
  generateLabelMap(transformed_vecmap, transformed_heatmap, img_aug, meta);
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::Output_keypoints(MetaData& meta, Dtype* transformed_kps) {
  const int stride = param_.stride();
  int img_w, img_h;
  if (phase_ == TRAIN) {
    img_w = param_.crop_size_x();
    img_h = param_.crop_size_y();
  } else {
    img_w = param_.resized_width();
    img_h = param_.resized_height();
  }
  const int grid_x = img_w / stride;
  const int grid_y = img_h / stride;
  const int channelOffset = grid_y * grid_x;
  CHECK_GE(channelOffset, (18 * 3 + 1) * 10 + 1);
  int total_people = 1 + meta.numOtherPeople;
  if (total_people >= 40) total_people = 40;
  // 首先输出main people -> c == 0
  transformed_kps[0] = total_people;
  transformed_kps[1] = meta.area;
  int idx = 2;
  for (int k = 0; k < 18; ++k) {
    float px = meta.joint_self.joints[k].x;
    float py = meta.joint_self.joints[k].y;
    int vis = int(meta.joint_self.isVisible[k]);
    if (vis <= 1 && (px >= 0) && (py >= 0) && (px < img_w) && (py < img_h)) {
      transformed_kps[idx++] = px;
      transformed_kps[idx++] = py;
      transformed_kps[idx++] = vis;
    } else {
      transformed_kps[idx++] = px;
      transformed_kps[idx++] = py;
      transformed_kps[idx++] = 2;
    }
  }
  // 输出后面的通道
  int p_count = 1;
  for (int p = 0; p < meta.numOtherPeople; ++p) {
    int c = (p + 1)/10;
    // 输出通道头部
    if (c > 0 && ((p+1) % 10 == 0)) {
      transformed_kps[c * channelOffset] = total_people;
    }
    // 输出person的area
    int idx = c * channelOffset + 1 + (18*3 + 1) * ((p+1) % 10);
    transformed_kps[idx++] = meta.area_other[p];
    for (int k = 0; k < 18; ++k) {
      float px = meta.joint_others[p].joints[k].x;
      float py = meta.joint_others[p].joints[k].y;
      int vis = int(meta.joint_others[p].isVisible[k]);
      if (vis <= 1 && (px >= 0) && (py >= 0) && (px < img_w) && (py < img_h)) {
        transformed_kps[idx++] = px;
        transformed_kps[idx++] = py;
        transformed_kps[idx++] = vis;
      } else {
        transformed_kps[idx++] = px;
        transformed_kps[idx++] = py;
        transformed_kps[idx++] = 2;
      }
    }
    p_count++;
    if (p_count >= total_people) break;
  }
}

// include mask_miss
template<typename Dtype>
float PoseDataTransformer<Dtype>::augmentation_scale(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  float scale_multiplier;
  if(dice > param_.scale_prob()) {
    img_aug = img.clone();
    scale_multiplier = 1;
  } else {
    float dice2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min();
  }
  float scale_abs = param_.target_dist() / meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  resize(img, img_aug, Size(), scale, scale, INTER_CUBIC);
  if (mode > 4) {
    resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);
  }
  if (mode > 5) {
    LOG(FATAL) << "Error - mode must be 5 in this version.";
    // resize(mask_all, mask_all, Size(), scale, scale, INTER_CUBIC);
  }
  //modify meta data
  meta.objpos *= scale;
  for (int i = 0; i < 18; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p = 0; p < meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i = 0; i < 18; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

// include masks
template<typename Dtype>
Size PoseDataTransformer<Dtype>::augmentation_croppad(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta, int mode) {
  float dice_x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  float dice_y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  const bool crop_resize_method = param_.crop_using_resize();
  // random aspect
  if (crop_resize_method) {
    float dice_as = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
    float as = (param_.crop_as_max() - param_.crop_as_min()) * dice_as + param_.crop_as_min();
    crop_x = (int)((float)crop_x * sqrt(as));
    crop_y = (int)((float)crop_y / sqrt(as));
  }

  // center offs
  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  // new center
  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));

  img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  // MISS:填充部分默认是255
  mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(255);
  // ALL: 填充部分默认是0
  mask_all_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
  // 开始裁剪图像和MASK
  for(int i = 0; i < crop_y; i++){
    for(int j = 0; j < crop_x; j++){
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img.cols, img.rows))) {
        img_aug.at<Vec3b>(i,j) = img.at<Vec3b>(coord_y_on_img, coord_x_on_img);
        if (mode > 4) {
          mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
        if (mode > 5) {
          LOG(FATAL) << "Error - mode must be 5 in this version.";
          // mask_all_aug.at<uchar>(i,j) = mask_all.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
      }
    }
  }
  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i = 0; i < 18; i++) {
    meta.joint_self.joints[i] += offset;
  }
  for(int p = 0; p < meta.numOtherPeople; p++) {
    meta.objpos_other[p] += offset;
    for(int i = 0; i < 18; i++) {
      meta.joint_others[p].joints[i] += offset;
    }
  }
  // if resize, then do it
  if (crop_resize_method) {
    int resized_width = param_.train_resized_width();
    int resized_height = param_.train_resized_height();
    float scale_x = (float)resized_width / img_aug.cols;
    float scale_y = (float)resized_height / img_aug.rows;
    resize(img_aug, img_aug, Size(resized_width,resized_height), INTER_CUBIC);
    resize(mask_miss_aug, mask_miss_aug, Size(resized_width,resized_height), INTER_CUBIC);
    // point
    meta.objpos.x *= scale_x;
    meta.objpos.y *= scale_y;
    meta.area *= (scale_x * scale_y);
    for (int i = 0; i < 18; i++){
      meta.joint_self.joints[i].x *= scale_x;
      meta.joint_self.joints[i].y *= scale_y;
    }
    for(int p = 0; p < meta.numOtherPeople; p++){
      meta.objpos_other[p].x *= scale_x;
      meta.objpos_other[p].y *= scale_y;
      meta.area_other[p] *= (scale_x * scale_y);
      for(int i = 0; i < 18; i++){
        meta.joint_others[p].joints[i].x *= scale_x;
        meta.joint_others[p].joints[i].y *= scale_y;
      }
    }
  }
  // return
  return Size(x_offset, y_offset);
}

// include masks
template<typename Dtype>
bool PoseDataTransformer<Dtype>::augmentation_flip(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  bool doflip = (dice <= param_.flip_prob());
  if(doflip) {
    flip(img, img_aug, 1);
    int w = img.cols;
    if(mode > 4) {
      flip(mask_miss, mask_miss, 1);
    }
    if(mode > 5) {
      LOG(FATAL) << "Error - mode must be 5 in this version.";
      // flip(mask_all, mask_all, 1);
    }
    // flip pos
    meta.objpos.x = w - 1 - meta.objpos.x;
    // flip joints
    for(int i = 0; i < 18; i++) {
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    // 左右交换
    if(param_.transform_body_joint()) {
      swapLeftRight(meta.joint_self);
    }
    // nop交换
    for(int p = 0; p < meta.numOtherPeople; p++) {
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i = 0; i < 18; i++) {
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint()) {
        swapLeftRight(meta.joint_others[p]);
      }
    }
  } else {
    img_aug = img.clone();
  }
  return doflip;
}

// include masks
template<typename Dtype>
float PoseDataTransformer<Dtype>::augmentation_rotate(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  float degree = (dice - 0.5) * 2 * param_.max_rotate_degree();

  Point2f center(img.cols/2.0, img.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img.size(), degree).boundingRect();
  // 输出坐标矫正到0,0
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //多余的部分全部用128/128/128填充
  warpAffine(img, img_aug, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  if(mode > 4) {
    // mask-miss填充, 多余的部分使用255
    warpAffine(mask_miss, mask_miss, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255));
  }
  if(mode > 5) {
    // mask-all使用0填充
    LOG(FATAL) << "Error - mode must be 5 in this version.";
    // warpAffine(mask_all, mask_all, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
  }
  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i = 0; i < 18; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p = 0; p < meta.numOtherPeople; p++) {
    RotatePoint(meta.objpos_other[p], R);
    for(int i = 0; i < 18; i++) {
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}

// no mask
template<typename Dtype>
float PoseDataTransformer<Dtype>::augmentation_scale(Mat& img, Mat& img_aug, MetaData& meta) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  float scale_multiplier;
  if(dice > param_.scale_prob()) {
    img_aug = img.clone();
    scale_multiplier = 1;
  } else {
    float dice2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min();
  }
  // 根据person的scale对乘积因子进行调节
  float scale_abs = param_.target_dist() / meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  resize(img, img_aug, Size(), scale, scale, INTER_CUBIC);
  //modify meta data
  meta.objpos *= scale;
  for(int i = 0; i < 18; i++) {
    meta.joint_self.joints[i] *= scale;
  }
  for(int p = 0; p < meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i = 0; i < 18; i++) {
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  // 返回随机因子
  return scale_multiplier;
}

template<typename Dtype>
bool PoseDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

// no masks
template<typename Dtype>
Size PoseDataTransformer<Dtype>::augmentation_croppad(Mat& img, Mat& img_aug, MetaData& meta) {
  float dice_x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  float dice_y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));

  img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  for(int i = 0; i < crop_y; i++) {
    for(int j = 0; j < crop_x; j++) {
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img.cols, img.rows))) {
        img_aug.at<Vec3b>(i,j) = img.at<Vec3b>(coord_y_on_img, coord_x_on_img);
      }
    }
  }
  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i = 0; i < 18; i++) {
    meta.joint_self.joints[i] += offset;
  }
  for(int p = 0; p < meta.numOtherPeople; p++) {
    meta.objpos_other[p] += offset;
    for(int i = 0; i < 18; i++) {
      meta.joint_others[p].joints[i] += offset;
    }
  }
  return Size(x_offset, y_offset);
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::swapLeftRight(Joints& j) {
  int right[8] = {3,4,5, 9,10,11,15,17};
  int left[8] =  {6,7,8,12,13,14,16,18};
  for(int i = 0; i < 8; i++) {
    int ri = right[i] - 1;
    int li = left[i] - 1;
    Point2f temp = j.joints[ri];
    j.joints[ri] = j.joints[li];
    j.joints[li] = temp;
    int temp_v = j.isVisible[ri];
    j.isVisible[ri] = j.isVisible[li];
    j.isVisible[li] = temp_v;
  }
}

// no masks
template<typename Dtype>
bool PoseDataTransformer<Dtype>::augmentation_flip(Mat& img, Mat& img_aug, MetaData& meta) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  bool doflip = (dice <= param_.flip_prob());
  if(doflip) {
    flip(img, img_aug, 1);
    int w = img.cols;
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i = 0; i < 18; i++) {
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);
    for(int p = 0; p < meta.numOtherPeople; p++) {
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i = 0; i < 18; i++) {
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  } else {
    img_aug = img.clone();
  }
  return doflip;
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat& R){
  Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

// no masks
template<typename Dtype>
float PoseDataTransformer<Dtype>::augmentation_rotate(Mat& img, Mat& img_aug, MetaData& meta) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  float degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  Point2f center(img.cols/2.0, img.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img.size(), degree).boundingRect();
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  warpAffine(img, img_aug, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  RotatePoint(meta.objpos, R);
  for(int i = 0; i < 18; i++) {
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p = 0; p < meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i = 0; i < 18; i++) {
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}

// 生成GaussMap
template<typename Dtype>
void PoseDataTransformer<Dtype>::putGaussianMaps(Dtype* map, Point2f center, int stride, int grid_x, int grid_y, float sigma) {
  float start = stride / 2.0 - 0.5;
  for (int g_y = 0; g_y < grid_y; g_y++) {
    for (int g_x = 0; g_x < grid_x; g_x++) {
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      // 0.01
      if(exponent > 4.6052) {
        continue;
      }
      map[g_y * grid_x + g_x] += exp(-exponent);
      if(map[g_y * grid_x + g_x] > 1) {
         map[g_y * grid_x + g_x] = 1;
      }
    }
  }
}

// 生成VecPeaks (MidPoint)
template<typename Dtype>
void PoseDataTransformer<Dtype>::putVecPeaks(Dtype* vecX, Dtype* vecY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  centerB = centerB / stride;
  centerA = centerA / stride;
  Point2f bc = centerB - centerA;
  float norm_bc = sqrt(bc.x * bc.x + bc.y * bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;
  for(int j = 0; j < 3; j++) {
    // center分别为A/M/B
    Point2f center = centerB * 0.5 * j + centerA * 0.5 * (2 - j);
    // 在center附近选择一段区域
    int min_x = std::max( int(floor(center.x - thre)), 0);
    int max_x = std::min( int( ceil(center.x + thre)), grid_x);
    int min_y = std::max( int(floor(center.y - thre)), 0);
    int max_y = std::min( int( ceil(center.y + thre)), grid_y);
    // 遍历该区域
    for (int g_y = min_y; g_y < max_y; g_y++) {
      for (int g_x = min_x; g_x < max_x; g_x++) {
        float dist = (g_x - center.x) * (g_x - center.x) + (g_y - center.y) * (g_y - center.y);
        if (dist <= thre) {
          int cnt = count.at<uchar>(g_y, g_x);
          if (cnt == 0){
            vecX[g_y * grid_x + g_x] = bc.x;
            vecY[g_y * grid_x + g_x] = bc.y;
          } else {
            vecX[g_y * grid_x + g_x] = (vecX[g_y * grid_x + g_x] * cnt + bc.x) / (cnt + 1);
            vecY[g_y * grid_x + g_x] = (vecY[g_y * grid_x + g_x] * cnt + bc.y) / (cnt + 1);
            count.at<uchar>(g_y, g_x) = cnt + 1;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::putVecMaps(Dtype* vecX, Dtype* vecY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre) {
  centerB = centerB / stride;
  centerA = centerA / stride;
  Point2f bc = centerB - centerA;
  // 选取线段所在的区域
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x) - thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x) + thre)), grid_x);
  int min_y = std::max( int(round(std::min(centerA.y, centerB.y) - thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y) + thre)), grid_y);
  // 计算法线方向
  float norm_bc = sqrt(bc.x * bc.x + bc.y * bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;
  // 遍历该区域
  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      // 垂直方向
      float dist = std::abs(ba.x * bc.y - ba.y * bc.x);
      // 如果小于阈值,则表明是线段区域
      // 注意count矩阵, 求取平均值
      if(dist <= thre) {
        int cnt = count.at<uchar>(g_y, g_x);
        if (cnt == 0){
          vecX[g_y * grid_x + g_x] = bc.x;
          vecY[g_y * grid_x + g_x] = bc.y;
        } else {
          vecX[g_y * grid_x + g_x] = (vecX[g_y * grid_x + g_x] * cnt + bc.x) / (cnt + 1);
          vecY[g_y * grid_x + g_x] = (vecY[g_y * grid_x + g_x] * cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
      }
    }
  }
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_vecmap, Dtype* transformed_heatmap,
                                                  cv::Mat& img_aug, MetaData& meta) {
  int stride = param_.stride();
  int grid_x = img_aug.cols / stride;
  int grid_y = img_aug.rows / stride;
  int channelOffset = grid_y * grid_x;
  // int mode = param_.mode();
  // init for maps
  for (int g_y = 0; g_y < grid_y; g_y++) {
    for (int g_x = 0; g_x < grid_x; g_x++) {
      for (int i = 0; i < 17 * 2; i++) {
        transformed_vecmap[i * channelOffset + g_y * grid_x + g_x] = 0;
      }
      for (int i = 0; i < 18; i++) {
        transformed_heatmap[i * channelOffset + g_y * grid_x + g_x] = 0;
      }
    }
  }
  // Maps
  // heatmap
  for (int i = 0; i < 18; i++) {
    Point2f center = meta.joint_self.joints[i];
    if(meta.joint_self.isVisible[i] <= 1) {
      putGaussianMaps(transformed_heatmap + i * channelOffset, center, param_.stride(),
                      grid_x, grid_y, param_.sigma());
    }
    for(int j = 0; j < meta.numOtherPeople; j++){
      Point2f center = meta.joint_others[j].joints[i];
      if(meta.joint_others[j].isVisible[i] <= 1) {
        putGaussianMaps(transformed_heatmap + i * channelOffset, center, param_.stride(),
                        grid_x, grid_y, param_.sigma());
      }
    }
  }
  // limbs
  int mid_1[17] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 2, 6, 7, 2, 1,  1,  15, 16};
  int mid_2[17] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8, 1, 15, 16, 17, 18};
  int thre = 1;
  for(int i = 0; i < 17; i++) {
    Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
    Joints jo = meta.joint_self;
    if(jo.isVisible[mid_1[i]-1] <= 1 && jo.isVisible[mid_2[i]-1] <= 1) {
      putVecMaps(transformed_vecmap + 2 * i * channelOffset, transformed_vecmap + (2 * i + 1) * channelOffset,
                count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre);
    }
    for(int j = 0; j < meta.numOtherPeople; j++) {
      Joints jo2 = meta.joint_others[j];
      if(jo2.isVisible[mid_1[i]-1] <= 1 && jo2.isVisible[mid_2[i]-1] <= 1) {
        putVecMaps(transformed_vecmap + 2 * i * channelOffset, transformed_vecmap + (2 * i + 1) * channelOffset,
                count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre);
      }
    }
  }
  //显示
  if(1 && param_.visualize()) {
    Mat label_map;
    // 显示每个通道
    for(int i = 0; i < 18; i++){
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      for (int g_y = 0; g_y < grid_y; g_y++) {
        for (int g_x = 0; g_x < grid_x; g_x++) {
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_heatmap[i * channelOffset + g_y * grid_x + g_x] * 255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
      char imagename [256];
      sprintf(imagename, "%saugment_%06d_label_part_%02d.jpg", param_.save_dir().c_str(), meta.annolist_index, i);
      imwrite(imagename, label_map);
    }
  }
}

// 显示函数
template<typename Dtype>
void PoseDataTransformer<Dtype>::visualize(Mat& img, MetaData& meta) {
  Mat img_vis = img.clone();
  static int counter = 0;
  // 绘制目标中心
  rectangle(img_vis, meta.objpos-Point2f(3,3), meta.objpos+Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
  //关节点绘制circle
  for(int i = 0; i < 18; i++) {
    if(meta.joint_self.isVisible[i] <= 1)
      circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(200,200,255), -1);
  }
  //绘制op的中心点
  for(int p = 0; p < meta.numOtherPeople; p++) {
    rectangle(img_vis, meta.objpos_other[p]-Point2f(3,3), meta.objpos_other[p]+Point2f(3,3), CV_RGB(0,255,255), CV_FILLED);
    //绘制op的关节点
    for(int i = 0; i < 18; i++) {
      if(meta.joint_others[p].isVisible[i] <= 1)
        circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,0,255), -1);
    }
  }
  char imagename [256];
  sprintf(imagename, "%saugment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add.";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void PoseDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> PoseDataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void PoseDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() || (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int PoseDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(PoseDataTransformer);

}  // namespace caffe
