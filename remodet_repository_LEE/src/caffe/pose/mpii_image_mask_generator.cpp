#include "caffe/pose/mpii_image_mask_generator.hpp"
#include "caffe/tracker/basic.hpp"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

namespace caffe {

using std::string;
using std::vector;
namespace bfs = boost::filesystem;
using namespace boost::property_tree;

template <typename Dtype>
MpiiMaskGenerator<Dtype>::MpiiMaskGenerator(
                  const std::string& box_xml_path,
                  const std::string& kps_xml_path,
                  const std::string& image_folder,
                  const std::string& output_folder,
                  const int xml_idx):
  box_xml_path_(box_xml_path), kps_xml_path_(kps_xml_path), image_folder_(image_folder), output_folder_(output_folder), xml_idx_(xml_idx) {
}

// save_image -> 是否要保存输出图像
// show_box -> 是否绘制box
template <typename Dtype>
int MpiiMaskGenerator<Dtype>::Generate(const bool save_image, const bool show_box) {
  // 读入boxes信息
  vector<BoundingBox<Dtype> > bboxes;
  bool status;
  status = ReadBoxes(box_xml_path_, &bboxes);
  // LOG(INFO) << "Step 1";
  if (!status) {
    LOG(WARNING) << "bounding box invalid in file: " << box_xml_path_ << ", skip this image.";
    return xml_idx_;
  }
  if (bboxes.size() == 0) {
    LOG(WARNING) << "no boxes found in " << box_xml_path_ << ", skip this image.";
    return xml_idx_;
  }
  // 读入kps信息
  MetaDataAll<Dtype> kps;
  ReadKps(kps_xml_path_, &kps);
  // LOG(INFO) << "Step 2";
  // 完成匹配
  vector<bool> matched;
  status = Match(bboxes, &kps, &matched);
  if (!status) {
    LOG(WARNING) << "found unmatched kps vs. boxes in file: " << box_xml_path_ << ", skip this image.";
    return xml_idx_;
  }
  // LOG(INFO) << "Step 3";
  // 生成mask：多余的box全部mask
  cv::Mat mask = cv::Mat::zeros(32,32,CV_8UC1);
  // GenMask(kps, bboxes, matched, &mask);
  // 保存mask：按照图片的名称命名
  // SaveMask(kps, mask);
  // 生成多个样本
  vector<MetaData<Dtype> > metas;
  Split(kps, mask, &metas);
  // 保存xml：按照xml_idx的索引依次编号
  int ret = SaveXml(metas);
  // 保存图像
  // if (save_image) {
  //   SaveImage(kps, mask, show_box);
  // }
  return ret;
}

template <typename Dtype>
bool MpiiMaskGenerator<Dtype>::ReadBoxes(const std::string& box_xml_path, vector<BoundingBox<Dtype> >* bboxes) {
  bboxes->clear();
  ptree pt;
  read_xml(box_xml_path, pt);
  int width, height;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
  } catch (const ptree_error &e) {
    LOG(FATAL) << "When parsing " << box_xml_path << ": " << e.what();
  }
  // scan over all
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    if (v1.first == "object") {
      ptree object = v1.second;
      // scan over all within the object
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        if (v2.first == "bndbox") {
          ptree pt2 = v2.second;
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          if (xmin > width || ymin > height || xmax > width || ymax > height
              || xmin < 0 || ymin < 0 || xmax < 0 || ymax < 0 || xmin >= xmax || ymin >= ymax) {
            return false;
          }
          BoundingBox<Dtype> box;
          box.x1_ = xmin;
          box.y1_ = ymin;
          box.x2_ = xmax;
          box.y2_ = ymax;
          bboxes->push_back(box);
        }
      }
    }
  }
  return true;
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::ReadKps(const std::string& kps_xml_path, MetaDataAll<Dtype>* kps) {
  ptree pt;
  read_xml(kps_xml_path, pt);
  kps->img_path = pt.get<string>("Annotations.ImagePath");
  kps->dataset = pt.get<string>("Annotations.MetaData.dataset");
  kps->isValidation = (pt.get<int>("Annotations.MetaData.isValidation") == 0 ? false : true);
  int width = pt.get<int>("Annotations.MetaData.width");
  int height = pt.get<int>("Annotations.MetaData.height");
  kps->img_size = Size(width,height);
  kps->image_idx = pt.get<int>("Annotations.MetaData.annolist_index");
  kps->numPeople = 1 + pt.get<int>("Annotations.MetaData.numOtherPeople");
  //----------------------------------------------------------------------------
  MetaData<Dtype> meta;
  meta.img_size = Size(width,height);
  meta.objpos.x = pt.get<Dtype>("Annotations.MetaData.objpos.center_x");
  meta.objpos.y = pt.get<Dtype>("Annotations.MetaData.objpos.center_y");
  meta.scale_self = pt.get<Dtype>("Annotations.MetaData.scale");
  meta.area = pt.get<Dtype>("Annotations.MetaData.area");
  // box
  meta.bbox.x1_ = pt.get<Dtype>("Annotations.MetaData.bbox.xmin");
  meta.bbox.y1_ = pt.get<Dtype>("Annotations.MetaData.bbox.ymin");
  meta.bbox.x2_ = meta.bbox.x1_ + pt.get<Dtype>("Annotations.MetaData.bbox.width");
  meta.bbox.y2_ = meta.bbox.y1_ + pt.get<Dtype>("Annotations.MetaData.bbox.height");
  // joints
  meta.joint_self.joints.resize(16);
  meta.joint_self.isVisible.resize(16);
  for(int i = 0; i < 16; ++i) {
    char temp_x[256], temp_y[256], temp_vis[256];
    sprintf(temp_x, "Annotations.MetaData.joint_self.kp_%d.x", i+1);
    sprintf(temp_y, "Annotations.MetaData.joint_self.kp_%d.y", i+1);
    sprintf(temp_vis, "Annotations.MetaData.joint_self.kp_%d.vis", i+1);
    meta.joint_self.joints[i].x = pt.get<Dtype>(temp_x);
    meta.joint_self.joints[i].y = pt.get<Dtype>(temp_y);
    int isVisible = pt.get<int>(temp_vis);
    meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
       meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height) {
      meta.joint_self.isVisible[i] = 2;
    }
  }
  // other people
  meta.objpos_other.clear();
  meta.scale_other.clear();
  meta.joint_others.clear();
  meta.area_other.clear();
  meta.bbox_others.clear();
  if (kps->numPeople > 1) {
    meta.objpos_other.resize(kps->numPeople - 1);
    meta.scale_other.resize(kps->numPeople - 1);
    meta.joint_others.resize(kps->numPeople - 1);
    meta.area_other.resize(kps->numPeople - 1);
    meta.bbox_others.resize(kps->numPeople - 1);
    for(int p = 0; p < kps->numPeople - 1; p++){
      // ojbpos & scale
      char temp_x[256], temp_y[256], temp_scale[256], temp_area[256];
      char temp_xmin[256], temp_ymin[256], temp_width[256], temp_height[256];
      sprintf(temp_x, "Annotations.MetaData.objpos_other.objpos_%d.center_x", p+1);
      sprintf(temp_y, "Annotations.MetaData.objpos_other.objpos_%d.center_y", p+1);
      sprintf(temp_scale, "Annotations.MetaData.scale_other.scale_%d", p+1);
      sprintf(temp_area, "Annotations.MetaData.area_other.area_%d", p+1);
      sprintf(temp_xmin, "Annotations.MetaData.bbox_other.bbox_%d.xmin", p+1);
      sprintf(temp_ymin, "Annotations.MetaData.bbox_other.bbox_%d.ymin", p+1);
      sprintf(temp_width, "Annotations.MetaData.bbox_other.bbox_%d.width", p+1);
      sprintf(temp_height, "Annotations.MetaData.bbox_other.bbox_%d.height", p+1);
      meta.objpos_other[p].x = pt.get<Dtype>(temp_x);
      meta.objpos_other[p].y = pt.get<Dtype>(temp_y);
      meta.scale_other[p] = pt.get<Dtype>(temp_scale);
      meta.area_other[p] = pt.get<Dtype>(temp_area);
      meta.bbox_others[p].x1_ = pt.get<Dtype>(temp_xmin);
      meta.bbox_others[p].y1_ = pt.get<Dtype>(temp_ymin);
      meta.bbox_others[p].x2_ = meta.bbox_others[p].x1_ + pt.get<Dtype>(temp_width);
      meta.bbox_others[p].y2_ = meta.bbox_others[p].y1_ + pt.get<Dtype>(temp_height);
      // joints
      meta.joint_others[p].joints.resize(16);
      meta.joint_others[p].isVisible.resize(16);
      for (int i = 0; i < 16; i++) {
        char joint_x[256], joint_y[256], joint_vis[256];
        sprintf(joint_x, "Annotations.MetaData.joint_others.joint_%d.kp_%d.x", p+1, i+1);
        sprintf(joint_y, "Annotations.MetaData.joint_others.joint_%d.kp_%d.y", p+1, i+1);
        sprintf(joint_vis, "Annotations.MetaData.joint_others.joint_%d.kp_%d.vis", p+1, i+1);
        meta.joint_others[p].joints[i].x = pt.get<Dtype>(joint_x);
        meta.joint_others[p].joints[i].y = pt.get<Dtype>(joint_y);
        int isVisible = pt.get<int>(joint_vis);
        meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
        if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
           meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
          meta.joint_others[p].isVisible[i] = 2;
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  // push first one
  kps->objpos.push_back(meta.objpos);
  kps->scale.push_back(meta.scale_self);
  kps->area.push_back(meta.area);
  kps->bbox.push_back(meta.bbox);
  kps->joint.push_back(meta.joint_self);
  for (int p = 0; p < kps->numPeople - 1; ++p) {
    kps->objpos.push_back(meta.objpos_other[p]);
    kps->scale.push_back(meta.scale_other[p]);
    kps->area.push_back(meta.area_other[p]);
    kps->bbox.push_back(meta.bbox_others[p]);
    kps->joint.push_back(meta.joint_others[p]);
  }
  CHECK_EQ(kps->objpos.size(), kps->numPeople);
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::LoadImage(const std::string& image_path, cv::Mat* image) {
  const std::string image_file = image_folder_ + '/' + image_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
}

template <typename Dtype>
int MpiiMaskGenerator<Dtype>::GetMaskPoints(const Joints& joint, const cv::Mat& mask){
  int num = 0;
  int width = mask.cols;
  int height = mask.rows;
  for (int i = 0; i < joint.joints.size(); ++i) {
    const cv::Point2f& point = joint.joints[i];
    int vis = joint.isVisible[i];
    if (vis <= 1) {
      int x = (int)(point.x);
      x = x < 0 ? 0 : (x > width - 1 ? width - 1 : x);
      int y = (int)(point.y);
      y = y < 0 ? 0 : (y > height - 1 ? height - 1 : y);
      if (mask.at<uchar>(y,x) > 128) {
        num++;
      }
    }
  }
  return num;
}

template <typename Dtype>
int MpiiMaskGenerator<Dtype>::Get_points(const Joints& joint, const BoundingBox<Dtype>& box) {
  int num = 0;
  for (int i = 0; i < joint.joints.size(); ++i) {
    const cv::Point2f& point = joint.joints[i];
    int vis = joint.isVisible[i];
    if ((vis <= 1) && (point.x >= box.x1_) && (point.y >= box.y1_) && (point.x <= box.x2_) && (point.y <= box.y2_)) {
      num++;
    }
  }
  return num;
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::ResizeBbox(const Joints& joint, const BoundingBox<Dtype>& box, BoundingBox<Dtype>* resized_box) {
  Dtype xmin = box.x1_;
  Dtype ymin = box.y1_;
  Dtype xmax = box.x2_;
  Dtype ymax = box.y2_;
  for (int i = 0; i < joint.joints.size(); ++i) {
    const cv::Point2f& point = joint.joints[i];
    int vis = joint.isVisible[i];
    if (vis <= 1) {
      // resized
      Dtype x = point.x;
      Dtype y = point.y;
      if (x < xmin) {
        xmin = x;
      }
      if (x > xmax) {
        xmax = x;
      }
      if (y < ymin) {
        ymin = y;
      }
      if (y > ymax) {
        ymax = y;
      }
    }
  }
  resized_box->x1_ = xmin;
  resized_box->y1_ = ymin;
  resized_box->x2_ = xmax;
  resized_box->y2_ = ymax;
}

template <typename Dtype>
bool MpiiMaskGenerator<Dtype>::Match(const vector<BoundingBox<Dtype> >& bboxes,
                                     MetaDataAll<Dtype>* kps,
                                     vector<bool>* matched) {
  matched->clear();
  for (int i = 0; i < bboxes.size(); ++i) {
    matched->push_back(false);
  }
  for (int p = 0; p < kps->joint.size(); ++p) {
    // match for this person
    int matched_id = -1;
    int max_points = 0;
    Dtype min_area = 1e6;
    for (int i = 0; i < bboxes.size(); ++i) {
      if ((*matched)[i]) continue;
      int num_points = Get_points(kps->joint[p], bboxes[i]);
      // 至少应该包含５个点
      if (num_points < 5) continue;
      if (num_points > max_points) {
        max_points = num_points;
        matched_id = i;
        min_area = bboxes[i].compute_area();
      } else if (num_points == max_points) {
        Dtype area = bboxes[i].compute_area();
        if (area < min_area) {
          matched_id = i;
          min_area = area;
        }
      } else {
        // do nothing
      }
    }
    if (matched_id >= 0) {
      // matched
      // BoundingBox<Dtype> resized_box;
      // ResizeBbox(kps->joint[p], bboxes[matched_id], &resized_box);
      kps->bbox[p].x1_ = bboxes[matched_id].x1_;
      kps->bbox[p].y1_ = bboxes[matched_id].y1_;
      kps->bbox[p].x2_ = bboxes[matched_id].x2_;
      kps->bbox[p].y2_ = bboxes[matched_id].y2_;
      kps->area[p] = kps->bbox[p].compute_area();
      kps->scale[p] = kps->bbox[p].get_height() / (Dtype)368;
      kps->objpos[p].x = (kps->bbox[p].x1_ + kps->bbox[p].x2_)/2;
      kps->objpos[p].y = (kps->bbox[p].y1_ + kps->bbox[p].y2_)/2;
      (*matched)[matched_id] = true;
    } else {
      // unmatched: do nothing
      return false;
    }
  }
  return true;
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::GenMask(const MetaDataAll<Dtype>& kps, const vector<BoundingBox<Dtype> >& bboxes, const vector<bool>& matched, cv::Mat* mask) {
  cv::Mat mask_image(kps.img_size,CV_8UC1,cv::Scalar(255));
  for (int k = 0; k < bboxes.size(); ++k) {
    if (matched[k]) continue;
    // 剩下的所有的boxes全部清零
    int tl_x = (int)bboxes[k].x1_;
    tl_x = std::min(std::max(tl_x,0),mask_image.cols-1);
    int tl_y = (int)bboxes[k].y1_;
    tl_y = std::min(std::max(tl_y,0),mask_image.rows-1);
    int width = (int)bboxes[k].get_width();
    width = std::min(std::max(width,0),mask_image.cols-tl_x);
    int height = (int)bboxes[k].get_height();
    height = std::min(std::max(height,0),mask_image.rows-tl_y);
    cv::Rect roi(tl_x,tl_y,width,height);
    cv::Mat mask_roi = mask_image(roi);
    for (int i = 0; i < mask_roi.rows; ++i) {
      for (int j = 0; j < mask_roi.cols; ++j) {
        mask_roi.at<uchar>(i,j) = 0;
      }
    }
  }
  *mask = mask_image;
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::SaveMask(const MetaDataAll<Dtype>& kps, const cv::Mat& mask) {
  int delim_pos = kps.img_path.find_last_of("/");
  int end_pos = kps.img_path.rfind(".");
  int num_str = end_pos - delim_pos - 1;
  const string image_name = kps.img_path.substr(delim_pos+1, num_str);
  const string mask_name = "mask_miss_" + image_name + ".jpg";
  const string mask_path = output_folder_ + '/' + mask_folder_ + mask_name;
  imwrite(mask_path, mask);
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::Split(const MetaDataAll<Dtype>& kps, const cv::Mat& mask, vector<MetaData<Dtype> >* metas) {
  metas->clear();
  // 获取图片名
  int num = kps.numPeople;
  int delim_pos = kps.img_path.find_last_of("/");
  int end_pos = kps.img_path.rfind(".");
  int num_str = end_pos - delim_pos - 1;
  const string image_name = kps.img_path.substr(delim_pos+1, num_str);
  // 获取mask的路径
  const string mask_name = "mask_miss_" + image_name + ".jpg";
  const string mask_path = mask_folder_ + mask_name;
  // 生成ｎｕｍ个样本
  int p_idx = 0;
  for (int i = 0; i < num; ++i) {
    MetaData<Dtype> meta;
    // 基础数据
    meta.img_path = "MPII/" + kps.img_path;
    // meta.mask_miss_path = "MPII/" + mask_path;
    meta.mask_miss_path = "None";
    meta.dataset = kps.dataset;
    meta.isValidation = kps.isValidation;
    meta.img_size = kps.img_size;
    meta.numOtherPeople = kps.numPeople - 1;
    meta.people_index = i;
    meta.annolist_index = xml_idx_ + p_idx;
    // self
    meta.objpos = kps.objpos[i];
    meta.area = kps.area[i];
    meta.scale_self = kps.scale[i];
    meta.joint_self = kps.joint[i];
    meta.bbox = kps.bbox[i];
    // int num_k = GetMaskPoints(meta.joint_self, mask);
    // if (num_k < 5) continue;
    // others
    if (kps.numPeople > 1) {
      meta.objpos_other.resize(kps.numPeople - 1);
      meta.scale_other.resize(kps.numPeople - 1);
      meta.area_other.resize(kps.numPeople - 1);
      meta.joint_others.resize(kps.numPeople - 1);
      meta.bbox_others.resize(kps.numPeople - 1);
      int op = 0;
      for (int p = 0; p < kps.numPeople; ++p) {
        if (p == i) continue;
        meta.objpos_other[op] = kps.objpos[p];
        meta.scale_other[op] = kps.scale[p];
        meta.area_other[op] = kps.area[p];
        meta.joint_others[op] = kps.joint[p];
        meta.bbox_others[op] = kps.bbox[p];
        op++;
      }
    }
    // push
    metas->push_back(meta);
    p_idx++;
  }
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::SaveXml(const MetaData<Dtype>& meta, const int idx){
  // 获取保存路径
  char xmlname[256];
  int delim_pos = meta.img_path.find_last_of("/");
  int end_pos = meta.img_path.rfind(".");
  int num_str = end_pos - delim_pos - 1;
  const string image_name = meta.img_path.substr(delim_pos+1, num_str);
  int image_id = atoi(image_name.c_str());
  // sprintf(xmlname, "%s/%s%08d.xml", output_folder_.c_str(), xml_folder_.c_str(), idx);
  sprintf(xmlname, "%s/%08d.xml", output_folder_.c_str(), idx);
  // root point
  ptree anno;
  // Path
  anno.put("ImagePath", meta.img_path);
  anno.put("MaskMissPath", meta.mask_miss_path);
  // MetaData
  ptree metadata;
  metadata.put("dataset", meta.dataset);
  metadata.put<int>("isValidation", (meta.isValidation ? 1 : 0));
  metadata.put<int>("width", meta.img_size.width);
  metadata.put<int>("height", meta.img_size.height);
  // objpos
  ptree objpos;
  objpos.put<Dtype>("center_x", meta.objpos.x);
  objpos.put<Dtype>("center_y", meta.objpos.y);
  metadata.add_child("objpos", objpos);
  // image_id
  metadata.put<int>("imageId", image_id);
  // bbox
  ptree bbox;
  bbox.put<Dtype>("xmin", meta.bbox.x1_);
  bbox.put<Dtype>("ymin", meta.bbox.y1_);
  bbox.put<Dtype>("width", meta.bbox.get_width());
  bbox.put<Dtype>("height", meta.bbox.get_height());
  metadata.add_child("bbox", bbox);
  // area
  metadata.put<Dtype>("area", meta.bbox.compute_area());
  // num_keypoints
  metadata.put<int>("num_keypoints", 16);
  // scale
  metadata.put<Dtype>("scale", meta.scale_self);
  // joint_self
  ptree joint_self;
  for (int k = 0; k < 16; ++k) {
    ptree kp_k;
    kp_k.put<Dtype>("x", meta.joint_self.joints[k].x);
    kp_k.put<Dtype>("y", meta.joint_self.joints[k].y);
    kp_k.put<int>("vis", meta.joint_self.isVisible[k]);
    char temp[32];
    sprintf(temp, "kp_%d", k+1);
    joint_self.add_child(temp,kp_k);
  }
  metadata.add_child("joint_self", joint_self);
  // annolist_index
  metadata.put<int>("annolist_index", meta.annolist_index);
  // people_index
  metadata.put<int>("people_index", meta.people_index);
  // numOtherPeople
  metadata.put<int>("numOtherPeople", meta.numOtherPeople);
  // op
  if (meta.numOtherPeople > 0) {
    // scale_other
    ptree scale_other;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      char temp[32];
      sprintf(temp, "scale_%d", p+1);
      scale_other.put<Dtype>(temp, meta.scale_other[p]);
    }
    metadata.add_child("scale_other", scale_other);
    // objpos_other
    ptree objpos_other;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      ptree objpos_o;
      char temp[32];
      sprintf(temp, "objpos_%d", p+1);
      objpos_o.put<Dtype>("center_x", meta.objpos_other[p].x);
      objpos_o.put<Dtype>("center_y", meta.objpos_other[p].y);
      objpos_other.add_child(temp, objpos_o);
    }
    metadata.add_child("objpos_other", objpos_other);
    // bbox_other
    ptree bbox_other;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      ptree bbox_o;
      char temp[32];
      sprintf(temp, "bbox_%d", p+1);
      bbox_o.put<Dtype>("xmin", meta.bbox_others[p].x1_);
      bbox_o.put<Dtype>("ymin", meta.bbox_others[p].y1_);
      bbox_o.put<Dtype>("width", meta.bbox_others[p].get_width());
      bbox_o.put<Dtype>("height", meta.bbox_others[p].get_height());
      bbox_other.add_child(temp, bbox_o);
    }
    metadata.add_child("bbox_other", bbox_other);
    // area_other
    ptree area_other;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      char temp[32];
      sprintf(temp, "area_%d", p+1);
      area_other.put<Dtype>(temp, meta.area_other[p]);
    }
    metadata.add_child("area_other", area_other);
    // num_keypoints_other
    ptree nk_other;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      char temp[48];
      sprintf(temp, "num_keypoints_%d", p+1);
      nk_other.put<int>(temp, 16);
    }
    metadata.add_child("num_keypoints_other", nk_other);
    // joint_others
    ptree joint_others;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      ptree joint_p;
      char temp1[32];
      sprintf(temp1, "joint_%d", p+1);
      for (int i = 0; i < 16; ++i) {
        ptree kp_i;
        char temp2[32];
        sprintf(temp2, "kp_%d", i+1);
        kp_i.put<Dtype>("x", meta.joint_others[p].joints[i].x);
        kp_i.put<Dtype>("y", meta.joint_others[p].joints[i].y);
        kp_i.put<int>("vis", meta.joint_others[p].isVisible[i]);
        joint_p.add_child(temp2,kp_i);
      }
      joint_others.add_child(temp1,joint_p);
    }
    metadata.add_child("joint_others", joint_others);
  }
  anno.add_child("MetaData", metadata);
  ptree doc;
  doc.add_child("Annotations",anno);
  // write
  xml_writer_settings<char> settings('\t',1);
  write_xml(xmlname, doc, std::locale(), settings);
  // LOG(INFO) << "save xml.";
}

template <typename Dtype>
int MpiiMaskGenerator<Dtype>::SaveXml(const vector<MetaData<Dtype> >& metas) {
  if (metas.size() == 0) return xml_idx_;
  for (int i = 0; i < metas.size(); ++i) {
    SaveXml(metas[i], xml_idx_ + i);
  }
  return (xml_idx_ + metas.size());
}

template <typename Dtype>
void MpiiMaskGenerator<Dtype>::SaveImage(const MetaDataAll<Dtype>& kps, const cv::Mat& mask, const bool show_box) {
  cv::Mat image;
  LoadImage(kps.img_path, &image);
  // draw kps & box
  for(int p = 0; p < kps.numPeople; p++) {
    // kps
    for(int i = 0; i < 16; i++) {
      if(kps.joint[p].isVisible[i] <= 1)
        circle(image, kps.joint[p].joints[i], 5, CV_RGB(255,0,0), -1);
    }
    // box
    const BoundingBox<Dtype>& bbox = kps.bbox[p];
    bbox.Draw(0,255,0,&image);
    /**
     * Modified for showing scale & w/h
     */
    //  show scale in right-corner
    cv::Point up_left_pt1(bbox.x1_ + 10, bbox.y1_ + 25);
    cv::Point up_left_pt2(bbox.x1_ + 8, bbox.y1_ + 23);
    char sm_buffer[50];
    snprintf(sm_buffer, sizeof(sm_buffer), "%.2f", (float)kps.scale[p]);
    cv::putText(image, sm_buffer, up_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(image, sm_buffer, up_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(128,128,255), 2);
    // show w/h in left-corner
    cv::Point bottom_left_pt1(bbox.x1_ + 10, bbox.y2_ - 10);
    cv::Point bottom_left_pt2(bbox.x1_ + 8, bbox.y2_ - 8);
    char whm_buffer[50];
    snprintf(whm_buffer, sizeof(whm_buffer), "%d/%d", (int)bbox.get_width(), (int)bbox.get_height());
    cv::putText(image, whm_buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(image, whm_buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(250,128,250), 2);
  }
  // mask image * mask
  // for (int i = 0; i < image.rows; ++i) {
  //   for (int j = 0; j < image.cols; ++j) {
  //     Vec3b& rgb = image.at<Vec3b>(i, j);
  //     Dtype v = (Dtype)mask.at<uchar>(i,j)/255;
  //     image.at<Vec3b>(i, j)[0] = (uchar)((Dtype)rgb[0] * v);
  //     image.at<Vec3b>(i, j)[1] = (uchar)((Dtype)rgb[1] * v);
  //     image.at<Vec3b>(i, j)[2] = (uchar)((Dtype)rgb[2] * v);
  //   }
  // }
  // save
  int delim_pos = kps.img_path.find_last_of("/");
  int end_pos = kps.img_path.rfind(".");
  int num_str = end_pos - delim_pos - 1;
  const string image_name = kps.img_path.substr(delim_pos+1, num_str);
  const string image_save_file = output_folder_ + '/' + image_name + ".jpg";
  imwrite(image_save_file, image);
}

INSTANTIATE_CLASS(MpiiMaskGenerator);
}
