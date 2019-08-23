#include "caffe/pose/coco_image_loader.hpp"
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
CocoImageLoader<Dtype>::CocoImageLoader(const std::string& image_folder, const std::string& xml_folder) {
  if (!bfs::is_directory(xml_folder)) {
    LOG(FATAL) << "Error - " << xml_folder <<  " is not a valid directory.";
    return;
  }
  // 获取文件夹内所有ｘｍｌ文件
  const boost::regex annotation_filter(".*\\.xml");
  vector<string> xml_files;
  find_matching_files(xml_folder, annotation_filter, &xml_files);

  if (xml_files.size() == 0) {
    LOG(FATAL) << "Error: Found no xml files in " << xml_folder;
  }
  for (int i = 0; i < xml_files.size(); ++i) {
    // 获得第i个标注文件的路径
    const string& xml_file = xml_files[i];
    MetaData<Dtype> meta;
    // xml路径
    const string xml_path = xml_folder + "/" + xml_file;
    LOG(INFO) << "Loading " << xml_path << " ... ";
    // 定义视频路径
    LoadAnnotationFromXmlFile(xml_path,image_folder,&meta);
    this->annotations_.push_back(meta);
  }
  LOG(INFO) << "Mpii ImageLoader has been initialized.";
}

template <typename Dtype>
bool CocoImageLoader<Dtype>::LoadAnnotationFromXmlFile(const string& annotation_file, const string& image_path,
                                                MetaData<Dtype>* meta) {
  ptree pt;
  read_xml(annotation_file, pt);
  // 读取路径
  string temp_img = pt.get<string>("Annotations.ImagePath");
  meta->img_path = image_path + '/' + temp_img;
  try {
    string temp_mask_all = pt.get<string>("Annotations.MaskAllPath");
    string temp_mask_miss = pt.get<string>("Annotations.MaskMissPath");
    meta->mask_all_path = image_path + '/' + temp_mask_all;
    meta->mask_miss_path = image_path + '/' + temp_mask_miss;
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << annotation_file << ": " << e.what();
    meta->mask_all_path = "";
    meta->mask_miss_path = "";
  }
  // 读取Metadata
  meta->dataset = pt.get<string>("Annotations.MetaData.dataset");
  meta->isValidation = (pt.get<int>("Annotations.MetaData.isValidation") == 0 ? false : true);
  int width = pt.get<int>("Annotations.MetaData.width");
  int height = pt.get<int>("Annotations.MetaData.height");
  meta->img_size = Size(width,height);
  meta->numOtherPeople = pt.get<int>("Annotations.MetaData.numOtherPeople");
  meta->people_index = pt.get<int>("Annotations.MetaData.people_index");
  meta->annolist_index = pt.get<int>("Annotations.MetaData.annolist_index");
  // objpos & scale
  meta->objpos.x = pt.get<Dtype>("Annotations.MetaData.objpos.center_x");
  meta->objpos.y = pt.get<Dtype>("Annotations.MetaData.objpos.center_y");
  meta->objpos -= Point2f(1,1);
  meta->scale_self = pt.get<Dtype>("Annotations.MetaData.scale");
  meta->area = pt.get<Dtype>("Annotations.MetaData.area");
  // box
  meta->bbox.x1_ = pt.get<Dtype>("Annotations.MetaData.bbox.xmin");
  meta->bbox.y1_ = pt.get<Dtype>("Annotations.MetaData.bbox.ymin");
  meta->bbox.x2_ = meta->bbox.x1_ + pt.get<Dtype>("Annotations.MetaData.bbox.width");
  meta->bbox.y2_ = meta->bbox.y1_ + pt.get<Dtype>("Annotations.MetaData.bbox.height");
  // joints
  meta->joint_self.joints.resize(17);
  meta->joint_self.isVisible.resize(17);
  for(int i = 0; i < 17; ++i) {
    char temp_x[256], temp_y[256], temp_vis[256];
    sprintf(temp_x, "Annotations.MetaData.joint_self.kp_%d.x", i+1);
    sprintf(temp_y, "Annotations.MetaData.joint_self.kp_%d.y", i+1);
    sprintf(temp_vis, "Annotations.MetaData.joint_self.kp_%d.vis", i+1);
    meta->joint_self.joints[i].x = pt.get<Dtype>(temp_x);
    meta->joint_self.joints[i].y = pt.get<Dtype>(temp_y);
    meta->joint_self.joints[i] -= Point2f(1,1);
    int isVisible = pt.get<int>(temp_vis);
    meta->joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
    if(meta->joint_self.joints[i].x < 0 || meta->joint_self.joints[i].y < 0 ||
       meta->joint_self.joints[i].x >= meta->img_size.width || meta->joint_self.joints[i].y >= meta->img_size.height) {
      meta->joint_self.isVisible[i] = 2;
    }
  }
  // other people
  meta->objpos_other.clear();
  meta->scale_other.clear();
  meta->joint_others.clear();
  meta->area_other.clear();
  meta->bbox_others.clear();
  if (meta->numOtherPeople > 0) {
    meta->objpos_other.resize(meta->numOtherPeople);
    meta->scale_other.resize(meta->numOtherPeople);
    meta->joint_others.resize(meta->numOtherPeople);
    meta->area_other.resize(meta->numOtherPeople);
    meta->bbox_others.resize(meta->numOtherPeople);
    for(int p = 0; p < meta->numOtherPeople; p++) {
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
      meta->objpos_other[p].x = pt.get<Dtype>(temp_x);
      meta->objpos_other[p].y = pt.get<Dtype>(temp_y);
      meta->objpos_other[p] -= Point2f(1,1);
      meta->scale_other[p] = pt.get<Dtype>(temp_scale);
      meta->area_other[p] = pt.get<Dtype>(temp_area);
      meta->bbox_others[p].x1_ = pt.get<Dtype>(temp_xmin);
      meta->bbox_others[p].y1_ = pt.get<Dtype>(temp_ymin);
      meta->bbox_others[p].x2_ = meta->bbox_others[p].x1_ + pt.get<Dtype>(temp_width);
      meta->bbox_others[p].y2_ = meta->bbox_others[p].y1_ + pt.get<Dtype>(temp_height);
      // joints
      meta->joint_others[p].joints.resize(17);
      meta->joint_others[p].isVisible.resize(17);
      for (int i = 0; i < 17; i++) {
        char joint_x[256], joint_y[256], joint_vis[256];
        sprintf(joint_x, "Annotations.MetaData.joint_others.joint_%d.kp_%d.x", p+1, i+1);
        sprintf(joint_y, "Annotations.MetaData.joint_others.joint_%d.kp_%d.y", p+1, i+1);
        sprintf(joint_vis, "Annotations.MetaData.joint_others.joint_%d.kp_%d.vis", p+1, i+1);
        meta->joint_others[p].joints[i].x = pt.get<Dtype>(joint_x);
        meta->joint_others[p].joints[i].y = pt.get<Dtype>(joint_y);
        meta->joint_others[p].joints[i] -= Point2f(1,1);
        int isVisible = pt.get<int>(joint_vis);
        meta->joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
        if(meta->joint_others[p].joints[i].x < 0 || meta->joint_others[p].joints[i].y < 0 ||
           meta->joint_others[p].joints[i].x >= meta->img_size.width || meta->joint_others[p].joints[i].y >= meta->img_size.height){
          meta->joint_others[p].isVisible[i] = 2;
        }
      }
    }
  }
  return true;
}

INSTANTIATE_CLASS(CocoImageLoader);
}
