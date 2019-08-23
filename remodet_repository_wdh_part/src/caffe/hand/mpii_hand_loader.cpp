#include "caffe/hand/mpii_hand_loader.hpp"
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
MpiiHandLoader<Dtype>::MpiiHandLoader(const std::string& image_folder, const std::string& xml_folder) {
  if (!bfs::is_directory(xml_folder)) {
    LOG(FATAL) << "Error - " << xml_folder <<  " is not a valid directory.";
    return;
  }
  const boost::regex annotation_filter(".*\\.xml");
  vector<string> xml_files;
  find_matching_files(xml_folder, annotation_filter, &xml_files);

  if (xml_files.size() == 0) {
    LOG(FATAL) << "Error: Found no xml files in " << xml_folder;
  }

  const bool doTest = false;
  const int num_images = doTest ? 100 : xml_files.size();

  for (int i = 0; i < num_images; ++i) {
    const string& xml_file = xml_files[i];
    MData<Dtype> meta;
    const string xml_path = xml_folder + "/" + xml_file;
    LOG(INFO) << "Loading " << xml_path << " ... ";
    LoadAnnotationFromXmlFile(xml_path,image_folder,&meta);
    this->annotations_.push_back(meta);
  }
  this->type_ = "MPII";
  LOG(INFO) << "MpiiHandLoader has been initialized.";
}

template <typename Dtype>
bool MpiiHandLoader<Dtype>::LoadAnnotationFromXmlFile(const string& annotation_file, const string& image_folder,
                                                      MData<Dtype>* meta) {
  ptree pt;
  read_xml(annotation_file, pt);
  // 读取路径
  meta->img_path = image_folder + '/' + pt.get<string>("Annotations.ImagePath");
  // dataset
  meta->dataset = pt.get<string>("Annotations.MetaData.dataset");
  // img_width
  meta->img_width = pt.get<int>("Annotations.MetaData.width");
  // img_height
  meta->img_height = pt.get<int>("Annotations.MetaData.height");
  // num_person
  int nop = pt.get<int>("Annotations.MetaData.numOtherPeople");
  meta->num_person = nop + 1;
  // main person
  InsData<Dtype> ins;
  ins.bbox.x1_ = pt.get<Dtype>("Annotations.MetaData.bbox.xmin");
  ins.bbox.y1_ = pt.get<Dtype>("Annotations.MetaData.bbox.ymin");
  ins.bbox.x2_ = ins.bbox.x1_ + pt.get<Dtype>("Annotations.MetaData.bbox.width");
  ins.bbox.y2_ = ins.bbox.y1_ + pt.get<Dtype>("Annotations.MetaData.bbox.height");
  ins.num_kps = pt.get<int>("Annotations.MetaData.num_keypoints");
  ins.kps_included = ins.num_kps >= 4 ? true : false;
  ins.joint.joints.resize(17);
  ins.joint.isVisible.resize(17);
  static const int mpii_coco_flags[16] = {16,13,10,8,9,10,-100,-100,-100,-100,0,-3,-6,-8,-7,-6};
  // the first 5 points rsvd
  for (int p = 0; p < 5; ++p) {
    ins.joint.joints[p].x = 0;
    ins.joint.joints[p].y = 0;
    ins.joint.isVisible[p] = 2;
  }
  for (int k = 0; k < 16; ++k) {
    if (mpii_coco_flags[k] < -10) continue;
    int rk = k + mpii_coco_flags[k];
    char temp_x[128], temp_y[128], temp_vis[128];
    sprintf(temp_x, "Annotations.MetaData.joint_self.kp_%d.x", k+1);
    sprintf(temp_y, "Annotations.MetaData.joint_self.kp_%d.y", k+1);
    sprintf(temp_vis, "Annotations.MetaData.joint_self.kp_%d.vis", k+1);
    ins.joint.joints[rk].x = pt.get<Dtype>(temp_x);
    ins.joint.joints[rk].y = pt.get<Dtype>(temp_y);
    ins.joint.isVisible[rk] = pt.get<int>(temp_vis);
  }
  meta->ins.push_back(ins);
  // OTHER Instance
  if (nop == 0) return true;
  for (int i = 0; i < nop; ++i) {
    InsData<Dtype> ins;
    char temp_xmin[128], temp_ymin[128], temp_width[128], temp_height[128];
    char temp_num_kps[128];
    sprintf(temp_xmin, "Annotations.MetaData.bbox_other.bbox_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.MetaData.bbox_other.bbox_%d.ymin", i+1);
    sprintf(temp_width, "Annotations.MetaData.bbox_other.bbox_%d.width", i+1);
    sprintf(temp_height, "Annotations.MetaData.bbox_other.bbox_%d.height", i+1);
    sprintf(temp_num_kps, "Annotations.MetaData.num_keypoints_other.num_keypoints_%d", i+1);
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_width) + ins.bbox.x1_;
    ins.bbox.y2_ = pt.get<Dtype>(temp_height) + ins.bbox.y1_;
    ins.num_kps = pt.get<int>(temp_num_kps);
    ins.kps_included = ins.num_kps >= 4 ? true : false;
    // kps
    ins.joint.joints.resize(17);
    ins.joint.isVisible.resize(17);
    for (int p = 0; p < 5; ++p) {
      ins.joint.joints[p].x = 0;
      ins.joint.joints[p].y = 0;
      ins.joint.isVisible[p] = 2;
    }
    for (int k = 0; k < 16; ++k) {
      if (mpii_coco_flags[k] < -10) continue;
      int rk = k + mpii_coco_flags[k];
      char temp_x[128], temp_y[128], temp_vis[128];
      sprintf(temp_x, "Annotations.MetaData.joint_others.joint_%d.kp_%d.x", i+1,k+1);
      sprintf(temp_y, "Annotations.MetaData.joint_others.joint_%d.kp_%d.y", i+1,k+1);
      sprintf(temp_vis, "Annotations.MetaData.joint_others.joint_%d.kp_%d.vis", i+1,k+1);
      ins.joint.joints[rk].x = pt.get<Dtype>(temp_x);
      ins.joint.joints[rk].y = pt.get<Dtype>(temp_y);
      ins.joint.isVisible[rk] = pt.get<int>(temp_vis);
    }
    meta->ins.push_back(ins);
  }
  return true;
}

INSTANTIATE_CLASS(MpiiHandLoader);
}
