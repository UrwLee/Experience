#include "caffe/pose/xml_modification.hpp"
#include "caffe/tracker/basic.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>

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

bool doTest = false;

template <typename Dtype>
XmlModifier<Dtype>::XmlModifier(const std::string& xml_folder,
                                const std::string& output_folder)
  : xml_folder_(xml_folder), output_folder_(output_folder) {
  if (!bfs::is_directory(xml_folder)) {
    LOG(FATAL) << "Error - " << xml_folder <<  " is not a valid directory.";
    return;
  }
}

template <typename Dtype>
void XmlModifier<Dtype>::Modify() {
  const boost::regex annotation_filter(".*\\.xml");
  vector<string> xml_files;
  find_matching_files(xml_folder_, annotation_filter, &xml_files);
  if (xml_files.size() == 0) {
    LOG(FATAL) << "Error: Found no xml files in " << xml_folder_;
  }

  int num = doTest ? 1 : xml_files.size();

  for (int i = 0; i < num; ++i) {
    const string& xml_file = xml_files[i];
    LOG(INFO) << "process for " << xml_file;
    const string xml_path = xml_folder_ + "/" + xml_file;
    MetaData<Dtype> meta;
    if (!LoadAnnotationFromXmlFile(xml_path,&meta)) {
      LOG(FATAL) << "Error: read metadata from " << xml_path;
    }
    // Modify
    ModifyXML(&meta);
    // Save
    SaveXml(meta, xml_file);
  }
}

template <typename Dtype>
bool XmlModifier<Dtype>::LoadAnnotationFromXmlFile(const string& annotation_file, MetaData<Dtype>* meta) {
  ptree pt;
  read_xml(annotation_file, pt);
  // 读取路径
  meta->img_path = pt.get<string>("Annotations.ImagePath");
  try {
    meta->mask_all_path = pt.get<string>("Annotations.MaskAllPath");
    meta->mask_miss_path = pt.get<string>("Annotations.MaskMissPath");
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
    meta->joint_self.isVisible[i] = pt.get<int>(temp_vis);
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
    for(int p = 0; p < meta->numOtherPeople; p++){
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
        meta->joint_others[p].isVisible[i] = pt.get<int>(joint_vis);
      }
    }
  }
  return true;
}

template <typename Dtype>
void XmlModifier<Dtype>::ModifyXML(MetaData<Dtype>* meta) {
  // path
  meta->img_path = "coco/" + meta->img_path;
  meta->mask_miss_path = "coco/" + meta->mask_miss_path;
  meta->mask_all_path = "coco/" + meta->mask_all_path;
  // dataset
  meta->dataset = "COCO";
}

template <typename Dtype>
void XmlModifier<Dtype>::SaveXml(const MetaData<Dtype>& meta, const std::string& xml_name) {
  // 获取保存路径
  const string xml_path = output_folder_ + '/' + xml_name;
  // root point
  ptree anno;
  // Path
  anno.put("ImagePath", meta.img_path);
  anno.put("MaskMissPath", meta.mask_miss_path);
  anno.put("MaskAllPath", meta.mask_all_path);
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
  // bbox
  ptree bbox;
  bbox.put<Dtype>("xmin", meta.bbox.x1_);
  bbox.put<Dtype>("ymin", meta.bbox.y1_);
  bbox.put<Dtype>("width", meta.bbox.get_width());
  bbox.put<Dtype>("height", meta.bbox.get_height());
  metadata.add_child("bbox", bbox);
  // area
  metadata.put<Dtype>("area", meta.area);
  // num_keypoints
  metadata.put<int>("num_keypoints", 17);
  // scale
  metadata.put<Dtype>("scale", meta.scale_self);
  // joint_self
  ptree joint_self;
  for (int k = 0; k < 17; ++k) {
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
      nk_other.put<int>(temp, 17);
    }
    metadata.add_child("num_keypoints_other", nk_other);
    // joint_others
    ptree joint_others;
    for (int p = 0; p < meta.numOtherPeople; ++p) {
      ptree joint_p;
      char temp1[32];
      sprintf(temp1, "joint_%d", p+1);
      for (int i = 0; i < 17; ++i) {
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
  write_xml(xml_path, doc, std::locale(), settings);
}

INSTANTIATE_CLASS(XmlModifier);
}
