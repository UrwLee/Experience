#include "caffe/mask/coco_anno_loader.hpp"
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
CocoAnnoLoader<Dtype>::CocoAnnoLoader(const std::string& image_folder, const std::string& xml_folder) {
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

  const bool doTest = true;
  const int num_images = doTest ? 500 : xml_files.size();

  for (int i = 0; i < num_images; ++i) {
    const string& xml_file = xml_files[i];
    AnnoData<Dtype> anno;
    // xml路径
    const string xml_path = xml_folder + "/" + xml_file;
    LOG(INFO) << "Loading " << xml_path << " ... ";
    LoadAnnotationFromXmlFile(xml_path,image_folder,&anno);
    this->annotations_.push_back(anno);
  }
  LOG(INFO) << "CocoAnnoLoader has been initialized.";
}

template <typename Dtype>
bool CocoAnnoLoader<Dtype>::LoadAnnotationFromXmlFile(const string& annotation_file, const string& image_path,
                                                      AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(annotation_file, pt);
  // 读取路径
  anno->img_path = image_path + '/' + pt.get<string>("Annotations.ImagePath");
  // dataset
  anno->dataset = pt.get<string>("Annotations.DataSet");
  // img_width
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  // img_height
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  // num_person
  anno->num_person = pt.get<int>("Annotations.NumPerson");
  // Instance
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_cid[128], temp_pid[128], temp_iscrowd[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    char temp_mask_included[128], temp_mask_path[128];
    char temp_kps_included[128], temp_num_kps[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i+1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);
    sprintf(temp_mask_included, "Annotations.Object_%d.mask_included", i+1);
    sprintf(temp_mask_path, "Annotations.Object_%d.mask_path", i+1);
    sprintf(temp_kps_included, "Annotations.Object_%d.kps_included", i+1);
    sprintf(temp_num_kps, "Annotations.Object_%d.num_kps", i+1);
    ins.cid = pt.get<int>(temp_cid);
    ins.pid = pt.get<int>(temp_pid);
    ins.iscrowd = pt.get<int>(temp_iscrowd) == 0 ? false : true;
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_xmax);
    ins.bbox.y2_ = pt.get<Dtype>(temp_ymax);
    ins.mask_included = pt.get<int>(temp_mask_included) == 0 ? false : true;
    ins.mask_path = image_path + '/' + pt.get<string>(temp_mask_path);
    ins.kps_included = pt.get<int>(temp_kps_included) == 0 ? false : true;
    ins.num_kps = pt.get<int>(temp_num_kps);
    // kps
    ins.joint.joints.resize(17);
    ins.joint.isVisible.resize(17);
    for (int k = 0; k < 17; ++k) {
      char temp_x[128], temp_y[128], temp_vis[128];
      sprintf(temp_x, "Annotations.Object_%d.joint.kp_%d.x", i+1,k+1);
      sprintf(temp_y, "Annotations.Object_%d.joint.kp_%d.y", i+1,k+1);
      sprintf(temp_vis, "Annotations.Object_%d.joint.kp_%d.vis", i+1,k+1);
      ins.joint.joints[k].x = pt.get<Dtype>(temp_x);
      ins.joint.joints[k].y = pt.get<Dtype>(temp_y);
      ins.joint.isVisible[k] = pt.get<int>(temp_vis);
    }
    // push_back
    anno->instances.push_back(ins);
  }
  return true;
}

INSTANTIATE_CLASS(CocoAnnoLoader);
}
