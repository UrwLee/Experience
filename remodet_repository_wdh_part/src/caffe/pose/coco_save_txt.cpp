#include "caffe/pose/coco_save_txt.hpp"
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

template <typename Dtype>
CocoTxtSaver<Dtype>::CocoTxtSaver(
                const std::string& xml_dir,
                const std::string& output_file): xml_dir_(xml_dir), output_file_(output_file) {
  output_file_ptr_ = fopen(output_file.c_str(), "w");
}

template <typename Dtype>
void CocoTxtSaver<Dtype>::Save() {
  if (!bfs::is_directory(xml_dir_)) {
    LOG(FATAL) << "Error - " << xml_dir_ <<  " is not a valid directory.";
    return;
  }
  // 获取文件夹内所有xml文件
  const boost::regex annotation_filter(".*\\.xml");
  vector<string> xml_files;
  find_matching_files(xml_dir_, annotation_filter, &xml_files);
  if (xml_files.size() == 0) {
    LOG(FATAL) << "Error: Found no xml files in " << xml_dir_;
  }
  //遍历
  const bool doTest = false;
  const int num = doTest ? 10 : xml_files.size();
  for (int i = 0; i < num; ++i) {
    const string xml_path = xml_dir_ + '/' + xml_files[i];
    // read xml
    MetaData<Dtype> meta;
    LoadAnnotationFromXmlFile(xml_path,&meta);
    // save to txt
    /**
     * format:
     * (1) image_path (string)
     * (2) image_size: width height (int)
     * (3) person_id within the image (int)
     * (4) scale (float)
     * (5) bbox: [xmin,ymin,xmax,ymax] (4 int)
     * (6) joints: [18x3] (int - 48)
     */
    LOG(INFO) << "process for " << meta.img_path;
    // image_path
    fprintf(output_file_ptr_, "%s ", meta.img_path.c_str());
    // image_size
    fprintf(output_file_ptr_, "%d %d ", meta.img_size.width, meta.img_size.height);
    // person_id
    fprintf(output_file_ptr_, "%d ", meta.people_index);
    // scale
    fprintf(output_file_ptr_, "%.2f ", (float)meta.scale_self);
    // bbox [xmin, ymin, xmax, ymax]
    fprintf(output_file_ptr_, "%d %d %d %d ", (int)meta.bbox.x1_,(int)meta.bbox.y1_,(int)meta.bbox.x2_,(int)meta.bbox.y2_);
    // joints
    CHECK_EQ(meta.joint_self.joints.size(), 17);
    CHECK_EQ(meta.joint_self.isVisible.size(), 17);
    for (int j = 0; j < 17; ++j) {
      if (j == 16) {
        fprintf(output_file_ptr_, "%d %d %d\n", (int)meta.joint_self.joints[j].x,(int)meta.joint_self.joints[j].y,meta.joint_self.isVisible[j]);
      } else {
        fprintf(output_file_ptr_, "%d %d %d ", (int)meta.joint_self.joints[j].x,(int)meta.joint_self.joints[j].y,meta.joint_self.isVisible[j]);
      }
    }
  }
  // close file
  fclose(output_file_ptr_);
}

template <typename Dtype>
bool CocoTxtSaver<Dtype>::LoadAnnotationFromXmlFile(const string& xml_file,
                                                MetaData<Dtype>* meta) {
  ptree pt;
  read_xml(xml_file, pt);
  // 读取路径
  meta->img_path = pt.get<string>("Annotations.ImagePath");
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
  return true;
}

INSTANTIATE_CLASS(CocoTxtSaver);
}
