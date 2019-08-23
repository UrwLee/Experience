#include "caffe/pose/mpii_image_masker.hpp"
#include "caffe/tracker/basic.hpp"
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

namespace caffe {

const bool doTest = false;
const int testNum = 100;

using std::string;
using std::vector;
namespace bfs = boost::filesystem;

template <typename Dtype>
MpiiMasker<Dtype>::MpiiMasker(
            const std::string& box_xml_dir,
            const std::string& kps_xml_dir,
            const std::string& image_folder,
            const std::string& output_folder):
  box_xml_dir_(box_xml_dir), kps_xml_dir_(kps_xml_dir), image_folder_(image_folder), output_folder_(output_folder){
}

template <typename Dtype>
void MpiiMasker<Dtype>::Process(const bool save_image, const bool show_box) {
  if (!bfs::is_directory(box_xml_dir_)) {
    LOG(FATAL) << "Error - " << box_xml_dir_ <<  " is not a valid directory.";
    return;
  }
  if (!bfs::is_directory(kps_xml_dir_)) {
    LOG(FATAL) << "Error - " << kps_xml_dir_ <<  " is not a valid directory.";
    return;
  }
  // 获取文件夹内所有xml-kps文件
  const boost::regex annotation_filter(".*\\.xml");
  vector<string> xml_kps_files;
  find_matching_files(kps_xml_dir_, annotation_filter, &xml_kps_files);
  if (xml_kps_files.size() == 0) {
    LOG(FATAL) << "Error: Found no xml files in " << kps_xml_dir_;
  }
  //遍历
  const int num = doTest ? testNum : xml_kps_files.size();
  int idx = 0;
  for (int i = 0; i < num; ++i) {
    const string box_xml_path = box_xml_dir_ + '/' + xml_kps_files[i];
    const string kps_xml_path = kps_xml_dir_ + '/' + xml_kps_files[i];
    MpiiMaskGenerator<Dtype> masker(box_xml_path, kps_xml_path,
                                    image_folder_, output_folder_, idx);
    LOG(INFO) << "process for " << xml_kps_files[i];
    idx = masker.Generate(save_image, show_box);
  }
}

INSTANTIATE_CLASS(MpiiMasker);
}
