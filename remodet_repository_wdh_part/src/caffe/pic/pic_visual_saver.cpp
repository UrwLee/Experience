#include "caffe/pic/pic_visual_saver.hpp"
#include <fstream>
#include <iostream>

#include "caffe/pic/pic_visualizer.hpp"

namespace caffe {

template <typename Dtype>
PicVisualSaver<Dtype>::PicVisualSaver(
                 const std::string& pic_file,
                 const std::string& image_dir,
                 const std::string& output_dir) {
  output_dir_ = output_dir;
  image_dir_ = image_dir;
  LoadMetas(pic_file, &metas_);
}

template <typename Dtype>
void PicVisualSaver<Dtype>::Save() {
  // save metas_
  for (int i = 0; i < metas_.size(); ++i) {
    LOG(INFO) << "Process for " << metas_[i].image_path;
    PicVisualizer<Dtype> picv(metas_[i], image_dir_, output_dir_);
    picv.Save();
  }
}

template <typename Dtype>
void PicVisualSaver<Dtype>::LoadMetas(const std::string& pic_file, std::vector<PicData<Dtype> >* metas) {
  std::ifstream input_file(pic_file.c_str());
  metas->clear();
  /**
   * 读取内容：
   * std::string image_path;
   * string bgw & bgh;
   * string x1 / y1 / x2 / y2;
   * string x,y,v * 3
   */
  std::string image_path;
  std::string bgw_str, bgh_str;
  std::string x1_str, y1_str, x2_str, y2_str;
  std::string pic_x1_str, pic_y1_str, pic_v1_str;
  std::string pic_x2_str, pic_y2_str, pic_v2_str;
  std::string pic_x3_str, pic_y3_str, pic_v3_str;
  while (input_file >> image_path >> bgw_str >> bgh_str
         >> x1_str >> y1_str >> x2_str >> y2_str
         >> pic_x1_str >> pic_y1_str >> pic_v1_str
         >> pic_x2_str >> pic_y2_str >> pic_v2_str
         >> pic_x3_str >> pic_y3_str >> pic_v3_str) {
    PicData<Dtype> meta;
    meta.image_path = image_path;
    meta.bgw = atoi(bgw_str.c_str());
    meta.bgh = atoi(bgh_str.c_str());
    meta.bbox.x1_ = atof(x1_str.c_str());
    meta.bbox.y1_ = atof(y1_str.c_str());
    meta.bbox.x2_ = atof(x2_str.c_str());
    meta.bbox.y2_ = atof(y2_str.c_str());
    meta.pic.push_back(atof(pic_x1_str.c_str()));
    meta.pic.push_back(atof(pic_y1_str.c_str()));
    meta.pic.push_back(atof(pic_v1_str.c_str()));
    meta.pic.push_back(atof(pic_x2_str.c_str()));
    meta.pic.push_back(atof(pic_y2_str.c_str()));
    meta.pic.push_back(atof(pic_v2_str.c_str()));
    meta.pic.push_back(atof(pic_x3_str.c_str()));
    meta.pic.push_back(atof(pic_y3_str.c_str()));
    meta.pic.push_back(atof(pic_v3_str.c_str()));
    metas->push_back(meta);
  }
}

INSTANTIATE_CLASS(PicVisualSaver);
}
