#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/handpose/handpose_util.hpp"

using namespace std;

namespace caffe {

// 读取一行标注记录
void parseHandPoseLine(const std::string& line, const std::string& root_dir, caffe::Phase phase,
                   HandPoseInstance* ins) {
  stringstream ss;
  ss.clear();
  ss.str(line);
  string path;
  int xmin, ymin, xmax, ymax;           // box
  int id;
  int is_rotated;
  ss >> path >> xmin >> ymin >> xmax >> ymax >> id >> is_rotated;
  //LOG(INFO)<<path<<" "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<id<<" "<<" "<<is_rotated;
  string image_path = root_dir + '/' + path;
  ins->path_ = image_path;
  ins->id_ = id;
  ins->is_rotated_ = is_rotated;
  ins->box_ = BBox(xmin, ymin, xmax, ymax);
}

}
