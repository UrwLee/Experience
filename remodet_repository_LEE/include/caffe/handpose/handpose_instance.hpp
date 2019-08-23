#ifndef __HANDPOSE_INSTANCE_HPP_
#define __HANDPOSE_INSTANCE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "gflags/gflags.h"
#include "glog/logging.h"
// include box
#include "caffe/handpose/bbox.hpp"

namespace caffe {

class HandPoseInstance {
  public:
    // 构造
    HandPoseInstance() {}
    // 成员
    string path_;  // image path
    int id_;       // instance id
    int is_rotated_;
    BBox box_;  // roi

};
}
#endif
