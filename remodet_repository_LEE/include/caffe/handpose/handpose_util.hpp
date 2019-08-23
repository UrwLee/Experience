#ifndef __HANDPOSE_UTIL_HPP_
#define __HANDPOSE_UTIL_HPP_

#include "caffe/caffe.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/handpose/bbox.hpp"
#include "caffe/handpose/handpose_instance.hpp"

namespace caffe {

void parseHandPoseLine(const std::string& line, const std::string& root_dir, caffe::Phase phase,
                   HandPoseInstance* ins);

}
#endif
