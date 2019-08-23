#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_MYIMG_PROC_H_
#define CAFFE_UTIL_MYIMG_PROC_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

/**
 * 不建议使用该文件中的方法。
 * 该文件已停止更新。
 */

namespace caffe {

  // grey-world, img
  void grayworld_awb_single(cv::Mat &src, cv::Mat &dst);
  // 通道增益处理
  void grayworld_awb(cv::Mat &src, cv::Mat &dst, float kb, float kg, float kr);
  // 求取awb增益
  void get_grayworld_gains(cv::Mat &src, float *kb, float *kg, float *kr);

  //dynamic awb
  void dynamic_awb(cv::Mat &src, cv::Mat &dst, float ratio);

  //sharp
  void sharp_2D(cv::Mat &src, cv::Mat &dst);

  template <typename Dtype>
  void blobTocvImage(const Dtype *data, const int height, const int width,
                     const int channels, cv::Mat *image);

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
