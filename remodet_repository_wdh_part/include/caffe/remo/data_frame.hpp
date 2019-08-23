#ifndef CAFFE_REMO_DATA_FRAME_H
#define CAFFE_REMO_DATA_FRAME_H

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类定义了一个数据帧的格式，包含其
 * （１）ID
 * （２）原始图像
 * （３）Resized图像，用于网络进行计算
 */

template <typename Dtype>
class DataFrame {
public:
  DataFrame();
  /**
   * 构造方法
   */
  DataFrame(int id, const cv::Mat& image, int resized_width, int resized_height);

  /**
   * 显示原始输入图像
   */
  void show_ori() const;

  /**
   * 显示Resized图像
   */
  void show_resized() const;

  /**
   * 获取帧ID
   * @return [ID]
   */
  int get_id() { return id_; }

  /**
   * 返回原始图像
   * @return [cv::Mat]
   */
  cv::Mat& get_ori_image() { return ori_image_; }

  /**
   * 返回resized图像
   * @return [cv::Mat]
   */
  cv::Mat& get_resized_image() { return resized_image_; }

  /**
   * 设置帧ID
   * @param id [设置的ID]
   */
  void set_id(int id) { id_ = id; }

  /**
   * 设置原始图像
   * @param image [输入的原始图像cv::Mat]
   */
  void set_ori_image(const cv::Mat& image) { ori_image_ = image; }

  /**
   * 设置resized图像
   * @param image [设置的resized图像cv::Mat]
   */
  void set_resized_image(const cv::Mat& image) { resized_image_ = image; }

protected:
  // 原始图像
  cv::Mat ori_image_;
  // resized图像
  cv::Mat resized_image_;
  // ID
  int id_;
};

}

#endif
