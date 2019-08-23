#ifndef CAFFE_TRACKER_TRACKER_BASE_H
#define CAFFE_TRACKER_TRACKER_BASE_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/regressor_base.hpp"

namespace caffe {

/**
 * 该类是Tracker对象的基类。
 * 该类提供了Tracking方法，为所有派生类共有的方法。
 */

template <typename Dtype>
class TrackerBase {
public:
  /**
   * 构造方法：
   * show_tracking：->　是否可视化跟踪结果
   */
  TrackerBase(const bool show_tracking): show_tracking_(show_tracking) {}

  /**
   * Tracking方法
   * @param image_curr               [当前输入帧]
   * @param reg                      [回归器：用于进行网络计算]
   * @param bbox_estimate_uncentered [返回结果]
   */
  virtual void Tracking(const cv::Mat& image_curr, RegressorBase<Dtype>* reg,
             BoundingBox<Dtype>* bbox_estimate_uncentered);

  /**
   * 初始化方法
   * @param image_curr [指定初始帧]
   * @param bbox_gt    [指定输出位置]
   */
  void Init(const cv::Mat& image_curr, const BoundingBox<Dtype>& bbox_gt);

private:
  /**
   * 可视化跟踪结果
   * @param target_pad         [历史的ROI-Patch]
   * @param curr_search_region [当前的ROI-Patch]
   * @param bbox_estimate      [回归器直接回归的位置，相当于在ROI-Patch中位置]
   */
  void ShowTracking(const cv::Mat& target_pad, const cv::Mat& curr_search_region, const BoundingBox<Dtype>& bbox_estimate) const;

  /**
   * bbox_curr_init_ : -> 当前帧初始的位置
   * bbox_prev_ : -> 过去帧的位置
   */
  BoundingBox<Dtype> bbox_curr_init_;
  BoundingBox<Dtype> bbox_prev_;

  /**
   * image_prev_: -> 过去帧的图片
   */
  cv::Mat image_prev_;

  // 是否显示Patch上的可视化过程
  bool show_tracking_;
};

}

#endif
