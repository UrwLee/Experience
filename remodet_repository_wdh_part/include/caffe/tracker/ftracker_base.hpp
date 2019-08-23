#ifndef CAFFE_TRACKER_FTRACKER_BASE_H
#define CAFFE_TRACKER_FTRACKER_BASE_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/shared_ptr.hpp>
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/fregressor_base.hpp"
#include "caffe/tracker/fe_roi_maker.hpp"
#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类是FTracker的基类。
 * 该类提供了Tracker的通用跟踪方法。
 */

template <typename Dtype>
class FTrackerBase {
public:

  /**
   * 构造方法：
   * network_proto/caffe_model :-> 特征抽取网络和权值文件
   * gpu_id: -> GPU ID
   * features: -> 特征抽取的特证名
   * resized_width/resized_height: -> ROI-Resized尺寸
   */
  FTrackerBase(const std::string& network_proto,
               const std::string& caffe_model,
               const int gpu_id,
               const std::string& features,
               const int resized_width,
               const int resized_height);

  // 没有特征抽取器的构造
  FTrackerBase(const int resized_width, const int resized_height);

  /**
   * 跟踪方法
   * @param image_curr               [当前图像]
   * @param freg                     [F回归器]
   * @param bbox_estimate_uncentered [跟踪结果]
   */
  virtual void Tracking(const cv::Mat& image_curr, FRegressorBase<Dtype>* freg,
                        BoundingBox<Dtype>* bbox_estimate_uncentered);
  // 不适用特征抽取器的跟踪方法: unused.
  virtual void Tracking(const Blob<Dtype>& fmap, FRegressorBase<Dtype>* freg,
                        BoundingBox<Dtype>* bbox_estimate_uncentered);
  // 初始化
  void Init(const cv::Mat& image_curr, const BoundingBox<Dtype>& bbox_gt);
  // 不使用特征抽取器：　unused.
  void Init(const Blob<Dtype>& fmap, const BoundingBox<Dtype>& bbox_gt);

private:
  // 特征抽取器
  boost::shared_ptr<FERoiMaker<Dtype> > roi_maker_;
  // roi-resize层
  boost::shared_ptr<caffe::Layer<Dtype> > roi_resize_layer_;
  // 不考虑运动模型，当前帧的初始位置默认与上一帧的位置相同
  BoundingBox<Dtype> bbox_curr_init_;
  // 过去帧的位置
  BoundingBox<Dtype> bbox_prev_;
  // 过去帧的输入特征Blob
  Blob<Dtype> prev_fmap_;
  // 是否使用basenet,默认使用
  bool use_basenet_;
};

}

#endif
