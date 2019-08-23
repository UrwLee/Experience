#ifndef CAFFE_TRACKER_FTRACKER_VIDEO_H
#define CAFFE_TRACKER_FTRACKER_VIDEO_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/fregressor.hpp"
#include "caffe/tracker/ftracker_base.hpp"
#include "caffe/tracker/video.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供对视频或摄像头的FTracker测试
 */

template <typename Dtype>
class FVideoTracker {
public:
  /**
   * 构造方法
   * param: -> 构造参数
   * fregressor: ->　F回归器
   * ftracker: -> F跟踪器
   */
  FVideoTracker(const VideoTrackerParameter& param,
               FRegressorBase<Dtype>* fregressor,
               FTrackerBase<Dtype>* ftracker);

  /**
   * 释放数据流
   */
  virtual ~FVideoTracker() {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

  /**
   * 跟踪方法
   */
  void Tracking() ;

  /**
   * 每一帧跟踪结束后的工作
   * @param frame_num      [帧ID]
   * @param image_curr     [当前帧]
   * @param bbox_estimated [估计位置]
   */
  void ProcessTrackOutput(const int frame_num, const cv::Mat& image_curr, const BoundingBox<Dtype>& bbox_estimated);

  /**
   * 视频跟踪结束后的行为
   */
  void PostProcessAll();

protected:
  // 构造参数
  VideoTrackerParameter param_;
  // F回归器
  FRegressorBase<Dtype>* fregressor_;
  // F跟踪器
  FTrackerBase<Dtype>* ftracker_;
  // 视频输入流
  cv::VideoCapture cap_;
  // 视频总帧数
  int total_frames_;
  // 已处理帧数
  int processed_frames_;
  // 视频初始跳过的帧数
  int initial_frame_;
  // 类型: 摄像头或视频
  bool is_type_video_;
  // 输出视频流
  cv::VideoWriter video_writer_;
};

}

#endif
