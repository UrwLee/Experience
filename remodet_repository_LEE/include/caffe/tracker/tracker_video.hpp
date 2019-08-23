#ifndef CAFFE_TRACKER_TRACKER_VIDEO_H
#define CAFFE_TRACKER_TRACKER_VIDEO_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/regressor.hpp"
#include "caffe/tracker/tracker_base.hpp"
#include "caffe/tracker/video.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 本类提供了对一个通用视频的跟踪。
 * 支持对象：
 * １．网络摄像头
 * ２．视频
 */

template <typename Dtype>
class VideoTracker {
public:
  /**
   * 构造方法：
   * １．跟踪参数
   * ２．回归器
   * ３．跟踪器
   */
  VideoTracker(const VideoTrackerParameter& param,
               RegressorBase<Dtype>* regressor,
               TrackerBase<Dtype>* tracker);

  /**
   * 析构方法：关闭视频流
   */
  virtual ~VideoTracker() {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

  /**
   * 跟踪方法
   */
  void Tracking() ;

  /**
   * 每一帧跟踪结束后的处理过程
   * @param frame_num      [帧编号]
   * @param image_curr     [当前帧]
   * @param bbox_estimated [估计位置]
   */
  void ProcessTrackOutput(const int frame_num, const cv::Mat& image_curr, const BoundingBox<Dtype>& bbox_estimated);

  // 视频处理结束后的打印消息
  void PostProcessAll();

protected:
  // 参数设置
  VideoTrackerParameter param_;
  // 回归器
  RegressorBase<Dtype>* regressor_;
  // 跟踪器
  TrackerBase<Dtype>* tracker_;
  // 视频输入流
  cv::VideoCapture cap_;
  // 总帧数：针对视频
  int total_frames_;
  // 已处理帧数
  int processed_frames_;
  // 初始跳过帧数：针对视频
  int initial_frame_;
  // 类型：摄像头或视频
  bool is_type_video_;
  // 输出视频流
  cv::VideoWriter video_writer_;
};

}

#endif
