#ifndef CAFFE_TRACKER_FTRACKER_MANAGER_H
#define CAFFE_TRACKER_FTRACKER_MANAGER_H

#include "caffe/tracker/fregressor.hpp"
#include "caffe/tracker/ftracker_base.hpp"
#include "caffe/tracker/video.hpp"

namespace caffe {

/**
 * FTracker的测试基类。
 * 它提供了FTracker所有必须的方法。
 * 包括：
 * １．跟踪方法
 * ２．视频跟踪前的初始化方法
 * ３．每一帧跟踪前的准备过程
 * ４．每一帧跟踪完的行为
 * ５．视频跟踪结束后的行为
 * ６．所有视频跟踪结束后的行为
 */

template <typename Dtype>
class FTrackerManager {
public:
  /**
   * 构造：
   * videos:　视频列表
   * fregressor:　F回归器
   * ftracker: F跟踪器
   */
  FTrackerManager(const std::vector<Video<Dtype> >& videos,
                  FRegressorBase<Dtype>* fregressor, FTrackerBase<Dtype>* ftracker);

  /**
   * 跟踪所有视频
   */
  void TrackAll() ;

  /**
   * 跟踪所有视频
   * @param start_video_num [视频编号]
   * @param pause_val       [暂停]
   */
  void TrackAll(const int start_video_num, const int pause_val);

  /**
   * 视频跟踪前的初始化工作
   * @param video     [视频]
   * @param video_num [编号]
   */
  virtual void VideoInit(const Video<Dtype>& video, const int video_num) {}

  /**
   * 每一帧跟踪开始前的工作
   */
  virtual void SetupEstimate() {}

  /**
   * 每一帧跟踪结束后的行为
   * @param frame_num                [帧ID]
   * @param image_curr               [当前帧图像]
   * @param has_annotation           [是否有标注GT]
   * @param bbox_gt                  [gt-box]
   * @param bbox_estimate_uncentered [估计的位置]
   * @param pause_val                [暂停键值]
   */
  virtual void ProcessTrackOutput(
      const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
      const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate_uncentered,
      const int pause_val) {}

  // 视频跟踪完成后的行为
  virtual void PostProcessVideo() {}

  // 所有视频跟踪完成后的行为
  virtual void PostProcessAll() {}

protected:
  //　视频列表
  const std::vector<Video<Dtype> >& videos_;

  // F回归器
  FRegressorBase<Dtype>* fregressor_;

  // F跟踪器
  FTrackerBase<Dtype>* ftracker_;
};

}

#endif
