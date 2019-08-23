#ifndef CAFFE_TRACKER_TRACKER_MANAGER_H
#define CAFFE_TRACKER_TRACKER_MANAGER_H

#include "caffe/tracker/regressor.hpp"
#include "caffe/tracker/tracker_base.hpp"
#include "caffe/tracker/video.hpp"

namespace caffe {

/**
 * 该类为实际测试Tracker的基类
 * 提供了一系列Tracker测试的公有方法。
 * 包括：
 * １．对所有视频序列进行跟踪；
 * ２．视频跟踪前的初始化行为；
 * ３．网络跟踪前的准备行为；
 * ４．每一帧跟踪结束后的行为；
 * ５．每个视频跟踪结束后的行为；
 * ６．所有视频跟踪结束后的行为；
 * 注意：默认上述所有行为均为Do Nothing.
 */

template <typename Dtype>
class TrackerManager {
public:
  /**
   * 构造方法：
   * videos: -> 跟踪视频集合
   * regressor: -> 回归器
   * tracker: -> Tracker
   */
  TrackerManager(const std::vector<Video<Dtype> >& videos,
                 RegressorBase<Dtype>* regressor, TrackerBase<Dtype>* tracker);

  // 跟踪所有视频
  void TrackAll() ;

  // 从第几个视频开始跟踪
  void TrackAll(const int start_video_num, const int pause_val);

  /**
   * 视频跟踪前的初始化行为
   * @param video     [视频]
   * @param video_num [视频编号]
   */
  virtual void VideoInit(const Video<Dtype>& video, const int video_num) {}

  /**
   * 网络估计前的准备行为
   */
  virtual void SetupEstimate() {}

  /**
   * 每帧跟踪结束后的行为
   * @param frame_num                [帧编号]
   * @param image_curr               [当真帧]
   * @param has_annotation           [是否有GT标记]
   * @param bbox_gt                  [box的gt位置]
   * @param bbox_estimate_uncentered [box的估计位置]
   * @param pause_val                [暂停键值]
   */
  virtual void ProcessTrackOutput(
      const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
      const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate_uncentered,
      const int pause_val) {}

  // 视频跟踪结束后的行为
  virtual void PostProcessVideo() {}

  // 所有视频跟踪结束后的行为
  virtual void PostProcessAll() {}

protected:

  /**
   * 视频集合
   */
  const std::vector<Video<Dtype> >& videos_;

  /**
   * 回归器
   */
  RegressorBase<Dtype>* regressor_;

  /**
   * Tracker
   */
  TrackerBase<Dtype>* tracker_;
};

}

#endif
