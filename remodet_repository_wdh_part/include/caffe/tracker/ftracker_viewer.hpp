#ifndef CAFFE_TRACKER_FTRACKER_VIEWER_H
#define CAFFE_TRACKER_FTRACKER_VIEWER_H

#include "caffe/tracker/ftracker_manager.hpp"

namespace caffe {

/**
 * 基于FTracker: 对测试视频的可视化类
 */

template <typename Dtype>
class FTrackerViewer : public FTrackerManager<Dtype> {
public:
  /**
   * 构造方法：
   * videos: 视频列表
   * fregressor:　F回归器
   * ftracker:　F跟踪器
   * save_videos:　是否保存跟踪视频
   * save_outputs:　是否保存跟踪结果
   * output_folder:　输出路径
   */
  FTrackerViewer(const std::vector<Video<Dtype> >& videos,
                FRegressorBase<Dtype>* fregressor,
                FTrackerBase<Dtype>* ftracker,
                const bool save_videos,
                const bool save_outputs,
                const std::string& output_folder);
  // 视频tracking前的工作： 例如打开视频流，打开保存文件指针
  virtual void VideoInit(const Video<Dtype>& video, const int video_num);
  // 每帧结束后的处理工作
  virtual void ProcessTrackOutput(
      const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
      const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate,
      const int pause_val);
  // 每个视频结束后的处理工作
  virtual void PostProcessVideo();
  // 所有视频结束后的处理工作
  virtual void PostProcessAll();

private:
  // 保存目录
  std::string output_folder_;
  // 输出文件指针
  FILE* output_file_ptr_;
  // 已处理帧数
  int num_frames_;
  // 输出视频流
  cv::VideoWriter video_writer_;
  // 输出控制字
  bool save_videos_;
  bool save_outputs_;
};

}

#endif
