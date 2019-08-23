#ifndef CAFFE_TRACKER_FTRACKER_TESTER_H
#define CAFFE_TRACKER_FTRACKER_TESTER_H

#include "caffe/tracker/ftracker_manager.hpp"

namespace caffe {

/**
 * FTracker的测试类
 * 提供对测试数据集的跟踪性能测试
 */

template <typename Dtype>
class FTrackerTester : public FTrackerManager<Dtype> {
public:
  /**
   * videos: 测试视频集合
   * fregressor: F回归器
   * ftracker: F跟踪器
   * show_tracking: 是否可视化
   * output_folder: 结果输出路径
   */
  FTrackerTester(const std::vector<Video<Dtype> >& videos,
                FRegressorBase<Dtype>* fregressor,
                FTrackerBase<Dtype>* ftracker,
                const bool show_tracking,
                const std::string& output_folder);
  // 视频tracking前的工作： 打开保存结果的文件指针
  virtual void VideoInit(const Video<Dtype>& video, const int video_num);
  // 每帧结束后的处理工作： 写每一帧的保存结果
  virtual void ProcessTrackOutput(
      const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
      const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate,
      const int pause_val);
  // 每个视频结束后的处理工作： 写视频的平均精度
  virtual void PostProcessVideo();
  // 所有视频结束后的处理工作： 写所有视频的平均精度
  virtual void PostProcessAll();

private:
  // 保存目录
  std::string output_folder_;
  // 输出文件指针
  FILE* output_file_ptr_;
  // 已处理帧数
  int num_frames_;
  bool show_tracking_;
  // for each video
  int video_frames_;
  Dtype iou_sum_;
  // for all
  int video_frames_all_;
  Dtype iou_sum_all_;
};

}

#endif
