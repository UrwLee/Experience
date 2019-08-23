#ifndef CAFFE_TRACKER_TRACKER_TESTER_H
#define CAFFE_TRACKER_TRACKER_TESTER_H

#include "caffe/tracker/tracker_manager.hpp"

namespace caffe {

/**
 * Tracker测试器，提供对一般测试对象的跟踪。
 * 作为Tracker的benchmark存在。
 */

template <typename Dtype>
class TrackerTester : public TrackerManager<Dtype> {
public:
  /**
   * 构造方法：
   * videos -> 测试视频集合
   * regressor -> 回归器
   * tracker -> 跟踪器
   * show_tracking -> 是否可视化
   * save_tracking -> 是否保存跟踪结果
   * output_folder -> 结果文件的输出目录
   * save_folder -> 跟踪图片的输出目录
   */
  TrackerTester(const std::vector<Video<Dtype> >& videos,
                RegressorBase<Dtype>* regressor,
                TrackerBase<Dtype>* tracker,
                const bool show_tracking,
                const bool save_tracking,
                const std::string& output_folder,
                const std::string& save_folder);

  /**
   * 视频跟踪前的初始化
   * @param video     [视频]
   * @param video_num [编号]
   */
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
  // 结果文件的输出目录
  std::string output_folder_;
  // 可视化结果的输出目录
  std::string save_folder_;
  // 路径和名称
  std::string save_path_;
  std::string save_name_;
  // 输出文件指针
  FILE* output_file_ptr_;
  // 已处理帧数
  int num_frames_;
  // 可视化／保存控制
  bool show_tracking_;
  bool save_tracking_;
  // for each video
  int video_frames_;
  Dtype iou_sum_;
  // for all
  int video_frames_all_;
  Dtype iou_sum_all_;
};

}

#endif
