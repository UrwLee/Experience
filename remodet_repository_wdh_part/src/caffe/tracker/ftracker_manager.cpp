#include "caffe/tracker/ftracker_manager.hpp"

#include <string>

#include "caffe/tracker/basic.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
FTrackerManager<Dtype>::FTrackerManager(const std::vector<Video<Dtype> >& videos,
                FRegressorBase<Dtype>* fregressor, FTrackerBase<Dtype>* ftracker)
  : videos_(videos), fregressor_(fregressor), ftracker_(ftracker) {
}

template <typename Dtype>
void FTrackerManager<Dtype>::TrackAll() {
  TrackAll(0, 1);
}

template <typename Dtype>
void FTrackerManager<Dtype>::TrackAll(const int start_video_num, const int pause_val) {
  // 遍历迭代每个视频序列
  for (int video_num = start_video_num; video_num < videos_.size(); ++video_num) {
    const Video<Dtype>& video = videos_[video_num];
    VideoInit(video, video_num);

    int first_frame;
    cv::Mat image_curr;
    BoundingBox<Dtype> bbox_gt;
    // 获取第一帧的结果: 图片和box
    video.LoadFirstAnnotation(&first_frame, &image_curr, &bbox_gt);
    // 初始化Tracker
    ftracker_->Init(image_curr, bbox_gt);
    // 跟踪下面的所有帧
    for (int frame_num = first_frame + 1; frame_num < video.all_frames_.size(); ++frame_num) {
      const bool draw_bounding_box = false;
      const bool load_only_annotation = false;
      // 获取当前的图像和gt
      cv::Mat image_curr;
      BoundingBox<Dtype> bbox_gt;
      bool has_annotation = video.LoadFrame(frame_num,
                                            draw_bounding_box,
                                            load_only_annotation,
                                            &image_curr, &bbox_gt);
      // 在Track前的setup工作: do nothing
      SetupEstimate();
      BoundingBox<Dtype> bbox_estimate_uncentered;
      // 使用当前图像进行估计
      ftracker_->Tracking(image_curr, fregressor_, &bbox_estimate_uncentered);
      ProcessTrackOutput(frame_num, image_curr, has_annotation, bbox_gt,
                         bbox_estimate_uncentered, pause_val);
    }
    PostProcessVideo();
  }
  PostProcessAll();
}

INSTANTIATE_CLASS(FTrackerManager);

}
