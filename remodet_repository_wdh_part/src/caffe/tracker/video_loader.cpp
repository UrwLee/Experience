#include "caffe/tracker/video_loader.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/image_proc.hpp"
#include "caffe/tracker/example_generator.hpp"

namespace caffe {

using std::string;
using std::vector;
namespace bfs = boost::filesystem;

template <typename Dtype>
VideoLoader<Dtype>::VideoLoader() {
}

template <typename Dtype>
void VideoLoader<Dtype>::ShowVideos() const {
  LOG(INFO) << "Showing " << videos_.size() << " videos.";
  for (int i = 0; i < videos_.size(); ++i) {
    const Video<Dtype>& video = videos_[i];
    LOG(INFO) << "Showing video " << i << ": "
              << video.path_;
    video.ShowVideo();
    cv::waitKey(0);
  }
}

template <typename Dtype>
void VideoLoader<Dtype>::ShowVideosShift() const {
  LOG(INFO) << "Showing " << videos_.size() << " videos.";
  for (int video_index = 0; video_index < videos_.size(); ++video_index) {
    const Video<Dtype>& video = videos_[video_index];
    const string& video_path = video.path_;
    LOG(INFO) << "Showing video " << video_index << ": " << video_path;
    // 获取标记
    const std::vector<Frame<Dtype> >& annotations = video.annotations_;
    // 前一帧的图片和box
    BoundingBox<Dtype> bbox_prev;
    cv::Mat image_prev;
    // 创建样本发生器
    ExampleGenerator<Dtype> example_generator((Dtype)1.0, (Dtype)5.0, (Dtype)(-0.4), (Dtype)0.4);
    // 遍历所有标注
    for (int frame_index = 0; frame_index < annotations.size(); ++frame_index) {
      // 获取frame
      const Frame<Dtype>& frame = annotations[frame_index];
      // 加载图像和box
      cv::Mat raw_image;
      BoundingBox<Dtype> bbox;
      const bool draw_bounding_box = false;
      const bool load_only_annotation = false;
      // 加载图片
      video.LoadFrame(frame.frame_num, draw_bounding_box, load_only_annotation,
                     &raw_image, &bbox);
      // 第一帧不做处理,直接保存为prev
      // 后续帧开始进行处理: 增广
      if (frame_index > 0) {
        // 先显示完整的图片和标记
        cv::Mat full_image_with_bbox;
        raw_image.copyTo(full_image_with_bbox);
        bbox.DrawBoundingBox(&full_image_with_bbox);
        cv::namedWindow("Raw image", cv::WINDOW_AUTOSIZE);// Create a window for display.
        cv::imshow("Raw image", full_image_with_bbox);                   // Show our image inside it.
        // 使用前一帧的box和当前帧的box
        // 以及前后两帧图像,复位样本发生器
        // 样本发生器设置视频ID和帧ID
        example_generator.Reset(bbox_prev, bbox, image_prev, raw_image);
        example_generator.set_indices(video_index, frame_index);
        // 生成样本
        cv::Mat image_rand_focus;
        cv::Mat target_pad;
        BoundingBox<Dtype> bbox_gt_scaled;
        // 可视化打开
        const bool visualize = true;
        // 只生成一个样本
        const int kNumGeneratedExamples = 1;
        // 样本生成
        for (int k = 0; k < kNumGeneratedExamples; ++k) {
          example_generator.MakeTrainingExampleBBShift(visualize, &image_rand_focus,
                                                       &target_pad, &bbox_gt_scaled);
        }
      }
      // 保存为prev
      bbox_prev = bbox;
      image_prev = raw_image;
    }
  }
}

template <typename Dtype>
void VideoLoader<Dtype>::merge_from(const VideoLoader<Dtype>* dst) {
  const std::vector<Video<Dtype> >& dst_videos = dst->get_videos();
  if (dst_videos.size() == 0) return;
  for (int i = 0; i < dst_videos.size(); ++i) {
    videos_.push_back(dst_videos[i]);
  }
  LOG(INFO) << "Add " << dst_videos.size() << " videos.";
}

INSTANTIATE_CLASS(VideoLoader);
}
