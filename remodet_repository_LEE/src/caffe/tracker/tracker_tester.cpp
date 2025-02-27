#include "caffe/tracker/tracker_tester.hpp"

#include <string>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "caffe/tracker/basic.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
TrackerTester<Dtype>::TrackerTester(const std::vector<Video<Dtype> >& videos,
                                    RegressorBase<Dtype>* regressor,
                                    TrackerBase<Dtype>* tracker,
                                    const bool show_tracking,
                                    const bool save_tracking,
                                    const std::string& output_folder,
                                    const std::string& save_folder)
 : TrackerManager<Dtype>(videos,regressor,tracker), output_folder_(output_folder), save_folder_(save_folder), num_frames_(0), show_tracking_(show_tracking) ,save_tracking_(save_tracking){
 video_frames_ = 0;
 iou_sum_ = 0;
 video_frames_all_ = 0;
 iou_sum_all_ = 0;
}

template <typename Dtype>
void TrackerTester<Dtype>::VideoInit(const Video<Dtype>& video, const int video_num) {
  int delim_pos = video.path_.find_last_of("/");
  const string& video_name = video.path_.substr(delim_pos+1, video.path_.length());
  LOG(INFO) << "Video " << video_num << ": " << video_name;
  // 结果文件指针
  const string& output_file = output_folder_ + "/" + video_name;
  output_file_ptr_ = fopen(output_file.c_str(), "w");
  if(access(save_path_.c_str(),0)==-1)
    mkdir(save_path_.c_str(),S_IRWXU);
  // 视频统计结果初始化
  video_frames_ = 0;
  iou_sum_ = 0;
}

template <typename Dtype>
void TrackerTester<Dtype>::ProcessTrackOutput(
    const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate_uncentered,
    const int pause_val) {
  num_frames_++;
  Dtype iou;
  if (has_annotation) {
    iou = bbox_estimate_uncentered.compute_iou(bbox_gt);
    video_frames_++;
    iou_sum_ += iou;
  } else {
    iou = 0;
  }
  // 写结果
  fprintf(output_file_ptr_, "%d, %.3f\n", frame_num, (float)iou);
  // 可视化
  if (show_tracking_) {
    cv::Mat full_output;
    image_curr.copyTo(full_output);
    if (has_annotation) {
      bbox_gt.DrawBoundingBox(&full_output);
    }
    bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", full_output);
    if(save_tracking_)
    {
      save_name_ = save_path_ + "/" + caffe::format_int(frame_num) + ".jpg";
      cv::imwrite(save_name_, full_output);
    }
    cv::waitKey(pause_val);
  }
}

template <typename Dtype>
void TrackerTester<Dtype>::PostProcessVideo() {
  // 统计整个视频结果
  Dtype iou_avg = iou_sum_ / video_frames_;
  LOG(INFO) << "[Test Result] Found " << video_frames_ << " annotated frames, "
            << "The avg IOU: " << iou_avg;
  fprintf(output_file_ptr_, "Has found %d annotated frames, with avg_iou is %.3f\n", video_frames_, (float)iou_avg);
  fclose(output_file_ptr_);
  // 统计整个视频结果
  video_frames_all_ += video_frames_;
  iou_sum_all_ += iou_sum_;
}

template <typename Dtype>
void TrackerTester<Dtype>::PostProcessAll() {
  Dtype iou_avg_all = iou_sum_all_ / video_frames_all_;
  LOG(INFO) << "Finished tracking " << this->videos_.size() << " videos.";
  LOG(INFO) << "[Test All] Found " << video_frames_all_ << " annotated frames in "
            << this->videos_.size() << " videos. We get an averaged IOU of " << iou_avg_all;
}

INSTANTIATE_CLASS(TrackerTester);

}
