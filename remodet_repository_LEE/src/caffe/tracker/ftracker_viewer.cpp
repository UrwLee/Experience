#include "caffe/tracker/ftracker_viewer.hpp"

#include <string>

#include "caffe/tracker/basic.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
FTrackerViewer<Dtype>::FTrackerViewer(const std::vector<Video<Dtype> >& videos,
                                    FRegressorBase<Dtype>* fregressor,
                                    FTrackerBase<Dtype>* ftracker,
                                    const bool save_videos,
                                    const bool save_outputs,
                                    const std::string& output_folder)
 : FTrackerManager<Dtype>(videos,fregressor,ftracker), output_folder_(output_folder), num_frames_(0), save_videos_(save_videos), save_outputs_(save_outputs) {
}

template <typename Dtype>
void FTrackerViewer<Dtype>::VideoInit(const Video<Dtype>& video, const int video_num) {
  int delim_pos = video.path_.find_last_of("/");
  const string& video_name = video.path_.substr(delim_pos+1, video.path_.length());
  LOG(INFO) << "Video " << video_num << ": " << video_name;
  if (save_outputs_) {
    const string& output_file = output_folder_ + "/" + video_name;
    output_file_ptr_ = fopen(output_file.c_str(), "w");
  }
  if (save_videos_) {
    const string& video_out_folder = output_folder_ + "/videos";
    boost::filesystem::create_directories(video_out_folder);
    cv::Mat image;
    BoundingBox<Dtype> box;
    video.LoadFrame(0, false, false, &image, &box);
    const string video_out_name = video_out_folder + "/Video" + num2str(static_cast<int>(video_num)) + ".avi";
    video_writer_.open(video_out_name, CV_FOURCC('M','J','P','G'), 50, image.size());
  }
}

template <typename Dtype>
void FTrackerViewer<Dtype>::ProcessTrackOutput(
    const int frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox<Dtype>& bbox_gt, const BoundingBox<Dtype>& bbox_estimate_uncentered,
    const int pause_val) {
  num_frames_++;
  const Dtype width = fabs(bbox_estimate_uncentered.get_width());
  const Dtype height = fabs(bbox_estimate_uncentered.get_height());
  const Dtype x_min = std::min(bbox_estimate_uncentered.x1_, bbox_estimate_uncentered.x2_);
  const Dtype y_min = std::min(bbox_estimate_uncentered.y1_, bbox_estimate_uncentered.y2_);
  if (save_outputs_) {
    fprintf(output_file_ptr_, "%d %f %f %f %f\n", frame_num + 1, (float)x_min, (float)y_min, (float)width, (float)height);
  }
  cv::Mat full_output;
  image_curr.copyTo(full_output);
  if (has_annotation) {
    bbox_gt.DrawBoundingBox(&full_output);
  }
  bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);
  if (save_videos_) {
    video_writer_.write(full_output);
  }
  cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
  cv::imshow("Output", full_output);
  cv::waitKey(pause_val);
}

template <typename Dtype>
void FTrackerViewer<Dtype>::PostProcessVideo() {
  if (save_outputs_) {
    fclose(output_file_ptr_);
  }
}

template <typename Dtype>
void FTrackerViewer<Dtype>::PostProcessAll() {
  LOG(INFO) << "Finished tracking " << this->videos_.size() << " videos.";
}

INSTANTIATE_CLASS(FTrackerViewer);

}
