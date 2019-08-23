#include "caffe/remo/frame_reader.hpp"

#include <csignal>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
FrameReader<Dtype>::FrameReader(int cam_id, int width, int height, int resized_width, int resized_height) {
  type_video_ = 1;
  if (!cap_.open(cam_id)) {
    LOG(FATAL) << "Failed to open webcam: " << cam_id;
  }
  cap_.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap_.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  cv::Mat cv_img;
  cap_ >> cv_img;
  CHECK(cv_img.data) << "Could not load image.";
  initial_frame_ = 0;
  processed_frames_ = 0;
  resized_width_ = resized_width;
  resized_height_ = resized_height;
}

template <typename Dtype>
FrameReader<Dtype>::FrameReader(const std::string& video_file, int start_frame, int resized_width, int resized_height) {
  type_video_ = 0;
  if (!cap_.open(video_file)) {
    LOG(FATAL) << "Failed to open video: " << video_file;
  }
  total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
  cap_.set(CV_CAP_PROP_POS_FRAMES, start_frame);
  processed_frames_ = start_frame + 1;
  initial_frame_ = start_frame;
  cv::Mat cv_img;
  cap_ >> cv_img;
  CHECK(cv_img.data) << "Could not load image.";
  resized_width_ = resized_width;
  resized_height_ = resized_height;
}

template <typename Dtype>
FrameReader<Dtype>::FrameReader(const std::string& ip_addr, int resized_width, int resized_height) {
  type_video_ = 2;
  if (!cap_.open(ip_addr)) {
    LOG(FATAL) << "Failed to open stream-server: " << ip_addr;
  }
  cv::Mat cv_img;
  cap_ >> cv_img;
  CHECK(cv_img.data) << "Could not load image.";
  initial_frame_ = 0;
  processed_frames_ = 0;
  resized_width_ = resized_width;
  resized_height_ = resized_height;
}

template <typename Dtype>
int FrameReader<Dtype>::pop(DataFrame<Dtype>* frame) {
  cv::Mat cv_img;
  // use cap_
  if (type_video_ == 0) {
    if (processed_frames_ >= total_frames_) {
      return 1;
    }
    cap_ >> cv_img;
  } else {
    cap_ >> cv_img;
    // -----------------------------------------------------------------------
    // color - contrast & brightness
    float alpha = 1.0;
    float beta = 0;
    for (int i = 0; i < cv_img.rows; ++i) {
      for (int j = 0; j < cv_img.cols; ++j) {
        for (int c = 0; c < 3; ++c) {
          cv_img.at<cv::Vec3b>(i,j)[c] = cv::saturate_cast<unsigned char>(alpha * cv_img.at<cv::Vec3b>(i,j)[c] + beta);
        }
      }
    }
    // -----------------------------------------------------------------------
  }
  DataFrame<Dtype> dframe(processed_frames_, cv_img, resized_width_, resized_height_);
  *frame = dframe;
  ++processed_frames_;
  return 0;
}

template <typename Dtype>
void FrameReader<Dtype>::show() {
  while(1) {
    cv::Mat cv_img;
    if (type_video_ == 0) {
      if (processed_frames_ >= total_frames_) {
        LOG(FATAL) << "Video has been finished for processing: " << total_frames_ << " frames.";
      }
      cap_ >> cv_img;
    } else {
      cap_ >> cv_img;
    }
    ++processed_frames_;
    cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
    cv::imshow( "Remo", cv_img);
    cv::waitKey(1);
  }
}

INSTANTIATE_CLASS(FrameReader);
}
