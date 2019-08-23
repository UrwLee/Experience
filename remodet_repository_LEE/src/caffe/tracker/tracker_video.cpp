#include "caffe/tracker/tracker_video.hpp"

#include <string>

#include "caffe/tracker/basic.hpp"

namespace caffe {

using namespace cv;
using namespace std;

// 截取ROI的全局变量
static Rect select_roi;
// 鼠标左键是否被按下
static bool mousedown_flag = false;
// box是否已经选取
static bool select_flag = false;
// 左上角的点
static Point origin;

static std::string windowName = "REMO Tracker";

void onMouse(int event, int x, int y, int, void* param) {
  cv::Mat* image = (cv::Mat*)param;
  // 鼠标左键已经按下： 计算现在所选择的box
  if (mousedown_flag) {
    select_roi.x = std::min(origin.x, x);
    select_roi.y = std::min(origin.y, y);
    select_roi.width = abs(x - origin.x);
    select_roi.height = abs(y - origin.y);
    select_roi &= Rect(0, 0, image->cols, image->rows);
  }
  // 左键按下：
  if (event==CV_EVENT_LBUTTONDOWN) {
    mousedown_flag = true;
    select_flag = false;
    origin = Point(x, y);
    select_roi = Rect(x, y, 0, 0);
  } else if (event==CV_EVENT_LBUTTONUP) {
    mousedown_flag = false;
    select_flag = true;
  }
  // 实时绘制box
  if (mousedown_flag || select_flag) {
    BoundingBox<float> box;
    box.x1_ = select_roi.x;
    box.y1_ = select_roi.y;
    box.x2_ = select_roi.x + select_roi.width;
    box.y2_ = select_roi.y + select_roi.height;
    cv::Mat temp;
    image->copyTo(temp);
    box.Draw(255, 0, 0, &temp);
    cv::imshow(windowName, temp);
  }
}

template <typename Dtype>
VideoTracker<Dtype>::VideoTracker(const VideoTrackerParameter& param,
                                   RegressorBase<Dtype>* regressor,
                                   TrackerBase<Dtype>* tracker)
  : param_(param),regressor_(regressor), tracker_(tracker) {
  // 视频类型：VIDEO & WEBCAM
  is_type_video_ = param_.is_type_video();
  // 视频
  cv::Mat cv_img;
  if (is_type_video_) {
    CHECK(param_.has_video_file()) << "Must provide video file!";
    const string& video_file = param_.video_file();
    initial_frame_ = param_.initial_frame();
    if (!cap_.open(video_file)) {
      LOG(FATAL) << "Failed to open video: " << video_file;
    }
    // 帧数量统计
    total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    cap_.set(CV_CAP_PROP_POS_FRAMES, initial_frame_);
    processed_frames_ = initial_frame_ + 1;
    cap_ >> cv_img;
  } else {
    const int webcam_width = param_.webcam_width();
    const int webcam_height = param_.webcam_height();
    const int device_id = param_.device_id();
    if (!cap_.open(device_id)) {
      LOG(FATAL) << "Failed to open webcam: " << device_id;
    }
    cap_.set(CV_CAP_PROP_FRAME_WIDTH, webcam_width);
    cap_.set(CV_CAP_PROP_FRAME_HEIGHT, webcam_height);
    cap_ >> cv_img;
  }
  CHECK(cv_img.data) << "Could not load image!";
  // 使用鼠标获取box
  cv::namedWindow(windowName);
  cv::setMouseCallback(windowName, onMouse, (void*)&cv_img);
  cv::imshow(windowName, cv_img);
  cv::waitKey(0);
  // 等待box选择结束
  LOG(INFO) << "Select the region of interest.";
  while(!select_flag);
  LOG(INFO) << "ROI has been selected, top-left(x,y): " << select_roi.x << ", " << select_roi.y
            << ", width-height(w,h): " << select_roi.width << ", " << select_roi.height;
  BoundingBox<Dtype> box;
  box.x1_ = select_roi.x;
  box.y1_ = select_roi.y;
  box.x2_ = select_roi.x + select_roi.width;
  box.y2_ = select_roi.y + select_roi.height;
  // 复位Tracker
  tracker_->Init(cv_img, box);
  LOG(INFO) << "The tracker has been initialized.";
  // 初始化保存
  if (param_.save_videos()) {
    CHECK(param_.has_output_folder());
    if (is_type_video_) {
      const string& video_file = param_.video_file();
      int delim_pos = video_file.find_last_of("/");
      const string& video_name = video_file.substr(delim_pos+1, video_file.length());
      LOG(INFO) << "Saving VideoTracker Results for: " << video_name;
      const string& video_out_folder = param_.output_folder() + "/VTResults_Video";
      boost::filesystem::create_directories(video_out_folder);
      const string video_out_name = video_out_folder + "/" + video_name + ".avi";
      video_writer_.open(video_out_name, CV_FOURCC('M','J','P','G'), 50, cv_img.size());
    } else {
      const string& video_out_folder = param_.output_folder() + "/VTResults_Webcam";
      boost::filesystem::create_directories(video_out_folder);
      const string video_out_name = video_out_folder + "/Webcam_" + num2str(static_cast<int>(param_.device_id())) + ".avi";
      video_writer_.open(video_out_name, CV_FOURCC('M','J','P','G'), 50, cv_img.size());
    }
  }
}

// Tracking
template <typename Dtype>
void VideoTracker<Dtype>::Tracking() {
  while(1) {
    cv::Mat cv_img;
    if (is_type_video_) {
      if (processed_frames_ >= total_frames_) {
        break;
      }
      ++processed_frames_;
      cap_ >> cv_img;
    } else {
      cap_ >> cv_img;
    }
    BoundingBox<Dtype> bbox_estimated;
    tracker_->Tracking(cv_img, regressor_, &bbox_estimated);
    ProcessTrackOutput(processed_frames_, cv_img, bbox_estimated);
  }
  PostProcessAll();
}

template <typename Dtype>
void VideoTracker<Dtype>::ProcessTrackOutput(
    const int frame_num, const cv::Mat& image_curr, const BoundingBox<Dtype>& bbox_estimated) {
  cv::Mat full_output;
  image_curr.copyTo(full_output);
  bbox_estimated.Draw(255, 0, 0, &full_output);
  if (param_.save_videos()) {
    video_writer_.write(full_output);
  }
  // display
  cv::imshow(windowName, full_output);
  cv::waitKey(1);
}

template <typename Dtype>
void VideoTracker<Dtype>::PostProcessAll() {
  if (is_type_video_) {
    LOG(INFO) << "Finished tracking the video.";
  }
}

INSTANTIATE_CLASS(VideoTracker);

}
