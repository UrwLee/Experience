#include "caffe/remo/data_frame.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

using std::string;
using std::vector;
using namespace cv;

template <typename Dtype>
DataFrame<Dtype>::DataFrame() {}

template <typename Dtype>
DataFrame<Dtype>::DataFrame(int id, const cv::Mat& image, int resized_width, int resized_height)
  : ori_image_(image), id_(id) {
  cv::resize(image,resized_image_,cv::Size(resized_width,resized_height),0,0,CV_INTER_LINEAR);
}

template <typename Dtype>
void DataFrame<Dtype>::show_ori() const {
  if(!ori_image_.data ) {
    LOG(FATAL) << "Error - open the frame failed.";
  } else {
    cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
    cv::imshow( "Remo", ori_image_);
    cv::waitKey(1);
  }
}

template <typename Dtype>
void DataFrame<Dtype>::show_resized() const {
  if(!resized_image_.data ) {
    LOG(FATAL) << "Error - open the frame (resized) failed.";
  } else {
    cv::namedWindow("Remo_resized", cv::WINDOW_AUTOSIZE);
    cv::imshow( "Remo_resized", resized_image_);
    cv::waitKey(1);
  }
}

INSTANTIATE_CLASS(DataFrame);
}
