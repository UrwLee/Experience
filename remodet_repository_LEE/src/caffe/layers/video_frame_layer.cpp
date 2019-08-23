#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include <csignal>
// #include "signal.h"
// #include <csignal>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_frame_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
VideoframeLayer<Dtype>::VideoframeLayer(const LayerParameter& param)
  : BasePrefetchingData2Layer<Dtype>(param) {
}

template <typename Dtype>
VideoframeLayer<Dtype>::~VideoframeLayer() {
  this->StopInternalThread();
  if (cap_.isOpened()) {
    cap_.release();
  }
}

template <typename Dtype>
void VideoframeLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // get the video param
  const VideoframeParameter& video_frame_param =
      this->layer_param_.video_frame_param();
  // get the video type : webcam or video
  video_type_ = video_frame_param.video_type();
  normalize_ = video_frame_param.normalize();
  if (video_frame_param.mean_value_size() > 0) {
    for (int i = 0; i < video_frame_param.mean_value_size(); ++i) {
      mean_values_.push_back(video_frame_param.mean_value(i));
    }
    CHECK_EQ(mean_values_.size(),3);
  }
  if (video_type_ == VideoframeParameter_VideoType_WEBCAM){
    webcam_width_ = video_frame_param.webcam_width();
    webcam_height_ = video_frame_param.webcam_height();
  }
  // Read an image, and use it to initialize the top blob.
  // top[0] -> resized transformed_data_
  // top[1] -> orig image
  cv::Mat cv_img;
  if (video_type_ == VideoframeParameter_VideoType_WEBCAM) {
    const int device_id = video_frame_param.device_id();
    if (!cap_.open(device_id)) {
      LOG(FATAL) << "Failed to open webcam: " << device_id;
    }
    // set the cam resolution
    cap_.set(CV_CAP_PROP_FRAME_WIDTH, webcam_width_);
    cap_.set(CV_CAP_PROP_FRAME_HEIGHT, webcam_height_);

    cap_ >> cv_img;
  } else if (video_type_ == VideoframeParameter_VideoType_VIDEO) {
    CHECK(video_frame_param.has_video_file()) << "Must provide video file!";
    const string& video_file = video_frame_param.video_file();
    const int initial_frame_ = video_frame_param.initial_frame();
    CHECK_GE(initial_frame_, 0);
    if (!cap_.open(video_file)) {
      LOG(FATAL) << "Failed to open video: " << video_file;
    }
    total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    cap_ >> cv_img;
    cap_.set(CV_CAP_PROP_POS_FRAMES, initial_frame_);
    processed_frames_ = initial_frame_;
  } else {
    LOG(FATAL) << "Unknow video type!";
  }
  CHECK(cv_img.data) << "Could not load image!";
  // 初始化输出shape0
  top0_shape_ = this->data_transformer_->InferBlobShape(cv_img);
  // 对变换器的输出shape进行设置
  this->transformed_data_.Reshape(top0_shape_);
  top[0]->Reshape(top0_shape_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top0_shape_);
  }
  LOG(INFO) << "output data [0] size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  const int img_height = cv_img.rows;
  const int img_width  = cv_img.cols;
  const int img_channels = cv_img.channels();
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK_GT(img_channels, 0);
  top1_shape_.clear();
  top1_shape_.push_back(1);
  top1_shape_.push_back(img_channels);
  top1_shape_.push_back(img_height);
  top1_shape_.push_back(img_width);
  top[1]->Reshape(top1_shape_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].orig_data_.Reshape(top1_shape_);
  }
  LOG(INFO) << "output data [1] size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

// This function is called on prefetch thread
template<typename Dtype>
void VideoframeLayer<Dtype>::load_batch_orig(Batch_Orig<Dtype>* batch_orig) {
  CHECK(batch_orig->data_.count());
  CHECK(batch_orig->orig_data_.count());
  CHECK(this->transformed_data_.count());
  this->transformed_data_.Reshape(top0_shape_);
  batch_orig->data_.Reshape(top0_shape_);
  batch_orig->orig_data_.Reshape(top1_shape_);

  Dtype* top0_data = batch_orig->data_.mutable_cpu_data();

  cv::Mat cv_img;
  if (video_type_ == VideoframeParameter_VideoType_WEBCAM) {
    cap_ >> cv_img;
  } else if (video_type_ == VideoframeParameter_VideoType_VIDEO) {
    if (processed_frames_ >= total_frames_) {
      LOG(INFO) << "Finished processing video.";
      raise(SIGINT);
    }
    ++processed_frames_;
    cap_ >> cv_img;
  } else {
    LOG(FATAL) << "Unknown video type.";
  }
  CHECK(cv_img.data) << "Could not load image!";
  this->transformed_data_.set_cpu_data(top0_data);
  this->data_transformer_->NormTransform(cv_img, &(this->transformed_data_), normalize_,mean_values_);

  //load top[1]
  Dtype* top1_data = batch_orig->orig_data_.mutable_cpu_data();
  int top1_index;
  for(int h = 0; h < cv_img.rows; ++h){
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for(int w = 0; w < cv_img.cols; ++w){
      for(int c = 0; c < cv_img.channels(); ++c){
        top1_index = (c * cv_img.rows + h) * cv_img.cols + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        top1_data[top1_index] = pixel;
      }
    }
  }
}

INSTANTIATE_CLASS(VideoframeLayer);
REGISTER_LAYER_CLASS(Videoframe);

}  // namespace caffe
#endif  // USE_OPENCV
