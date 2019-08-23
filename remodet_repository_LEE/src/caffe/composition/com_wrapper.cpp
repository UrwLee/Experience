#include "caffe/composition/com_wrapper.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
ComWrapper<Dtype>::ComWrapper(const std::string& network_proto,
                              const std::string& caffe_model,
                              const bool mode,
                              const int gpu_id,
                              const std::string& scoremap,
                              const int display_size) {
  if (mode) {
    // use GPU
    LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    net_.reset(new Net<Dtype>(network_proto, caffe::TEST));
  } else {
    // use CPU
    LOG(INFO) << "Using CPU mode in Caffe.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    net_.reset(new Net<Dtype>(network_proto, caffe::TEST));
  }
  if (caffe_model != "NONE") {
    net_->CopyTrainedLayersFrom(caffe_model);
  } else {
    LOG(FATAL) << "Must define a pre-trained model.";
  }
  CHECK_EQ(this->net_->num_inputs(), 3) << "Network must have three inputs.";
  // bottom[0] -> featureMaps [1,C,H,W]
  // bottom[1] -> Roi [1,1,1,7]
  // bottom[2] -> hw [1,1,1,2] (w,h)
  Blob<Dtype>* input_feature_layer = this->net_->input_blobs()[0];
  CHECK_EQ(input_feature_layer->num(),1);
  LOG(INFO) << "Network requires input size for input[0]: (channel, width, height) "
            << input_feature_layer->channels() << ", " << input_feature_layer->width()
            << ", " << input_feature_layer->height();
  Blob<Dtype>* input_roi_layer = this->net_->input_blobs()[1];
  CHECK_EQ(input_roi_layer->count(), 7);
  Blob<Dtype>* input_hw_layer = this->net_->input_blobs()[2];
  CHECK_EQ(input_hw_layer->count(), 2);
  frames_ = 0;
  pre_load_time_ = 0;
  process_time_ = 0;
  drawn_time_ = 0;
  pre_load_time_sum_ = 0;
  process_time_sum_ = 0;
  drawn_time_sum_ = 0;
  scoremap_ = scoremap;
  display_size_ = display_size;
  LOG(INFO) << "Network initialization done.";
}

template <typename Dtype>
ComWrapper<Dtype>::ComWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
                              const int gpu_id,
                              const std::string& scoremap,
                              const int display_size) {
  LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  net_ = net;
  CHECK_EQ(this->net_->num_inputs(), 3) << "Network must have three inputs.";
  // bottom[0] -> featureMaps [1,C,H,W]
  // bottom[1] -> Roi [1,1,1,7]
  // bottom[2] -> hw [1,1,1,2] (w,h)
  Blob<Dtype>* input_feature_layer = this->net_->input_blobs()[0];
  CHECK_EQ(input_feature_layer->num(),1);
  LOG(INFO) << "Network requires input size for input[0]: (channel, width, height) "
            << input_feature_layer->channels() << ", " << input_feature_layer->width()
            << ", " << input_feature_layer->height();
  Blob<Dtype>* input_roi_layer = this->net_->input_blobs()[1];
  CHECK_EQ(input_roi_layer->count(), 7);
  Blob<Dtype>* input_hw_layer = this->net_->input_blobs()[2];
  CHECK_EQ(input_hw_layer->count(), 2);
  frames_ = 0;
  pre_load_time_ = 0;
  process_time_ = 0;
  drawn_time_ = 0;
  pre_load_time_sum_ = 0;
  process_time_sum_ = 0;
  drawn_time_sum_ = 0;
  scoremap_ = scoremap;
  display_size_ = display_size;
}

template <typename Dtype>
void ComWrapper<Dtype>::load(vector<Blob<Dtype>* >& input_blobs) {
  CHECK_EQ(input_blobs.size(), 3);
  CHECK_EQ(input_blobs[1]->count(), 7);
  CHECK_EQ(input_blobs[2]->count(), 2);
  CHECK_EQ(input_blobs[0]->num(), 1);
  // feature maps
  Blob<Dtype>* input_feature_layer = this->net_->input_blobs()[0];
  input_feature_layer->ReshapeLike(*(input_blobs[0]));
  input_feature_layer->ShareData(*(input_blobs[0]));
  // roi
  Blob<Dtype>* input_roi_layer = this->net_->input_blobs()[1];
  input_roi_layer->ReshapeLike(*(input_blobs[1]));
  input_roi_layer->ShareData(*(input_blobs[1]));
  // hw
  Blob<Dtype>* input_hw_layer = this->net_->input_blobs()[2];
  input_hw_layer->ReshapeLike(*(input_blobs[2]));
  input_hw_layer->ShareData(*(input_blobs[2]));
  net_->Reshape();
}

template <typename Dtype>
void ComWrapper<Dtype>::getFeatures(const std::string& feature_name, Blob<Dtype>* data) {
  const boost::shared_ptr<Blob<Dtype> > feature = net_->blob_by_name(feature_name.c_str());
  const Blob<Dtype>* f_b = feature.get();
  data->ReshapeLike(*f_b);
  data->ShareData(*f_b);
}

template <typename Dtype>
void ComWrapper<Dtype>::step(vector<Blob<Dtype>* >& input_blobs) {
  ++frames_;
  // load
  caffe::Timer preload_timer;
  preload_timer.Start();
  load(input_blobs);
  pre_load_time_sum_ += preload_timer.MicroSeconds();
  // process
  caffe::Timer process_timer;
  process_timer.Start();
  net_->Forward();
  process_time_sum_ += process_timer.MicroSeconds();
  // 30 frames
  if (frames_ % 30 == 0) {
    // update
    pre_load_time_ = pre_load_time_sum_ / 30;
    process_time_ = process_time_sum_ / 30;
    pre_load_time_sum_ = 0;
    process_time_sum_ = 0;
  }
}

template <typename Dtype>
void ComWrapper<Dtype>::get_result(vector<Blob<Dtype>* >& input_blobs, Blob<Dtype>* output) {
  step(input_blobs);
  getFeatures(scoremap_,output);
}

template <typename Dtype>
cv::Mat ComWrapper<Dtype>::get_scoremap(vector<Blob<Dtype>* >& input_blobs, Blob<Dtype>* output) {
  step(input_blobs);
  getFeatures(scoremap_,output);
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  // resized image
  const int map_width = output->width();
  const int map_height = output->height();
  const int maxsize = (map_width > map_height) ? map_width : map_height;
  const Dtype ratio = (Dtype)display_size_ / maxsize;
  const int display_width = static_cast<int>(map_width * ratio);
  const int display_height = static_cast<int>(map_height * ratio);
  // blob to cv
  cv::Mat score_map = cv::Mat::zeros(map_height,map_width,CV_8UC1);
  const Dtype* map_data = output->cpu_data();
  for (int i = 0; i < score_map.rows; ++i) {
    for (int j = 0; j < score_map.cols; ++j) {
      score_map.at<uchar>(i,j) = static_cast<unsigned char>(map_data[i*map_width+j]*256.0);
    }
  }
  cv::Mat display_image;
  cv::resize(score_map, display_image, cv::Size(display_width,display_height), cv::INTER_CUBIC);
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
  return display_image;
}

INSTANTIATE_CLASS(ComWrapper);
}
