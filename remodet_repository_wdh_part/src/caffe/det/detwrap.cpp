#include "caffe/det/detwrap.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/io.hpp"
namespace caffe {

static int DRAW_COLOR_MAPS[18] = {255,0,0,0,255,0,0,0,255,255,255,0,0,255,255,255,0,255};

template <typename Dtype>
DetWrapper<Dtype>::DetWrapper(const std::string& network_proto,
                              const std::string& caffe_model,
                              const bool mode,
                              const int gpu_id,
                              const std::string& proposals,
                              const int max_dis_size) {
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
  Blob<Dtype>* input_layer = this->net_->input_blobs()[0];
  input_width_ = input_layer->width();
  input_height_ = input_layer->height();
  LOG(INFO) << "Network requires input size: (width, height) "
            << input_width_ << ", " << input_height_;
  CHECK_EQ(input_layer->channels(), 3) << "Input layer should have 3 channels.";
  frames_ = 0;
  pre_load_time_ = 0;
  process_time_ = 0;
  drawn_time_ = 0;
  pre_load_time_sum_ = 0;
  process_time_sum_ = 0;
  drawn_time_sum_ = 0;
  proposals_ = proposals;
  max_dis_size_ = max_dis_size;
  hisi_data_ = true;
  if (hisi_data_){
    string hisi_data_maps = "/home/ethan/work/hisi_data_maps.txt";
    makeHisiDataMaps(hisi_data_maps, &maps_);
  }
  LOG(INFO) << "Network initialization done.";
}

template <typename Dtype>
DetWrapper<Dtype>::DetWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
                              const int gpu_id,
                              const std::string& proposals,
                              const int max_dis_size) {
  LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  net_ = net;
  Blob<Dtype>* input_layer = this->net_->input_blobs()[0];
  input_width_ = input_layer->width();
  input_height_ = input_layer->height();
  LOG(INFO) << "Network requires input size: (width, height) "
            << input_width_ << ", " << input_height_;
  CHECK_EQ(input_layer->channels(), 3) << "Input layer should have 3 channels.";
  frames_ = 0;
  pre_load_time_ = 0;
  process_time_ = 0;
  drawn_time_ = 0;
  pre_load_time_sum_ = 0;
  process_time_sum_ = 0;
  drawn_time_sum_ = 0;
  proposals_ = proposals;
  max_dis_size_ = max_dis_size;
}

template <typename Dtype>
void DetWrapper<Dtype>::load(const cv::Mat& image) {
  Blob<Dtype>* input_blob = net_->input_blobs()[0];
  input_blob->Reshape(1, 3, image.rows, image.cols);
  net_->Reshape();
  Dtype* transformed_data = input_blob->mutable_cpu_data();
  const int offset = image.rows * image.cols;
  bool normalize = false;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      const cv::Vec3b& rgb = image.at<cv::Vec3b>(i, j);
      if (hisi_data_){
        transformed_data[i * image.cols + j] = maps_[rgb[0] - 104];
        transformed_data[offset + i * image.cols + j] = maps_[rgb[1] - 117];
        transformed_data[2 * offset + i * image.cols + j] = maps_[rgb[2] - 123];
      }else{
        if (normalize) {
          transformed_data[i * image.cols + j] = (rgb[0] - 127.5)/128.0;
          transformed_data[offset + i * image.cols + j] = (rgb[1] - 127.5)/128.0;
          transformed_data[2 * offset + i * image.cols + j] = (rgb[2] - 127.5)/128.0;
        } else {
          transformed_data[i * image.cols + j] = rgb[0] - 104;
          transformed_data[offset + i * image.cols + j] = rgb[1] - 117;;
          transformed_data[2 * offset + i * image.cols + j] = rgb[2] - 123;;
        }
      }
    }
  }
}

template <typename Dtype>
void DetWrapper<Dtype>::getFeatures(const std::string& feature_name, Blob<Dtype>* data) {
  const boost::shared_ptr<Blob<Dtype> > feature = net_->blob_by_name(feature_name.c_str());
  const Blob<Dtype>* f_b = feature.get();
  data->ReshapeLike(*f_b);
  data->ShareData(*f_b);
}

template <typename Dtype>
void DetWrapper<Dtype>::step(DataFrame<Dtype>& frame) {
  ++frames_;
  // load
  caffe::Timer preload_timer;
  preload_timer.Start();
  cv::Mat& resized_image = frame.get_resized_image();
  load(resized_image);
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
void DetWrapper<Dtype>::get_rois(DataFrame<Dtype>& frame, std::vector<LabeledBBox<Dtype> >* rois) {
  step(frame);
  Blob<Dtype> proposals;
  getFeatures(proposals_,&proposals);
  CHECK_EQ(proposals.count() % 7, 0);
  const Dtype* top_data = proposals.cpu_data();
  rois->clear();
  for (int i = 0; i < proposals.count() / 7; ++i) {
    if (top_data[i*7] < 0) continue;
    LabeledBBox<Dtype> tbox;
    tbox.bindex = top_data[i*7];
    tbox.cid    = top_data[i*7+1];
    tbox.score  = top_data[i*7+2];
    tbox.bbox.x1_ = top_data[i*7+3];
    tbox.bbox.y1_ = top_data[i*7+4];
    tbox.bbox.x2_ = top_data[i*7+5];
    tbox.bbox.y2_ = top_data[i*7+6];
    rois->push_back(tbox);
  }
}

template <typename Dtype>
cv::Mat DetWrapper<Dtype>::get_drawn_bboxes(DataFrame<Dtype>& frame, const std::vector<int>& labels, std::vector<LabeledBBox<Dtype> >* rois) {
  get_rois(frame,rois);
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  // resized image
  cv::Mat image = frame.get_ori_image();
  const int maxsize = (image.cols > image.rows) ? image.cols : image.rows;
  const Dtype ratio = (Dtype)max_dis_size_ / maxsize;
  const int display_width = static_cast<int>(image.cols * ratio);
  const int display_height = static_cast<int>(image.rows * ratio);
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(display_width,display_height), cv::INTER_LINEAR);
  // draw bboxes
  for (int i = 0; i < rois->size(); ++i) {
    LabeledBBox<Dtype>& roi = (*rois)[i];
    float area = (roi.bbox.x2_-roi.bbox.x1_)*(roi.bbox.y2_-roi.bbox.y1_);
    LOG(INFO)<<"area; "<<area<<"; bboxsize: "<<std::sqrt(area)<<" cid"<<roi.cid<<".";
    roi.bbox.x1_ *= resized_image.cols;
    roi.bbox.x2_ *= resized_image.cols;
    roi.bbox.y1_ *= resized_image.rows;
    roi.bbox.y2_ *= resized_image.rows;
    int cid = roi.cid;
    // 如果labels为空，则直接绘制
    // 如果labels有效，则进行选择
    if (labels.size() > 0) {
      bool found = false;
      for (int k = 0; k < labels.size(); ++k) {
        if (cid == labels[k]) {
          found = true;
          break;
        }
      }
      if (!found) continue;
    }
    int r = DRAW_COLOR_MAPS[cid*3];
    int g = DRAW_COLOR_MAPS[cid*3+1];
    int b = DRAW_COLOR_MAPS[cid*3+2];
    roi.bbox.Draw(r, g, b, &resized_image);
    char tmp_str[256];
    // // write FPS
    // snprintf(tmp_str, 256, "%0.3f", area);
    // cv::putText(resized_image, tmp_str, cv::Point(roi.bbox.x1_,roi.bbox.y1_),
    //     cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,0,255), 3);
  }
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
  return resized_image;
}

template <typename Dtype>
void DetWrapper<Dtype>::getMonitorBlob(DataFrame<Dtype>& frame, const std::string& monitor_blob_name, Blob<Dtype>* res_blob) {
  step(frame);
  getFeatures(monitor_blob_name,res_blob);
}

INSTANTIATE_CLASS(DetWrapper);
}
