#include "caffe/remo/net_wrap.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
NetWrapper<Dtype>::NetWrapper(const std::string& network_proto,
                              const std::string& caffe_model,
                              const bool mode,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
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
  heatmaps_ = heatmaps;
  max_dis_size_ = max_dis_size;
  LOG(INFO) << "Network initialization done.";
}

template <typename Dtype>
NetWrapper<Dtype>::NetWrapper(const std::string& network_proto,
                              const std::string& caffe_model,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size) {
  NetWrapper(network_proto,caffe_model,true,gpu_id,proposals,heatmaps,max_dis_size);
}

template <typename Dtype>
NetWrapper<Dtype>::NetWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
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
  heatmaps_ = heatmaps;
  max_dis_size_ = max_dis_size;
}

template <typename Dtype>
void NetWrapper<Dtype>::load(const cv::Mat& image) {
  // size CHECK
  // CHECK_EQ(image.cols, input_width_) << "Input width not matched.";
  // CHECK_EQ(image.rows, input_height_) << "Input height not matched.";
  Blob<Dtype>* input_blob = net_->input_blobs()[0];
  input_blob->Reshape(1, 3, image.rows, image.cols);
  net_->Reshape();
  Dtype* transformed_data = input_blob->mutable_cpu_data();
  const int offset = image.rows * image.cols;
  bool normalize = false;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      const cv::Vec3b& rgb = image.at<cv::Vec3b>(i, j);
      if (normalize) {
        transformed_data[i * image.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[offset + i * image.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2 * offset + i * image.cols + j] = (rgb[2] - 128)/256.0;
      } else {
        transformed_data[i * image.cols + j] = rgb[0] - 104;
        transformed_data[offset + i * image.cols + j] = rgb[1] - 117;;
        transformed_data[2 * offset + i * image.cols + j] = rgb[2] - 123;;
      }
    }
  }
}

template <typename Dtype>
void NetWrapper<Dtype>::getFeatures(const std::string& feature_name, Blob<Dtype>* data) {
  const boost::shared_ptr<Blob<Dtype> > feature = net_->blob_by_name(feature_name.c_str());
  const Blob<Dtype>* f_b = feature.get();
  data->ReshapeLike(*f_b);
  data->ShareData(*f_b);
}

template <typename Dtype>
void NetWrapper<Dtype>::step(DataFrame<Dtype>& frame, std::vector<PMeta<Dtype> >* meta,
                             Blob<Dtype>* map) {
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
  // get proposals
  Blob<Dtype> proposals;
  getFeatures(proposals_, &proposals);
  CHECK_EQ(proposals.num(),1) << "Proposals must have a size of (1,1,N,63).";
  CHECK_EQ(proposals.channels(),1) << "Proposals must have a size of (1,1,N,63).";
  // CHECK_EQ(proposals.width(),63) << "Proposals must have a size of (1,1,N,63).";
  transfer_meta(proposals.cpu_data(), proposals.height(), meta);
  // get map
  getFeatures(heatmaps_, map);
  CHECK_EQ(map->channels(), 52) << "Heatmaps must have a size of (1,52,mh,mw).";
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
void NetWrapper<Dtype>::get_meta(DataFrame<Dtype>& frame,
                  std::vector<PMeta<Dtype> >* meta) {
  Blob<Dtype> map;
  step(frame,meta,&map);
}

template <typename Dtype>
void NetWrapper<Dtype>::get_meta(DataFrame<Dtype>& frame,
                  std::vector<std::vector<Dtype> >* meta) {
  // get Results
  Blob<Dtype> map;
  std::vector<PMeta<Dtype> > metaf;
  step(frame,&metaf,&map);
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  ResFrame<Dtype> resFrame(frame.get_ori_image(), frames_, max_dis_size_, metaf);
  resFrame.get_meta(meta);
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_vecmap(DataFrame<Dtype>& frame, const bool show_bbox, const bool show_id,
                                       std::vector<std::vector<Dtype> >* meta) {
   Blob<Dtype> map;
   std::vector<PMeta<Dtype> > metaf;
   step(frame,&metaf,&map);
   // drawn time
   caffe::Timer drawn_timer;
   drawn_timer.Start();
   ResFrame<Dtype> resFrame(frame.get_ori_image(), frames_, max_dis_size_, metaf);
   cv::Mat vecmap_image = resFrame.get_drawn_vecmap(map.gpu_data(), map.width(), map.height(), show_bbox, show_id);
   resFrame.get_meta(meta);
   drawn_time_sum_ += drawn_timer.MicroSeconds();
   if (frames_ % 30 == 0) {
     drawn_time_ = drawn_time_sum_ / 30;
     drawn_time_sum_ = 0;
   }
   return vecmap_image;
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_vecmap(DataFrame<Dtype>& frame,
                        std::vector<std::vector<Dtype> >* meta) {
  return get_vecmap(frame,false,false,meta);
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_heatmap(DataFrame<Dtype>& frame, const bool show_bbox, const bool show_id,
                                        std::vector<std::vector<Dtype> >* meta) {
  Blob<Dtype> map;
  std::vector<PMeta<Dtype> > metaf;
  step(frame,&metaf,&map);
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  ResFrame<Dtype> resFrame(frame.get_ori_image(), frames_, max_dis_size_, metaf);
  cv::Mat heatmap_image = resFrame.get_drawn_heatmap(map.gpu_data(), map.width(), map.height(), show_bbox, show_id);
  resFrame.get_meta(meta);
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
  return heatmap_image;
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_heatmap(DataFrame<Dtype>& frame,
                          std::vector<std::vector<Dtype> >* meta) {
  return get_heatmap(frame,false,false,meta);
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_skeleton(DataFrame<Dtype>& frame,
                            const bool show_bbox, const bool show_id,
                            std::vector<std::vector<Dtype> >* meta) {
  // get Results
  // LOG(INFO) << "#1";
  Blob<Dtype> map;
  std::vector<PMeta<Dtype> > metaf;
  step(frame,&metaf,&map);
  // LOG(INFO) << "#2";
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  ResFrame<Dtype> resFrame(frame.get_ori_image(), frames_, max_dis_size_, metaf);
  cv::Mat skeleton_image = resFrame.get_drawn_skeleton(show_bbox, show_id);
  resFrame.get_meta(meta);
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
  return skeleton_image;
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_skeleton(DataFrame<Dtype>& frame,
                          std::vector<std::vector<Dtype> >* meta) {
  return get_skeleton(frame,false,false,meta);
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get_bbox(DataFrame<Dtype>& frame, const bool show_id,
                                     std::vector<std::vector<Dtype> >* meta) {
  // get Results
  Blob<Dtype> map;
  std::vector<PMeta<Dtype> > metaf;
  step(frame,&metaf,&map);
  // drawn time
  caffe::Timer drawn_timer;
  drawn_timer.Start();
  ResFrame<Dtype> resFrame(frame.get_ori_image(), frames_, max_dis_size_, metaf);
  cv::Mat bbox_image = resFrame.get_drawn_bbox(show_id);
  resFrame.get_meta(meta);
  drawn_time_sum_ += drawn_timer.MicroSeconds();
  if (frames_ % 30 == 0) {
    drawn_time_ = drawn_time_sum_ / 30;
    drawn_time_sum_ = 0;
  }
  return bbox_image;
}

template <typename Dtype>
cv::Mat NetWrapper<Dtype>::get(DataFrame<Dtype>& frame,
                  std::vector<std::vector<Dtype> >* meta) {
  return get_skeleton(frame, true, false, meta);
}

INSTANTIATE_CLASS(NetWrapper);
}
