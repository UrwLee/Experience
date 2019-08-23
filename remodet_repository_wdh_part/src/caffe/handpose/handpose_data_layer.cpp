#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/handpose/handpose_data_layer.hpp"

namespace caffe {

template <typename Dtype>
HandPoseDataLayer<Dtype>::~HandPoseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void HandPoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 增广
  HandPoseDataParameter handpose_data_param = this->layer_param_.handpose_data_param();
  const int resize_width = handpose_data_param.resize_w();
  const int resize_height = handpose_data_param.resize_h();
  const bool flip = handpose_data_param.flip();
  const float flip_prob = handpose_data_param.flip_prob();
  const bool save = handpose_data_param.save();
  const string save_path = handpose_data_param.save_path();
  const float bbox_extend_min = handpose_data_param.bbox_extend_min();
  const float bbox_extend_max = handpose_data_param.bbox_extend_max();
  const float rotate_angle = handpose_data_param.rotate_angle();
  bool clip = false;
  bool flag_augIntrain = true;
  if (handpose_data_param.has_clip()){
    clip = handpose_data_param.clip();
  }
  if (handpose_data_param.has_flag_augintrain()){
    flag_augIntrain = handpose_data_param.flag_augintrain();
  }
  TransformationParameter transform_param = this->layer_param_.transform_param();
  const DistortionParameter distortion_param = transform_param.distort_param();
  augPtr_.reset(new HandPoseAugmenter(flip,flip_prob,resize_width,resize_height,save,save_path,distortion_param,
                                  bbox_extend_min,bbox_extend_max, rotate_angle,clip,flag_augIntrain));
  // 数据

  const string source = handpose_data_param.source();
  const string root_folder = handpose_data_param.root_folder();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "Failed to open file "<< source;
  std::string str_line;
  while (std::getline(infile, str_line)) {
    lines_.push_back(make_pair(root_folder, str_line));
  }
  
  
  CHECK(!lines_.empty()) << "File is empty";
  if (handpose_data_param.shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " samples.";
  lines_id_ = 0;
  if (handpose_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % handpose_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // 特征提取
  // const string network_proto = handpose_data_param.network_proto();
  // const string network_weight = handpose_data_param.network_weight();
  // const string feature_name = handpose_data_param.feature_name();
  // const int gpu_id = handpose_data_param.gpu_id();
  // handposeBase_.reset(new handposeBase<Dtype>(network_proto,network_weight,gpu_id,feature_name));
  // const int feature_channels = handpose_data_param.feature_channels();
  // const int feature_height = handpose_data_param.feature_height();
  // const int feature_width = handpose_data_param.feature_width();

  const int batch_size = handpose_data_param.batch_size();
  vector<int> top_shape;
  top_shape.push_back(batch_size);
  top_shape.push_back(3);
  top_shape.push_back(resize_height);
  top_shape.push_back(resize_width);

  // top data
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // top label
  int label_num = top_shape[0];
  vector<int> label_shape(1, label_num);
  LOG(INFO)<<label_shape.size();
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  LOG(INFO) << "Exit Datasetup";
}

template <typename Dtype>
void HandPoseDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void HandPoseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  CHECK(batch->data_.count());

  HandPoseDataParameter handpose_data_param = this->layer_param_.handpose_data_param();
  const string root_folder = handpose_data_param.root_folder();
  const int batch_size = handpose_data_param.batch_size();
  const int resize_w = handpose_data_param.resize_w();
  const int resize_h = handpose_data_param.resize_h();
  // const int feature_channels = handpose_data_param.feature_channels();
  // const int feature_height = handpose_data_param.feature_height();
  // const int feature_width = handpose_data_param.feature_width();
  vector<int> top_shape;
  top_shape.push_back(batch_size);
  top_shape.push_back(3);
  top_shape.push_back(resize_h);
  top_shape.push_back(resize_w);
 
  // 定义数据和标签的尺寸
  batch->data_.Reshape(top_shape);
  vector<int> label_shape(1, batch_size);
  batch->label_.Reshape(label_shape);
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id){
    CHECK_GT(lines_.size(), lines_id_);
    // 读取标注
    HandPoseInstance ins;
    parseHandPoseLine(lines_[lines_id_].second, lines_[lines_id_].first, this->phase_, &ins);
    // 获取标签和图像
    cv::Mat image;
    int id;
    augPtr_->aug(ins, &image, &id, this->phase_);
    // 读取信息到输出
    CHECK_EQ(image.cols, resize_w);
    CHECK_EQ(image.rows, resize_h);
    const int offset = resize_h * resize_w;
    bool normalize = false;
    for (int i = 0; i < resize_h; ++i) {
      for (int j = 0; j < resize_w; ++j) {
        const cv::Vec3b& rgb = image.at<cv::Vec3b>(i, j);
        if (normalize) {
          prefetch_data[             i * resize_w + j] = (rgb[0] - 128)/256.0;
          prefetch_data[    offset + i * resize_w + j] = (rgb[1] - 128)/256.0;
          prefetch_data[2 * offset + i * resize_w + j] = (rgb[2] - 128)/256.0;
        } else {
          prefetch_data[             i * resize_w + j] = rgb[0] - 104;
          prefetch_data[    offset + i * resize_w + j] = rgb[1] - 117;
          prefetch_data[2 * offset + i * resize_w + j] = rgb[2] - 123;
        }
      }
    }
    // 指向下一个样本
    prefetch_data += 3 * resize_h * resize_w;
    // 读取label
    *prefetch_label = id;
    //LOG(INFO)<<id<<" "<<this->phase_<<" datalayer";
    // 指向下一个样本
    ++prefetch_label;
    // 循环遍历
    lines_id_++;
    if (lines_id_ >= lines_.size()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.handpose_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "   Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(HandPoseDataLayer);
REGISTER_LAYER_CLASS(HandPoseData);
}
#endif
