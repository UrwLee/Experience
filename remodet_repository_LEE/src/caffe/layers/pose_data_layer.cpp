#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/pose_data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/pose_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
PoseDataLayer<Dtype>::~PoseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void PoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 初始化数据转换器
  pose_data_transformer_.reset(
     new PoseDataTransformer<Dtype>(pose_data_transform_param_, this->phase_));
  pose_data_transformer_->InitRand();
  // 获取数据Layer参数
  const PoseDataParameter& pose_data_param = this->layer_param_.pose_data_param();
  // xml列表文件
  string xml_list = pose_data_param.xml_list();
  // xml根文件路径
  string xml_root = pose_data_param.xml_root();
  // 获取XML文件列表
  LOG(INFO) << "Opening file " << xml_list;
  std::ifstream infile(xml_list.c_str());
  std::string xmlname;
  while (infile >> xmlname) {
    lines_.push_back(xmlname);
  }
  CHECK(!lines_.empty()) << "File is empty";
  // 随机乱序
  if (pose_data_param.shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLists();
  }
  LOG(INFO) << "A total of " << lines_.size() << " instances.";
  // 随机跳过头部数据点
  lines_id_ = 0;
  if (pose_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % pose_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " instances.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // 输出数据尺寸
  const int batch_size = pose_data_param.batch_size();
  // TRAIN: crop尺寸, TEST:resized尺寸
  const int height = this->phase_ == TRAIN ? pose_data_transform_param_.crop_size_y() : pose_data_transform_param_.resized_height();
  const int width = this->phase_ == TRAIN ? pose_data_transform_param_.crop_size_x() : pose_data_transform_param_.resized_width();
  top[0]->Reshape(batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
  }
  this->transformed_data_.Reshape(1, 3, height, width);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  const int stride = pose_data_transform_param_.stride();
  int num_parts = 18;
  int num_limbs = 17;
  // 0~2*num_limbs-1: -> vecmask
  // 0~num_parts: -> heatmask {num_parts+bkg}
  // 0~2*num_limbs-1: -> VecMap
  // 0~num_parts: -> heatmap {num_parts+bkg}
  // 0~3: label/x/y/belong {KPS}
  if (pose_data_param.out_kps()) {
    top[1]->Reshape(batch_size, 2*(2*num_limbs+num_parts)+4, height/stride, width/stride);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 2*(2*num_limbs+num_parts)+4, height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(2*num_limbs+num_parts)+4, height/stride, width/stride);
  } else {
    top[1]->Reshape(batch_size, 2*(2*num_limbs+num_parts), height/stride, width/stride);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 2*(2*num_limbs+num_parts), height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(2*num_limbs+num_parts), height/stride, width/stride);
  }
}

template <typename Dtype>
void PoseDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void PoseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  // 获取batchsize和输出h/w
  const PoseDataParameter& pose_data_param = this->layer_param_.pose_data_param();
  string xml_root = pose_data_param.xml_root();
  const int batch_size = pose_data_param.batch_size();
  const int height = this->phase_ == TRAIN ? pose_data_transform_param_.crop_size_y() : pose_data_transform_param_.resized_height();
  const int width = this->phase_ == TRAIN ? pose_data_transform_param_.crop_size_x() : pose_data_transform_param_.resized_width();
  batch->data_.Reshape(batch_size, 3, height, width);
  this->transformed_data_.Reshape(1, 3, height, width);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  // 设置label
  const int stride = pose_data_transform_param_.stride();
  int num_parts = 18;
  int num_limbs = 17;
  if (pose_data_param.out_kps()) {
    batch->label_.Reshape(batch_size, 2*(2*num_limbs+num_parts)+4, height/stride, width/stride);
    this->transformed_label_.Reshape(1, 2*(2*num_limbs+num_parts)+4, height/stride, width/stride);
  } else {
    batch->label_.Reshape(batch_size, 2*(2*num_limbs+num_parts), height/stride, width/stride);
    this->transformed_label_.Reshape(1, 2*(2*num_limbs+num_parts), height/stride, width/stride);
  }
  Dtype* top_label = batch->label_.mutable_cpu_data();
  int channelOffset = (height/stride) * (width/stride);
  // 依次输出b个样本
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_path = xml_root + lines_[lines_id_];
    const int offset_data = batch->data_.offset(item_id);
    const int offset_label = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);
    // 设置转换数据地址
    Dtype* transformed_data = this->transformed_data_.mutable_cpu_data();
    Dtype* transformed_vec_mask = this->transformed_label_.mutable_cpu_data();
    Dtype* transformed_heat_mask = this->transformed_label_.mutable_cpu_data() + (2*num_limbs)*channelOffset;
    Dtype* transformed_vecmap = this->transformed_label_.mutable_cpu_data() + (2*num_limbs+num_parts)*channelOffset;
    Dtype* transformed_heatmap = this->transformed_label_.mutable_cpu_data() + (4*num_limbs+num_parts)*channelOffset;
    if (!pose_data_param.out_kps()) {
      this->pose_data_transformer_->Transform_nv(xml_path,transformed_data,transformed_vec_mask,
          transformed_heat_mask,transformed_vecmap,transformed_heatmap);
    } else {
      Dtype* transformed_kps = this->transformed_label_.mutable_cpu_data() + (4*num_limbs+2*num_parts)*channelOffset;
      this->pose_data_transformer_->Transform_nv_out(xml_path,transformed_data,transformed_vec_mask,
          transformed_heat_mask,transformed_vecmap,transformed_heatmap,transformed_kps);
    }
    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (pose_data_param.shuffle()) {
        ShuffleLists();
      }
    }
  }
}

INSTANTIATE_CLASS(PoseDataLayer);
REGISTER_LAYER_CLASS(PoseData);
}  // namespace caffe
