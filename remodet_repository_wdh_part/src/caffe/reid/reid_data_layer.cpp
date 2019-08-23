#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/reid/reid_data_layer.hpp"

namespace caffe {

template <typename Dtype>
ReidDataLayer<Dtype>::~ReidDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ReidDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 初始化数据转换器
  reid_transformer_.reset(new ReidTransformer<Dtype>(reid_transform_param_, this->phase_));
  reid_transformer_->InitRand();
  // 获取数据Layer参数
  const ReidDataParameter& reid_data_param = this->layer_param_.reid_data_param();
  // xml列表文件
  string xml_list = reid_data_param.xml_list();
  // xml根文件路径
  string xml_root = reid_data_param.xml_root();
  // 获取XML文件列表
  LOG(INFO) << "Opening file " << xml_list;
  std::ifstream infile(xml_list.c_str());
  std::string xmlname;
  while (infile >> xmlname) {
    lines_.push_back(xmlname);
  }
  CHECK(!lines_.empty()) << "File is empty";
  // 随机乱序
  if (reid_data_param.shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLists();
  }
  LOG(INFO) << "A total of " << lines_.size() << " instances.";
  // 随机跳过头部数据点
  lines_id_ = 0;
  if (reid_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % reid_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " instances.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // 输出数据尺寸
  const int batch_size = reid_data_param.batch_size();
  const int height = reid_transform_param_.resized_height();
  const int width = reid_transform_param_.resized_width();
  top[0]->Reshape(batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
  }
  this->transformed_data_.Reshape(1, 3, height, width);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  top[1]->Reshape(1,1,1,7);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(1,1,1,7);
  }
}

template <typename Dtype>
void ReidDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void ReidDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  // 获取batchsize和输出h/w
  const ReidDataParameter& reid_data_param = this->layer_param_.reid_data_param();
  string xml_root = reid_data_param.xml_root();
  const int batch_size = reid_data_param.batch_size();
  const int height = reid_transform_param_.resized_height();
  const int width = reid_transform_param_.resized_width();
  batch->data_.Reshape(batch_size, 3, height, width);
  this->transformed_data_.Reshape(1, 3, height, width);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  // 设置label
  const int lines_size = lines_.size();
  int total_instances = 0;
  vector<std::string> xml_paths;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    int ins;
    // 获取路径
    string xml_path = xml_root + lines_[lines_id_];
    // 保存
    xml_paths.push_back(xml_path);
    // 获取目标数
    reid_transformer_->getNumSamples(xml_path,&ins);
    total_instances += ins;
    // 指向下一条记录
    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (reid_data_param.shuffle()) {
        ShuffleLists();
      }
    }
  }
  // label
  batch->label_.Reshape(1, 1, total_instances, 7);
  Dtype* top_label = batch->label_.mutable_cpu_data();
  CHECK_EQ(xml_paths.size(), batch_size);
  int ins_idx = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    const int offset_data = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    // 设置转换数据地址
    Dtype* transformed_data = this->transformed_data_.mutable_cpu_data();
    Dtype* transformed_label = top_label + ins_idx * 7;
    int temp;
    reid_transformer_->Transform(xml_paths[item_id],item_id,transformed_data,transformed_label,&temp);
    ins_idx += temp;
  }
  CHECK_EQ(ins_idx,total_instances) << "Mismatch between total_instances and transformed_label.";
}

INSTANTIATE_CLASS(ReidDataLayer);
REGISTER_LAYER_CLASS(ReidData);
}  // namespace caffe
