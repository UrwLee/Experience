#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/minihand/minihand_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace caffe {

using namespace boost::property_tree;

template <typename Dtype>
MinihandDataLayer<Dtype>::~MinihandDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MinihandDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  minihand_transformer_.reset(
     new MinihandTransformer<Dtype>(minihand_transform_param_, this->phase_));
  // 获取数据Layer参数
  const MinihandDataParameter& minihand_data_param = this->layer_param_.minihand_data_param();
  CHECK_EQ(minihand_data_param.mean_value_size(), 3);
  for (int i = 0; i < 3; ++i) {
    mean_values_.push_back(minihand_data_param.mean_value(i));
  }
  if(minihand_data_param.has_xml_list()){
    string xml_list = minihand_data_param.xml_list();
    string xml_root = minihand_data_param.xml_root();
    LOG(INFO) << "Opening file " << xml_list;
    std::ifstream infile(xml_list.c_str());
    CHECK(infile.good()) << "Failed to open file "<< xml_list;
    std::string xmlname;
    while (infile >> xmlname) {
      lines_.push_back(make_pair(xml_root, xmlname));
    }
  } else{
    LOG(INFO)<<"size of minihand_data_param.xml_list_multiple_size() "<<minihand_data_param.xml_list_multiple_size();
    for (int i=0; i<minihand_data_param.xml_list_multiple_size(); i++){
      string xml_list = minihand_data_param.xml_list_multiple(i);
      string xml_root = minihand_data_param.xml_root_multiple(i);
      LOG(INFO) << "Opening file \"" << xml_list << "\"";
      std::ifstream infile(xml_list.c_str());
      CHECK(infile.good()) << "Failed to open file "<< xml_list;
      std::string xmlname;
      while (infile >> xmlname) {
        lines_.push_back(make_pair(xml_root, xmlname));
      }
      LOG(INFO) << "Finished Reading " << xml_list;
    }
  }

  base_bindex_ = 0;
  if (minihand_data_param.has_base_bindex()){
    base_bindex_ = minihand_data_param.base_bindex();
  }
  
  CHECK(!lines_.empty()) << "File is empty.";
  // 随机乱序
  if (minihand_data_param.shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLists();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  if (minihand_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % minihand_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " instances.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  
  // data
  const int batch_size = minihand_data_param.batch_size();
  const int height = minihand_transform_param_.resized_height();
  const int width = minihand_transform_param_.resized_width();
  top[0]->Reshape(batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  // label
  top[1]->Reshape(1,1,1,9);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(1,1,1,9);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
}

template <typename Dtype>
void MinihandDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void MinihandDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  const MinihandDataParameter& minihand_data_param = this->layer_param_.minihand_data_param();
  string xml_root = minihand_data_param.xml_root();
  string image_root = minihand_data_param.image_root();
  const int batch_size = minihand_data_param.batch_size();
  const int height = minihand_transform_param_.resized_height();
  const int width = minihand_transform_param_.resized_width();
  const int lines_size = lines_.size();
  // 输出Label
  vector<cv::Mat> all_images;   // RGB图像
  vector<vector<LabeledBBox<Dtype> > > all_bboxes;  // Labels
  // 获取batch_size个样本
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_root = lines_[lines_id_].first;
    string xml_path = xml_root + '/' + lines_[lines_id_].second;
    // Read Anno
    HandAnnoData<Dtype> anno;
    ReadHandDataFromXml(item_id, xml_path, xml_root, &anno);
    // 转换
    cv::Mat image;
    vector<LabeledBBox<Dtype> > bboxes;
    bool flag = minihand_transformer_->Transform(anno, &image, &bboxes);
    if (! flag) { // 增广失败
      item_id--;  // 回滚到下一条记录
    } else {
      all_images.push_back(image);
      all_bboxes.push_back(bboxes);
    }
    lines_id_++;
    if (lines_id_ >= lines_.size()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (minihand_data_param.shuffle()) {
        ShuffleLists();
      }
    }
  }
  // 检查数据完整性
  CHECK_EQ(all_images.size(), all_bboxes.size());
  CHECK_EQ(all_images.size(), batch_size);
  // 拷贝数据
  // top[0]
  batch->data_.Reshape(batch_size,3,height,width);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  bool normalize = false;
  const int offset = height * width;
  for (int i = 0; i < batch_size; ++i) {
    Dtype* top_data_item = top_data + i * 3 * offset;
    cv::Mat& image = all_images[i];
    CHECK_EQ(image.rows, height);
    CHECK_EQ(image.cols, width);
    if(minihand_transform_param_.flag_eqhist()){
      vector<cv::Mat> BGR;
      cv::split(image, BGR);
      cv::equalizeHist(BGR[0],BGR[0]);
      cv::equalizeHist(BGR[1],BGR[1]);
      cv::equalizeHist(BGR[2],BGR[2]);
      cv::merge(BGR, image);
    }
    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        const cv::Vec3b& rgb = image.at<cv::Vec3b>(y, x);
        if (normalize) {
          top_data_item[y * image.cols + x] = (rgb[0] - 128)/256.0;
          top_data_item[offset + y * image.cols + x] = (rgb[1] - 128)/256.0;
          top_data_item[2 * offset + y * image.cols + x] = (rgb[2] - 128)/256.0;
        } else {
          top_data_item[             y * image.cols + x] = rgb[0] - mean_values_[0];
          top_data_item[    offset + y * image.cols + x] = rgb[1] - mean_values_[1];
          top_data_item[2 * offset + y * image.cols + x] = rgb[2] - mean_values_[2];
        }
      }
    }
  }
  // top[1]
  int num_gt = 0;
  for (int i = 0; i < all_bboxes.size(); ++i) {
    num_gt += all_bboxes[i].size();
  }
  CHECK_GT(num_gt, 0) << "Found No Ground-Truth.";
  batch->label_.Reshape(1,1,num_gt,9);
  Dtype* top_label = batch->label_.mutable_cpu_data();
  int idx = 0;
  for (int i = 0; i < all_bboxes.size(); ++i) {
    for (int j = 0; j < all_bboxes[i].size(); ++j) {
      LabeledBBox<Dtype>& box = all_bboxes[i][j];
      top_label[idx++] = box.bindex + base_bindex_;
      top_label[idx++] = box.cid;
      top_label[idx++] = box.pid;
      top_label[idx++] = box.is_diff;
      top_label[idx++] = box.iscrowd;
      top_label[idx++] = box.bbox.x1_;
      top_label[idx++] = box.bbox.y1_;
      top_label[idx++] = box.bbox.x2_;
      top_label[idx++] = box.bbox.y2_;
    }
  }
}

template <typename Dtype>
void MinihandDataLayer<Dtype>::ReadHandDataFromXml(const int bindex, const string& xml_file, const string& image_root,
                                                  HandAnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->image_path = image_root + '/' + pt.get<string>("Annotations.ImagePath");
  anno->image_width = pt.get<int>("Annotations.ImageWidth");
  anno->image_height = pt.get<int>("Annotations.ImageHeight");
  int num;
  try {
    num = pt.get<int>("Annotations.NumPerson");
  } catch (const ptree_error &e) {
    num = pt.get<int>("Annotations.NumPart");
  }
  anno->hands.clear();
  for (int i = 0; i < num; ++i) {
    LabeledBBox<Dtype> ins;
    char temp_cid[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);
    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid);
    // NOTE: 只提取cid == 1 的记录
    //if (ins.cid != 1) continue;
    // bbox: must be defined
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_xmax);
    ins.bbox.y2_ = pt.get<Dtype>(temp_ymax);
    // 默认score = 1
    ins.score = 1;
    anno->hands.push_back(ins);
  }
  anno->num_hands = anno->hands.size();
}

INSTANTIATE_CLASS(MinihandDataLayer);
REGISTER_LAYER_CLASS(MinihandData);
}  // namespace caffe
