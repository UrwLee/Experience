#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/mask/face_data_transformer.hpp"
#include "caffe/mask/face_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace caffe {

using namespace boost::property_tree;

template <typename Dtype>
FaceDataLayer<Dtype>::~FaceDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FaceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  face_data_transformer_.reset(
     new FaceDataTransformer<Dtype>(unified_data_transform_param_, this->phase_));
  // 获取数据Layer参数
  const UnifiedDataParameter& unified_data_param = this->layer_param_.unified_data_param();
  CHECK_EQ(unified_data_param.mean_value_size(), 3);
  for (int i = 0; i < 3; ++i) {
    mean_values_.push_back(unified_data_param.mean_value(i));
  }

  string xml_list = unified_data_param.xml_list();

  LOG(INFO) << "Opening file " << xml_list;
  std::ifstream infile(xml_list.c_str());
  std::string xmlname;
  while (infile >> xmlname) {
    lines_.push_back(xmlname);
  }
  CHECK(!lines_.empty()) << "File is empty.";
  // 随机乱序
  if (unified_data_param.shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLists();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  if (unified_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % unified_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " instances.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // data
  const int batch_size = unified_data_param.batch_size();
  const int height = unified_data_transform_param_.resized_height();
  const int width = unified_data_transform_param_.resized_width();
  
  // box -> 9
  // kps -> 2 + 54 (has_kps, num_kps, 18*3)
  // mask-> 1 + HW (has_mask, HW)
  
  top[0]->Reshape(batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  // label
  // bindex,cid,pid,is_diff,iscrowd,xmin,ymin,xmax,ymax
  // has_kps, num_kps, 18*3(x,y,v)
  // has_mask, h*w
  // 12+54(66)+h*w
  top_offs_ = 9; 
  top[1]->Reshape(1,1,1,top_offs_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(1,1,1,top_offs_);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
}

template <typename Dtype>
void FaceDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void FaceDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  // CHECK(this->transformed_data_.count());
  // 获取batchsize和输出h/w
  const UnifiedDataParameter& unified_data_param = this->layer_param_.unified_data_param();
  string xml_root = unified_data_param.xml_root();
  const int batch_size = unified_data_param.batch_size();
  const int height = unified_data_transform_param_.resized_height();
  const int width = unified_data_transform_param_.resized_width();
  vector<cv::Mat> images_all;
  vector<vector<BBoxData<Dtype> > > bboxes_all;
  const int lines_size = lines_.size();
  // perform BATCHSIZE samples
  int num_p = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_path = xml_root + '/' + lines_[lines_id_];
    // LOG(INFO)<<"hzw xml_path"<<xml_path;
    // Read Anno
    AnnoData<Dtype> anno;
    ReadAnnoDataFromXml(item_id, xml_path, xml_root, &anno);
    // 转换
    cv::Mat image;
    vector<BBoxData<Dtype> > bboxes;

    
    BoundingBox<Dtype> crop_bbox;
    bool doflip;
    face_data_transformer_->Transform(anno, &image, &bboxes, &crop_bbox, &doflip);
    num_p += bboxes.size();
    images_all.push_back(image);
    // 增加parts的标注

    bboxes_all.push_back(bboxes);

    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (unified_data_param.shuffle()) {
        ShuffleLists();
      }
    }
    if (item_id >= batch_size - 1) {
      if (num_p == 0) {
        // re-run a minibatch
        item_id = -1;
        images_all.clear();
        bboxes_all.clear();
      }
    }
  }
  // check data
  CHECK_EQ(images_all.size(), batch_size);
  CHECK_EQ(bboxes_all.size(), batch_size);//hzw
  /**
   * visualize
   */
  // if (unified_data_transform_param_.visualize()) {
  //   for (int i = 0; i < images_all.size(); ++i) {
  //     unified_data_transformer_->visualize(images_all[i], bboxes_all[i], kpses_all[i], masks_all[i]);
  //   }
  // }
  // copy data to top[0/1]
  // top[0]
  batch->data_.Reshape(batch_size,3,height,width);
  Dtype* const top_data = batch->data_.mutable_cpu_data();
  bool normalize = false;
  const int offset = height * width;
  for (int i = 0; i < batch_size; ++i) {
    Dtype* top_data_item = top_data + i * 3 * offset;
    cv::Mat& image = images_all[i];
    CHECK_EQ(image.rows, height);
    CHECK_EQ(image.cols, width);
    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        const cv::Vec3b& rgb = image.at<cv::Vec3b>(y, x);
        if (normalize) {
          top_data_item[y * image.cols + x] = (rgb[0] - 128)/256.0;
          top_data_item[offset + y * image.cols + x] = (rgb[1] - 128)/256.0;
          top_data_item[2 * offset + y * image.cols + x] = (rgb[2] - 128)/256.0;
        } else {
          top_data_item[y * image.cols + x] = rgb[0] - mean_values_[0];
          top_data_item[offset + y * image.cols + x] = rgb[1] - mean_values_[1];
          top_data_item[2 * offset + y * image.cols + x] = rgb[2] - mean_values_[2];
        }
      }
    }
  }
  // top[1]
  int num_gt = 0;
  for (int i = 0; i < bboxes_all.size(); ++i) {
    num_gt += bboxes_all[i].size();
  }
  CHECK_GT(num_gt, 0) << "Found No Ground-Truth.";
  batch->label_.Reshape(1,1,num_gt,top_offs_);
  Dtype* top_label = batch->label_.mutable_cpu_data();
  const int offs = top_offs_;
  int count = 0;
  for (int i = 0; i < bboxes_all.size(); ++i) {
    for (int j = 0; j < bboxes_all[i].size(); ++j) {
      Dtype* top_data_ptr = top_label + offs * count;
      int idx = 0;
      // bbox -> 9
      BBoxData<Dtype>& box = bboxes_all[i][j];
      top_data_ptr[idx++] = box.bindex;
      top_data_ptr[idx++] = box.cid;
      top_data_ptr[idx++] = box.pid;
      top_data_ptr[idx++] = box.is_diff;
      top_data_ptr[idx++] = box.iscrowd;
      top_data_ptr[idx++] = box.bbox.x1_;
      top_data_ptr[idx++] = box.bbox.y1_;
      top_data_ptr[idx++] = box.bbox.x2_;
      top_data_ptr[idx++] = box.bbox.y2_;
      // kps
      // LOG(INFO)<<"hzw box.bindex:"<<box.bindex<<"box.cid:"<<box.cid<<"box.pid:"<<box.pid<<box.bbox.x1_<<box.bbox.y1_<<box.bbox.x2_<<box.bbox.y2_;

      // mask
      
      CHECK_EQ(idx, top_offs_);
      count++;
    }
  }
  CHECK_EQ(count,num_gt) << "Size unmatched.";
}

template <typename Dtype>
void FaceDataLayer<Dtype>::ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                                                  AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  anno->num_person = pt.get<int>("Annotations.NumPart");
  anno->instances.clear();
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    // sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    // sprintf(temp_pid, "Annotations.Object_%d.pid", 0);
    // sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i+1);
    // sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);

    // bindex & cid & pid
    ins.bindex = bindex;
    // ins.cid = pt.get<int>(temp_cid); // 0
    // ins.pid = pt.get<int>(temp_pid);
    // is_diff
    ins.cid = 0;
    ins.pid = i;
    ins.is_diff = false;
    // iscrowd
    ins.iscrowd = false;
    // filter crowd & diff
    if (ins.iscrowd || ins.is_diff) continue;
    // bbox: must be defined
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_xmax);
    ins.bbox.y2_ = pt.get<Dtype>(temp_ymax);
   
    
    anno->instances.push_back(ins);
  }
}

INSTANTIATE_CLASS(FaceDataLayer);
REGISTER_LAYER_CLASS(FaceData);
}  // namespace caffe
