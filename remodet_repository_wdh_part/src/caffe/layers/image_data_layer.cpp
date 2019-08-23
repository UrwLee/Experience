#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ImageDataParameter& image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const bool is_color  = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // 检查尺寸
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // 获取图片路径及标注文件路径列表
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  std::string filename;
  std::string labelname;
  while (infile >> filename >> labelname) {
    lines_.push_back(std::make_pair(filename, labelname));
  }

  CHECK(!lines_.empty()) << "File is empty";

  // 随机乱序文件的编号
  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  // 随机跳过最前面的n个样本
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  // 直接读取一副图片,用于初始化数据变换器
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  // 确定数据变换器的尺寸
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  // 确定batchsize
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  //修改prefetch中的data数据的尺寸
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  // top[0]也进行相应修改
  top[0]->Reshape(top_shape);
  // 定义输出尺寸[0]
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // 下面定义label的尺寸!
  // 每个gtbox有13个值
  vector<int> label_shape(4, 1);
  label_shape[0] = 1;
  label_shape[1] = 1;
  label_shape[2] = 1;
  label_shape[3] = 9;
  // 初始化prefetch
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  // 初始化top[1]
  top[1]->Reshape(label_shape);

  // batch id init
  batch_id_ = 0;
  // batch sampler
  for (int i = 0; i < this->layer_param_.transform_param().batch_sampler_size(); ++i) {
    batch_samplers_.push_back(this->layer_param_.transform_param().batch_sampler(i));
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  // std::random_shuffle(lines_.begin(), lines_.end());
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  batch_id_ ++;
  CPUTimer batch_timer;
  batch_timer.Start();

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  TransformationParameter transform_param = this->layer_param_.transform_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  const float body_boxsize_thre = transform_param.boxsize_threshold();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = NULL;

  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;
  // datum scales
  const int lines_size = lines_.size();
  // 载入batch个样本
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // 读入图片
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    int src_img_width = cv_img.cols;
    int src_img_height = cv_img.rows;
    int src_img_channels = cv_img.channels();

    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

    // 读取xml标记
    vector<PersonBBox> item_anno_person;
    vector<AnnotationGroup> item_anno;
    if (!ReadPersonAnnotationFromXml(root_folder + lines_[lines_id_].second, src_img_height,
        src_img_width, src_img_channels, 0, &item_anno_person)) {
        LOG(FATAL) << "Error occurred in reading annotation file: "
                    << root_folder + lines_[lines_id_].second;
    }
    this->data_transformer_->TransformAnnotationOfPerson(item_anno_person, &item_anno);

    // Distortion：颜色失真处理，无需标记转换
    if (transform_param.has_distort_param()) {
      cv_img = DistortImage(cv_img,transform_param.distort_param());
    }
    // Expansion: 随机膨胀，标记进行转换
    if (transform_param.has_expand_param()) {
      vector<AnnotationGroup> expand_item_anno;
      NormalizedBBox expand_bbox;
      cv_img = ExpandImage(cv_img, transform_param.expand_param(),&expand_bbox);
      this->data_transformer_->TransformAnnotation(item_anno,
                  expand_bbox,false,0,&expand_item_anno);
      item_anno = expand_item_anno;
    }
    // Random-Sampler
    if (batch_samplers_.size() > 0) {
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(item_anno, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        vector<AnnotationGroup> crop_item_anno;
        int idx = caffe_rng_rand() % sampled_bboxes.size();
        cv_img = ApplyCrop(cv_img,sampled_bboxes[idx]);
        this->data_transformer_->TransformAnnotation(item_anno,
                  sampled_bboxes[idx],false,body_boxsize_thre,&crop_item_anno);
        item_anno = crop_item_anno;
      }
    }

    /**
     * Release 2: check if the Current image has no boxes. if no boxes found: get to the next image.
     * @ZhangM, 01.19.15:00
     */
    int boxes_found = 0;
    for (int g = 0; g < item_anno.size(); ++g) {
       boxes_found += item_anno[g].annotation_size();
       if (boxes_found > 0) break;
    }
    if (boxes_found == 0) {
      item_id--;
      lines_id_++;
      if (lines_id_ >= lines_size) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.image_data_param().shuffle()) {
          ShuffleImages();
        }
      }
      continue;
    }
    // end of modifications

    // Resize: 网络标准输入尺寸
    if (transform_param.has_resize_param()) {
      cv_img = ApplyResize(cv_img, transform_param.resize_param());
    } else {
      LOG(FATAL) << "ResizeParameter must be defined for fixed input dims.";
    }
    // Mirror and Scale / Mean-sub
    vector<AnnotationGroup> transformed_item_anno;
    bool do_mirror;
    this->data_transformer_->EasyTransform(cv_img,&(this->transformed_data_),&do_mirror);
    this->data_transformer_->TransformAnnotation(item_anno,
              UnitBBox(),do_mirror,0,&transformed_item_anno);

    // 统计gtboxes的数量
    for (int g = 0; g < transformed_item_anno.size(); ++g) {
      num_bboxes += transformed_item_anno[g].annotation_size();
    }

    all_anno[item_id] = transformed_item_anno;

    // 指向下一张样本
    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  //下面开始输出top[1]
  vector<int> label_shape(4);
  label_shape[0] = 1;
  label_shape[1] = 1;
  label_shape[3] = 9;
  // LOG(INFO) << "[Batch ID] -> " << batch_id_;
  if (num_bboxes == 0) {
    label_shape[2] = 1;
    batch->label_.Reshape(label_shape);
    caffe_set<Dtype>(9, -1, batch->label_.mutable_cpu_data());
    LOG(INFO) << "[Batch Ground Truth] No boxes are found.";
  } else {
    // 正常,开始输出
    label_shape[2] = num_bboxes;
    batch->label_.Reshape(label_shape);
    // 指向数据指针
    prefetch_label = batch->label_.mutable_cpu_data();
    int idx = 0;
    // 遍历所有样本
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      const vector<AnnotationGroup>& anno_group = all_anno[item_id];
      for (int i = 0; i < anno_group.size(); ++i) {
        const AnnotationGroup& anno = anno_group[i];
        for (int j = 0; j < anno.annotation_size(); ++j) {
          const Annotation& an = anno.annotation(j);
          const NormalizedBBox& bbox = an.bbox();
          /**
           * 0: -> bindex
           * 1: -> cid == 0
           * 2: -> pid == an.instance_id();
           * 3: -> is_diff -> bbox.difficult();
           * 4: -> iscrowd: 0
           * 5: xmin
           * 6: ymin
           * 7: xmax
           * 8: ymax
           */
          prefetch_label[idx++] = item_id;
          prefetch_label[idx++] = 0;
          prefetch_label[idx++] = an.instance_id();
          prefetch_label[idx++] = bbox.difficult();
          prefetch_label[idx++] = 0;
          prefetch_label[idx++] = bbox.xmin();
          prefetch_label[idx++] = bbox.ymin();
          prefetch_label[idx++] = bbox.xmax();
          prefetch_label[idx++] = bbox.ymax();
        }
      }
    }
    // LOG(INFO) << "[Batch Ground Truth] Done.";
  }
  batch_timer.Stop();
  DLOG(INFO) << "   Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);
}  // namespace caffe
#endif  // USE_OPENCV
