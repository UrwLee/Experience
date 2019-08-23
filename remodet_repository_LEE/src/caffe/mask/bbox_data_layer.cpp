#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/mask/bbox_data_transformer.hpp"
#include "caffe/mask/bbox_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace caffe {

using namespace boost::property_tree;

template <typename Dtype>
BBoxDataLayer<Dtype>::~BBoxDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void BBoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  bbox_data_transformer_.reset(
    new BBoxDataTransformer<Dtype>(bbox_data_transform_param_, this->phase_));
  // 获取数据Layer参数
  // 仍然使用unified_data_param
  const UnifiedDataParameter& unified_data_param = this->layer_param_.unified_data_param();

  //clipGTorNot
  clip_ignoreGT_ = unified_data_param.clip_ignoregt();
  CHECK_EQ(unified_data_param.mean_value_size(), 3);
  CHECK_NE(bbox_data_transform_param_.sample_sixteennine(), bbox_data_transform_param_.sample_ninesixteen());
  for (int i = 0; i < 3; ++i) {
    mean_values_.push_back(unified_data_param.mean_value(i));
  }
  if (unified_data_param.has_xml_list()) {
    string xml_list = unified_data_param.xml_list();
    string xml_root = unified_data_param.xml_root();
    LOG(INFO) << "Opening file " << xml_list;
    std::ifstream infile(xml_list.c_str());
    CHECK(infile.good()) << "Failed to open file " << xml_list;
    std::string xmlname;
    while (infile >> xmlname) {
      lines_.push_back(make_pair(xml_root, xmlname));
    }
  } else {
    LOG(INFO) << "size of unified_data_param.xml_list_multiple_size() " << unified_data_param.xml_list_multiple_size();
    for (int i = 0; i < unified_data_param.xml_list_multiple_size(); i++) {
      string xml_list = unified_data_param.xml_list_multiple(i);
      string xml_root = unified_data_param.xml_root_multiple(i);
      LOG(INFO) << "Opening file \"" << xml_list << "\"";
      std::ifstream infile(xml_list.c_str());
      CHECK(infile.good()) << "Failed to open file " << xml_list;
      std::string xmlname;
      while (infile >> xmlname) {
        lines_.push_back(make_pair(xml_root, xmlname));
      }
      LOG(INFO) << "Finished Reading " << xml_list;
    }
  }
  flag_hisimap_ = unified_data_param.has_hisi_data_maps();
  if (flag_hisimap_) {
    string hisi_data_maps = unified_data_param.hisi_data_maps();
    makeHisiDataMaps(hisi_data_maps, &maps_);
  }
  base_bindex_ = 0;
  if (unified_data_param.has_base_bindex()) {
    base_bindex_ = unified_data_param.base_bindex();
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
  add_parts_ = unified_data_param.add_parts();
  const int height = bbox_data_transform_param_.resized_height();
  const int width = bbox_data_transform_param_.resized_width();
  if (clip_ignoreGT_)
  {
    top[0]->Reshape(batch_size, 6, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, 6, height, width);
    }
  } else {
    top[0]->Reshape(batch_size, 3, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
    }
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();
  // label
  if (unified_data_param.flag_imginfo()) {
    ndim_label_ = 9 + 4;
  } else {
    ndim_label_ = 9;
  }
  top[1]->Reshape(1, 1, 1, ndim_label_);


  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {

    this->prefetch_[i].label_.Reshape(1, 1, 1, ndim_label_);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
            << top[1]->channels() << "," << top[1]->height() << ","
            << top[1]->width();
  for (int i = 0; i < 10; i++) {
    check_area_.push_back(0);
  }
}

template <typename Dtype>
void BBoxDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void BBoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  const UnifiedDataParameter& unified_data_param = this->layer_param_.unified_data_param();
  // string xml_root = unified_data_param.xml_root();
  const int batch_size = unified_data_param.batch_size();
  const int height = bbox_data_transform_param_.resized_height();
  const int width = bbox_data_transform_param_.resized_width();
  vector<cv::Mat> images_all;
  vector<vector<BBoxData<Dtype> > > bboxes_all;
  const int lines_size = lines_.size();
  // perform BATCHSIZE samples
  vector<BoundingBox<Dtype> > image_bboxes;
  int num_p = 0;
  //#######################################
  int num_pic_gt = 0;
  int num_pic_no_gt = 0; 
  //#######################################
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_root = lines_[lines_id_].first;
    string xml_path = xml_root + '/' + lines_[lines_id_].second;
    // LOG(INFO)<<xml_root<<" "<<xml_path;
    // Read Anno
    AnnoData<Dtype> anno;
    if (unified_data_param.use_torsowithhead()) {
      ReadTHAnnoDataFromXml(item_id, xml_path, xml_root, &anno);
    } else {
      ReadAnnoDataFromXml(item_id, xml_path, xml_root, &anno);
    }
    //#######################################
    // if(anno.num_person==0) num_pic_no_gt += 1;
    // else num_pic_gt+=1;
    //#######################################
    // 转换
    cv::Mat image;
    vector<BBoxData<Dtype> > bboxes;
    BoundingBox<Dtype> crop_bbox;
    BoundingBox<Dtype> image_bbox;
    bool doflip;
    if (clip_ignoreGT_)
    {
      cv::Mat mask8(height, width, CV_8UC1, cv::Scalar(1));
      cv::Mat mask16(height, width, CV_8UC1, cv::Scalar(1));
      cv::Mat mask32(height, width, CV_8UC1, cv::Scalar(1));
      std::vector<cv::Mat> imagelist;

      //no person add a bbox cover the whole pic
      if (anno.num_person == 0 && this->phase_ == TRAIN) {
        bbox_data_transformer_->Transform(anno, &image, &doflip);
        image_bbox.x1_ = 0;
        image_bbox.y1_ = 0;
        image_bbox.x2_ = width;
        image_bbox.y2_ = height;
      } else {
        std::vector<BoundingBox<Dtype> > ignore_bboxes;
        bbox_data_transformer_->Transform(anno, &image, &bboxes, &crop_bbox, &doflip, &image_bbox, ignore_bboxes);
        num_p += bboxes.size();
        bboxes_all.push_back(bboxes);
        typename std::vector<BoundingBox<Dtype> >::iterator it;
        for (it = ignore_bboxes.begin(); it != ignore_bboxes.end(); ++it) {
          BoundingBox<Dtype>& bbox_a = *it;
          int ww = image.cols;
          int hh = image.rows;
          cv::rectangle(mask8, cv::Point((int)(bbox_a.x1_ * (ww / 8)), (int)(bbox_a.y1_ * (hh / 8))),
                        cv::Point((int)(bbox_a.x2_ * (ww / 8)), (int)(bbox_a.y2_ * (hh / 8))), cv::Scalar(0), CV_FILLED);
          cv::rectangle(mask16, cv::Point((int)(bbox_a.x1_ * (ww / 16)), (int)(bbox_a.y1_ * (hh / 16))),
                        cv::Point((int)(bbox_a.x2_ * (ww / 16)), (int)(bbox_a.y2_ * (hh / 16))), cv::Scalar(0), CV_FILLED);
          cv::rectangle(mask32, cv::Point((int)(bbox_a.x1_ * (ww / 32)), (int)(bbox_a.y1_ * (hh / 32))),
                        cv::Point((int)(bbox_a.x2_ * (ww / 32)), (int)(bbox_a.y2_ * (hh / 32))), cv::Scalar(0), CV_FILLED);
        }
      }
      imagelist.push_back(image);
      imagelist.push_back(mask8);
      imagelist.push_back(mask16);
      imagelist.push_back(mask32);
      cv::merge(imagelist, image);
    } else {
      if (anno.num_person == 0 && this->phase_ == TRAIN) {
        bbox_data_transformer_->Transform(anno, &image, &doflip);
        image_bbox.x1_ = 0 ;
        image_bbox.y1_ = 0;
        image_bbox.x2_ = width;
        image_bbox.y2_ = height;
      } else {
        bbox_data_transformer_->Transform(anno, &image, &bboxes, &crop_bbox, &doflip, &image_bbox);
        num_p += bboxes.size();
        bboxes_all.push_back(bboxes);
      }
    }
    image_bboxes.push_back(image_bbox);
    images_all.push_back(image);
    lines_id_++;
    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (unified_data_param.shuffle()) {
        ShuffleLists();
      }
    }
    if (item_id >= batch_size - 1) {
      //#######################################
      // LOG(INFO)<<"Num pic gt minibatch:"<<num_pic_gt;
      // LOG(INFO)<<"Num pic no gt minibatch:"<<num_pic_no_gt;
      // LOG(INFO)<<"Num gts one minibatch:"<<num_p;
      //#######################################
      if (num_p == 0) {
        // re-run a minibatch
        item_id = -1;
        images_all.clear();
        bboxes_all.clear();
      }
    }
    if (lines_id_ % 960 == 0) {
      LOG(INFO) << check_area_[0] << " " << check_area_[1] << " " << check_area_[2] << " "
                << check_area_[3] << " " << check_area_[4] << " " << check_area_[5] << " " << check_area_[6] << " " << check_area_[7] << " " << check_area_[8] << " " << check_area_[9];
    }

  }
  // check data
  CHECK_EQ(images_all.size(), batch_size);
  CHECK_LE(bboxes_all.size(), batch_size);
  // LOG(INFO) << bboxes_all.size() << " " << batch_size;

  // if (bbox_data_transform_param_.visualize()) {
  //   for (int i = 0; i < images_all.size(); ++i) {
  //     unified_data_transformer_->visualize(images_all[i], bboxes_all[i], kpses_all[i], masks_all[i]);
  //   }
  // }
  // copy data to top[0/1]
  // top[0]
  if (clip_ignoreGT_)
  {
    batch->data_.Reshape(batch_size, 6, height, width);
    Dtype* const top_data = batch->data_.mutable_cpu_data();
    for (int i = 0; i < batch_size; ++i) {
      int top_index;
      cv::Mat& image = images_all[i];
      for (int h = 0; h < height; ++h) {
        const uchar* ptr = image.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < width; ++w) {
          for (int c = 0; c < 6; ++c) {
            top_index = ((i * 6 + c) * height + h) * width + w;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            if (c < 3) {
              top_data[top_index] = pixel - mean_values_[c];
            } else {
              top_data[top_index] = pixel;
            }
          }
        }
      }
    }
  } else {
    batch->data_.Reshape(batch_size, 3, height, width);
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
          if (flag_hisimap_) {
            top_data_item[y * image.cols + x] = maps_[rgb[0] - mean_values_[0]];
            top_data_item[offset + y * image.cols + x] = maps_[rgb[1] - mean_values_[1]];
            top_data_item[2 * offset + y * image.cols + x] = maps_[rgb[2] - mean_values_[2]];
          } else {
            if (normalize) {
              top_data_item[y * image.cols + x] = (rgb[0] - 128) / 256.0;
              top_data_item[offset + y * image.cols + x] = (rgb[1] - 128) / 256.0;
              top_data_item[2 * offset + y * image.cols + x] = (rgb[2] - 128) / 256.0;
            } else {
              top_data_item[y * image.cols + x] = rgb[0] - mean_values_[0];
              top_data_item[offset + y * image.cols + x] = rgb[1] - mean_values_[1];
              top_data_item[2 * offset + y * image.cols + x] = rgb[2] - mean_values_[2];
            }
          }
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

  // batch->label_.Reshape(1,1,num_gt,9 + 4);
  batch->label_.Reshape(1, 1, num_gt, ndim_label_);

  Dtype* top_label = batch->label_.mutable_cpu_data();
  int count = 0;
  int idx = 0;
  for (int i = 0; i < bboxes_all.size(); ++i) {
    for (int j = 0; j < bboxes_all[i].size(); ++j) {
      BBoxData<Dtype>& box = bboxes_all[i][j];

      top_label[idx++] = box.bindex + base_bindex_;
      top_label[idx++] = box.cid;
      top_label[idx++] = box.pid;
      top_label[idx++] = box.is_diff;
      top_label[idx++] = box.iscrowd;
      top_label[idx++] = box.bbox.x1_;
      top_label[idx++] = box.bbox.y1_;
      top_label[idx++] = box.bbox.x2_;
      top_label[idx++] = box.bbox.y2_;
      if (unified_data_param.flag_imginfo()) {
        top_label[idx++] = (Dtype)(image_bboxes[box.bindex].x1_ * width);
        top_label[idx++] = (Dtype)(image_bboxes[box.bindex].y1_ * height);
        top_label[idx++] = (Dtype)(image_bboxes[box.bindex].x2_ * width);
        top_label[idx++] = (Dtype)(image_bboxes[box.bindex].y2_ * height);
      }
      count++;

      if (box.cid == 1 ||  box.cid == 3) {
        float area = box.bbox.compute_area();
        int level = -1;
        if (area > 0)
          level ++;
        if (area > 0.0016)
          level ++;
        if (area > 0.0064)
          level ++;
        if (area > 0.01)
          level ++;
        if (area > 0.1)
          level ++;
        if (area > 0.15)
          level ++;
        if (area > 0.25)
          level ++;
        if (area > 0.45)
          level ++;
        if (area > 0.65)
          level ++;
        if (area > 0.85)
          level ++;
        check_area_[level] ++;
      }

    }
  }
  CHECK_EQ(count, num_gt) << "Size unmatched.";



}

template <typename Dtype>
void BBoxDataLayer<Dtype>::ReadTHAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
    AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  bool THfound = false;
  if (anno->dataset == "AICDataWithTorse") {
    THfound = true;
  }
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  try {
    anno->num_person = pt.get<int>("Annotations.NumPerson");
  } catch (const ptree_error &e) {
    anno->num_person = pt.get<int>("Annotations.NumPart");
  }
  anno->instances.clear();
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_cid[128], temp_pid[128], temp_iscrowd[128], temp_is_diff[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    char temp_thxmin[128], temp_thymin[128], temp_thxmax[128], temp_thymax[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i + 1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i + 1);
    sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i + 1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i + 1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i + 1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i + 1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i + 1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i + 1);
    sprintf(temp_thxmin, "Annotations.Object_%d.thxmin", i + 1);
    sprintf(temp_thymin, "Annotations.Object_%d.thymin", i + 1);
    sprintf(temp_thxmax, "Annotations.Object_%d.thxmax", i + 1);
    sprintf(temp_thymax, "Annotations.Object_%d.thymax", i + 1);
    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid);
    if (!add_parts_) {
      ins.cid = 0;
    }
    try {
      ins.pid = pt.get<int>(temp_pid);
    } catch (const ptree_error &e) {
      ins.pid = 0;
    }
    // is_diff
    try {
      ins.is_diff = pt.get<int>(temp_is_diff) == 0 ? false : true;
    } catch (const ptree_error &e) {
      ins.is_diff = false;
    }
    // iscrowd
    try {
      ins.iscrowd = pt.get<int>(temp_iscrowd) == 0 ? false : true;
    } catch (const ptree_error &e) {
      ins.iscrowd = false;
    }
    // filter crowd & diff
    if (ins.iscrowd || ins.is_diff) continue;
    // bbox: must be defined
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_xmax);
    ins.bbox.y2_ = pt.get<Dtype>(temp_ymax);
    if (ins.cid == 0 && THfound) {
      ins.THbbox.x1_ = pt.get<Dtype>(temp_thxmin);
      ins.THbbox.y1_ = pt.get<Dtype>(temp_thymin);
      ins.THbbox.x2_ = pt.get<Dtype>(temp_thxmax);
      ins.THbbox.y2_ = pt.get<Dtype>(temp_thymax);
    }
    anno->instances.push_back(ins);
  }
}

template <typename Dtype>
void BBoxDataLayer<Dtype>::ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
    AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  try {
    anno->num_person = pt.get<int>("Annotations.NumPerson");
  } catch (const ptree_error &e) {
    anno->num_person = pt.get<int>("Annotations.NumPart");
  }
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  anno->instances.clear();
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_cid[128], temp_pid[128], temp_iscrowd[128], temp_is_diff[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i + 1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i + 1);
    sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i + 1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i + 1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i + 1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i + 1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i + 1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i + 1);
    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid);
    if (!add_parts_) {
      ins.cid = 0;
    }
    try {
      ins.pid = pt.get<int>(temp_pid);
    } catch (const ptree_error &e) {
      ins.pid = 0;
    }
    // is_diff
    try {
      ins.is_diff = pt.get<int>(temp_is_diff) == 0 ? false : true;
    } catch (const ptree_error &e) {
      ins.is_diff = false;
    }
    // iscrowd
    try {
      ins.iscrowd = pt.get<int>(temp_iscrowd) == 0 ? false : true;
    } catch (const ptree_error &e) {
      ins.iscrowd = false;
    }
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

INSTANTIATE_CLASS(BBoxDataLayer);
REGISTER_LAYER_CLASS(BBoxData);
}  // namespace caffe
