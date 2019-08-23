#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/mask/unified_data_transformer.hpp"
#include "caffe/mask/unified_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace caffe {

using namespace boost::property_tree;

template <typename Dtype>
UnifiedDataLayer<Dtype>::~UnifiedDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void UnifiedDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  unified_data_transformer_.reset(
     new UnifiedDataTransformer<Dtype>(unified_data_transform_param_, this->phase_));
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
  add_parts_ = unified_data_param.add_parts();
  if (add_parts_) {
    // we use parts_xml_dir_ + '/' + 'image_id' + "_Parts.xml" -> parts label
    parts_xml_dir_ = unified_data_param.parts_xml_dir();
  }
  // box -> 9
  // kps -> 2 + 54 (has_kps, num_kps, 18*3)
  // mask-> 1 + HW (has_mask, HW)
  add_kps_ = unified_data_param.add_kps();
  add_mask_ = unified_data_param.add_mask();
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
  top_offs_ = 9 + 56 * add_kps_ + (height * width + 1) * add_mask_;
  top[1]->Reshape(1,1,1,top_offs_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(1,1,1,top_offs_);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
}

template <typename Dtype>
void UnifiedDataLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void UnifiedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  vector<vector<KpsData<Dtype> > > kpses_all;
  vector<vector<MaskData<Dtype> > > masks_all;
  const int lines_size = lines_.size();
  // perform BATCHSIZE samples
  int num_p = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_path = xml_root + '/' + lines_[lines_id_];
    // Read Anno
    AnnoData<Dtype> anno;
    ReadAnnoDataFromXml(item_id, xml_path, xml_root, &anno);
    // 转换
    cv::Mat image;
    vector<BBoxData<Dtype> > bboxes;
    vector<KpsData<Dtype> > kpses;
    vector<MaskData<Dtype> > masks;
    BoundingBox<Dtype> crop_bbox;
    bool doflip;
    unified_data_transformer_->Transform(anno, &image, &bboxes, &kpses, &masks, &crop_bbox, &doflip);
    num_p += bboxes.size();
    images_all.push_back(image);
    // 增加parts的标注
    if (add_parts_) {
      const int delim_pos = lines_[lines_id_].find_last_of("/");
      const string& xml_name = lines_[lines_id_].substr(delim_pos+1, lines_[lines_id_].length());
      const int point_pos = xml_name.find_last_of('.');
      const string& xname = xml_name.substr(0,point_pos);
      const string xml_parts_path = parts_xml_dir_ + '/' + xname + "_Parts.xml";
      vector<LabeledBBox<Dtype> > labeled_boxes;
      ReadPartBoxesFromXml(item_id, xml_parts_path, xml_root, &labeled_boxes);//hzw
      // copy bboxes to bboxes
      vector<BBoxData<Dtype> > part_bboxes;
      unified_data_transformer_->ApplyCrop(crop_bbox, labeled_boxes, &part_bboxes);
      if (doflip) {
        unified_data_transformer_->FlipBoxes(&part_bboxes);
      }
      for (int i = 0; i < part_bboxes.size(); ++i) {
        bboxes.push_back(part_bboxes[i]);
        KpsData<Dtype> kps;
        MaskData<Dtype> mask;
        kpses.push_back(kps);
        masks.push_back(mask);
      }
    }
    CHECK_EQ(bboxes.size(), kpses.size());
    CHECK_EQ(bboxes.size(), masks.size());
    bboxes_all.push_back(bboxes);
    kpses_all.push_back(kpses);
    masks_all.push_back(masks);
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
        kpses_all.clear();
        masks_all.clear();
      }
    }
  }
  // check data
  CHECK_EQ(images_all.size(), batch_size);
  CHECK_EQ(bboxes_all.size(), batch_size);//hzw
  CHECK_EQ(bboxes_all.size(), kpses_all.size());
  CHECK_EQ(bboxes_all.size(), masks_all.size());
  /**
   * visualize
   */
  if (unified_data_transform_param_.visualize()) {
    for (int i = 0; i < images_all.size(); ++i) {
      unified_data_transformer_->visualize(images_all[i], bboxes_all[i], kpses_all[i], masks_all[i]);
    }
  }
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
      if (add_kps_) {
        if (box.cid != 0) {
          // parts
          top_data_ptr[idx++] = 0;
          top_data_ptr[idx++] = 0;
          for (int k = 0; k < 18; ++k) {
            top_data_ptr[idx++] = 0;
            top_data_ptr[idx++] = 0;
            top_data_ptr[idx++] = 2;
          }
        } else {
          KpsData<Dtype>& kps = kpses_all[i][j];
          CHECK_EQ(kps.bindex, box.bindex);
          CHECK_EQ(kps.cid, box.cid);
          CHECK_EQ(kps.pid, box.pid);
          CHECK_EQ(kps.is_diff, box.is_diff);
          CHECK_EQ(kps.iscrowd, box.iscrowd);
          top_data_ptr[idx++] = kps.has_kps;
          top_data_ptr[idx++] = kps.num_kps;
          CHECK_EQ(kps.joint.joints.size(), 18);
          for (int k = 0; k < 18; ++k) {
            top_data_ptr[idx++] = kps.joint.joints[k].x;
            top_data_ptr[idx++] = kps.joint.joints[k].y;
            top_data_ptr[idx++] = kps.joint.isVisible[k];
          }
        }
      }
      // mask
      if (add_mask_) {
        if (box.cid != 0) {
          // parts
          top_data_ptr[idx++] = 0;
          caffe_set(height*width, Dtype(0), top_data_ptr + idx);
          idx += height*width;
        } else {
          MaskData<Dtype>& mask_e = masks_all[i][j];
          CHECK_EQ(mask_e.bindex, box.bindex);
          CHECK_EQ(mask_e.cid, box.cid);
          CHECK_EQ(mask_e.pid, box.pid);
          CHECK_EQ(mask_e.is_diff, box.is_diff);
          CHECK_EQ(mask_e.iscrowd, box.iscrowd);
          top_data_ptr[idx++] = mask_e.has_mask;
          CHECK_EQ(mask_e.mask.rows, height);
          CHECK_EQ(mask_e.mask.cols, width);
          cv::Mat& mask_img = mask_e.mask;
          for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
              top_data_ptr[idx++] = mask_img.at<uchar>(y,x);
            }
          }
        }
      }
      CHECK_EQ(idx, top_offs_);
      count++;
    }
  }
  CHECK_EQ(count,num_gt) << "Size unmatched.";
}

template <typename Dtype>
void UnifiedDataLayer<Dtype>::ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                                                  AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  anno->num_person = pt.get<int>("Annotations.NumPerson");
  anno->instances.clear();
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_cid[128], temp_pid[128], temp_iscrowd[128], temp_is_diff[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    char temp_mask_included[128], temp_mask_path[128];
    char temp_kps_included[128], temp_num_kps[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i+1);
    sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i+1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);
    sprintf(temp_mask_included, "Annotations.Object_%d.mask_included", i+1);
    sprintf(temp_mask_path, "Annotations.Object_%d.mask_path", i+1);
    sprintf(temp_kps_included, "Annotations.Object_%d.kps_included", i+1);
    sprintf(temp_num_kps, "Annotations.Object_%d.num_kps", i+1);
    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid); // 0
    ins.pid = pt.get<int>(temp_pid);
    // is_diff
    try {
      ins.is_diff = pt.get<int>(temp_is_diff) == 0 ? false : true;
    } catch (const ptree_error &e) {
      DLOG(WARNING) << "When parsing " << xml_file << ": " << e.what();
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
    // mask
    try {
      ins.mask_included = pt.get<int>(temp_mask_included) == 0 ? false : true;
      ins.mask_path = root_dir + '/' + pt.get<string>(temp_mask_path);
    } catch (const ptree_error &e) {
      ins.mask_included = false;
      ins.mask_path = "None";
    }
    // kps
    try {
      ins.kps_included = pt.get<int>(temp_kps_included) == 0 ? false : true;
      ins.num_kps = pt.get<int>(temp_num_kps);
      if (anno->dataset.find("COCO") != std::string::npos) {
        ins.joint.joints.resize(17);
        ins.joint.isVisible.resize(17);
        for (int k = 0; k < 17; ++k) {
          char temp_x[128], temp_y[128], temp_vis[128];
          sprintf(temp_x, "Annotations.Object_%d.joint.kp_%d.x", i+1,k+1);
          sprintf(temp_y, "Annotations.Object_%d.joint.kp_%d.y", i+1,k+1);
          sprintf(temp_vis, "Annotations.Object_%d.joint.kp_%d.vis", i+1,k+1);
          ins.joint.joints[k].x = pt.get<Dtype>(temp_x);
          ins.joint.joints[k].y = pt.get<Dtype>(temp_y);
          ins.joint.isVisible[k] = pt.get<int>(temp_vis);
        }
      } else {
        LOG(FATAL) << "Not support for dataset-type: " << anno->dataset;
      }
    } catch (const ptree_error &e) {
      ins.kps_included = false;
      ins.num_kps = 0;
      // initial
      if (anno->dataset.find("COCO") != std::string::npos) {
        ins.joint.joints.resize(17);
        ins.joint.isVisible.resize(17);
        for (int k = 0; k < 17; ++k) {
          ins.joint.joints[k].x = 0;
          ins.joint.joints[k].y = 0;
          ins.joint.isVisible[k] = 2;
        }
      } else {
        LOG(FATAL) << "Not support for dataset-type: " << anno->dataset;
      }
    }
    anno->instances.push_back(ins);
  }
}

template <typename Dtype>
void UnifiedDataLayer<Dtype>::ReadPartBoxesFromXml(const int bindex, const string& xml_file, const string& root_dir,
                         vector<LabeledBBox<Dtype> >* labeled_boxes) {
  labeled_boxes->clear();
  ptree pt;
  read_xml(xml_file, pt);
  const int num = pt.get<int>("Annotations.NumPart");
  const int width = pt.get<int>("Annotations.ImageWidth");
  const int height = pt.get<int>("Annotations.ImageHeight");
  const std::string path = pt.get<string>("Annotations.ImagePath");
  if (num == 0) return;
  for (int i = 0; i < num; ++i) {
    LabeledBBox<Dtype> lbox;
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128], temp_cid[128];
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);
    sprintf(temp_cid,  "Annotations.Object_%d.cid",  i+1);
    const int cid =  pt.get<int>(temp_cid);
    const Dtype xmin =  pt.get<Dtype>(temp_xmin);
    const Dtype ymin =  pt.get<Dtype>(temp_ymin);
    const Dtype xmax =  pt.get<Dtype>(temp_xmax);
    const Dtype ymax =  pt.get<Dtype>(temp_ymax);
    if (xmin >= xmax || ymin >= ymax) {
      LOG(INFO) << "Found an illegal box (dsize < 0): " << xml_file;
      continue;
    }
    if (xmin < 0 || xmin >= width || xmax < 0 || xmax >= width) {
      LOG(INFO) << "Found an illegal box (exceed the boundary): " << xml_file;
      continue;
    }
    if (ymin < 0 || ymin >= height || ymax < 0 || ymax >= height) {
      LOG(INFO) << "Found an illegal box (exceed the boundary): " << xml_file;
    }
    lbox.bindex = bindex;
    lbox.cid = cid;
    lbox.pid = -1;  // unused.
    lbox.score = 1;
    lbox.bbox.x1_ = xmin / Dtype(width);
    lbox.bbox.y1_ = ymin / Dtype(height);
    lbox.bbox.x2_ = xmax / Dtype(width);
    lbox.bbox.y2_ = ymax / Dtype(height);
    labeled_boxes->push_back(lbox);
  }
}

INSTANTIATE_CLASS(UnifiedDataLayer);
REGISTER_LAYER_CLASS(UnifiedData);
}  // namespace caffe
