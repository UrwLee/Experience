#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "caffe/handkeypoint/bbox_data_transformer_handkeypoint.hpp"
#include "caffe/handkeypoint/bbox_data_handkeypoint_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace caffe {

using namespace boost::property_tree;

template <typename Dtype>
BBoxDataHandKeypointLayer<Dtype>::~BBoxDataHandKeypointLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void BBoxDataHandKeypointLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bbox_data_transformer_.reset(
     new BBoxDataHandKeypointTransformer<Dtype>(bbox_data_transform_param_, this->phase_));
  // 获取数据Layer参数
  // 仍然使用unified_data_param
  const UnifiedDataParameter& unified_data_param = this->layer_param_.unified_data_param();
  CHECK_EQ(unified_data_param.mean_value_size(), 3);
  for (int i = 0; i < 3; ++i) {
    mean_values_.push_back(unified_data_param.mean_value(i));
  }
  if(unified_data_param.has_xml_list()){
    string xml_list = unified_data_param.xml_list();
    string xml_root = unified_data_param.xml_root();
    LOG(INFO) << "Opening file " << xml_list;
    std::ifstream infile(xml_list.c_str());
    CHECK(infile.good()) << "Failed to open file "<< xml_list;
    std::string xmlname;
    while (infile >> xmlname) {
      lines_.push_back(make_pair(xml_root, xmlname));
    }
  } else{
    LOG(INFO)<<"size of unified_data_param.xml_list_multiple_size() "<<unified_data_param.xml_list_multiple_size();
    for (int i=0; i<unified_data_param.xml_list_multiple_size(); i++){
      string xml_list = unified_data_param.xml_list_multiple(i);
      string xml_root = unified_data_param.xml_root_multiple(i);
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
    skip=0;
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // data
  const int batch_size = unified_data_param.batch_size();
  Add_parts_ = unified_data_param.add_parts();
  const int height = bbox_data_transform_param_.resized_height();
  const int width = bbox_data_transform_param_.resized_width();
  top[0]->Reshape(batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  // label
  int stride=bbox_data_transform_param_.stride();
  
  int channelOffset = (height/stride) * (width/stride);
    // // 设置转换数据地址
  top[1]->Reshape(1,1,1,9+batch_size*(num_parts_+2*num_limbs_)*channelOffset);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(1,1,1,9+(num_parts_+2*num_limbs_)*channelOffset);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
}

template <typename Dtype>
void BBoxDataHandKeypointLayer<Dtype>::ShuffleLists() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void BBoxDataHandKeypointLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  int num_p = 0;
  // Dtype* top_labels = batch->label_.mutable_cpu_data();
     int stride=bbox_data_transform_param_.stride();
  
    int channelOffset = (height/stride) * (width/stride);
    // // 设置转换数据地址
    // batch->label_.Reshape(1,1,1,9+batch_size*(4*num_limbs_+2*num_parts_)*channelOffset);
      Blob<Dtype> blob_tmp;
    blob_tmp.Reshape(1,1,1,batch_size*(num_parts_+2*num_limbs_)*channelOffset);

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, lines_id_);
    string xml_root = lines_[lines_id_].first;
    string xml_path = xml_root + '/' + lines_[lines_id_].second;
    // LOG(INFO)<<xml_root<<" "<<xml_path;
    // Read Anno
    AnnoData<Dtype> anno;
    ReadAnnoDataFromXml(item_id, xml_path, xml_root, &anno);
    // 转换
    cv::Mat image;
    cv::Mat maskmiss;
    vector<BBoxData<Dtype> > bboxes;
    BoundingBox<Dtype> crop_bbox;
    bool doflip;
    // const int offset_label = batch->label_.offset(item_id);
    // this->transformed_label_.set_cpu_data(top_labels + offset_label);
    int old=(num_parts_+2*num_limbs_)*item_id;
    Dtype* transformed_heatmap = blob_tmp.mutable_cpu_data()+old*channelOffset;
    Dtype* transformed_vecmap =blob_tmp.mutable_cpu_data()+ (num_parts_+old)*channelOffset;

    bbox_data_transformer_->Transform(anno, &image, &bboxes, &crop_bbox, &doflip,transformed_heatmap,transformed_vecmap);
    num_p += bboxes.size();
    bboxes_all.push_back(bboxes);
    images_all.push_back(image);
    // 增加parts的标注
    // if (Add_parts_) {
    //   const int delim_pos = lines_[lines_id_].find_last_of("/");
    //   const string& xml_name = lines_[lines_id_].substr(delim_pos+1, lines_[lines_id_].length());
    //   const int point_pos = xml_name.find_last_of('.');
    //   const string& xname = xml_name.substr(0,point_pos);
    //   const string xml_parts_path = parts_xml_dir_ + '/' + xname + "_Parts.xml";
    //   vector<LabeledBBox<Dtype> > labeled_boxes;
    //   ReadPartBoxesFromXml(item_id, xml_parts_path, xml_root, &labeled_boxes);//hzw
    //   // copy bboxes to bboxes
    //   vector<BBoxDataHandPose<Dtype> > part_bboxes;
    //   unified_data_transformer_->ApplyCrop(crop_bbox, labeled_boxes, &part_bboxes);
    //   if (doflip) {
    //     unified_data_transformer_->FlipBoxes(&part_bboxes);
    //   }
    //   for (int i = 0; i < part_bboxes.size(); ++i) {
    //     bboxes.push_back(part_bboxes[i]);
    //     KpsData<Dtype> kps;
    //     MaskData<Dtype> mask;
    //     kpses.push_back(kps);
    //     masks.push_back(mask);
    //   }
    // }
    
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
  CHECK_LE(bboxes_all.size(), batch_size);
  // LOG(INFO) << bboxes_all.size() << " " << batch_size;

  // if (bbox_data_transform_param_.visualize()) {
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
  // int stride=bbox_data_transform_param_.stride();
  int num_gt = 0;
  for (int i = 0; i < bboxes_all.size(); ++i) {
    num_gt += bboxes_all[i].size();
  }
  // int channelOffset = (height/stride) * (width/stride);
    // 设置转换数据地址
  // int num_limbs_=17;
  // int num_parts_=18;
  CHECK_GT(num_gt, 0) << "Found No Ground-Truth.";
  batch->label_.Reshape(1,1,1,num_gt*9+batch_size*(num_parts_+2*num_limbs_)*channelOffset);
    Dtype* top_pose = batch->label_.mutable_cpu_data();
caffe_copy<Dtype>(batch_size*(num_parts_+2*num_limbs_)*channelOffset, blob_tmp.mutable_cpu_data(),top_pose);
//   for(int t=0;t<1916928;++t ){
//     if(batch->label_.mutable_cpu_data()[t]!=0){
// //      LOG(INFO)<<"yesyesye!!!"<<top_pose[t]<<"  "<<t;
//     }
//else LOG(INFO)<<t;
  // }
 // caffe_copy<Dtype>(batch_size*(4*num_limbs_+2*num_parts_)*channelOffset, blob_tmp.mutable_cpu_data(),top_pose);
  Dtype* top_label = batch->label_.mutable_cpu_data()+batch_size*(num_parts_+2*num_limbs_)*channelOffset;
  int count = 0;
  int idx = 0;
  for (int i = 0; i < bboxes_all.size(); ++i) {
    for (int j = 0; j < bboxes_all[i].size(); ++j) {
      BBoxData<Dtype>& box = bboxes_all[i][j];
      top_label[idx++] = box.bindex;
      top_label[idx++] = box.cid;
      top_label[idx++] = box.pid;
      // top_label[idx++]=caffe_rng_rand() % 10;
      top_label[idx++] = box.is_diff;
      top_label[idx++] = box.iscrowd;
      top_label[idx++] = box.bbox.x1_;
      top_label[idx++] = box.bbox.y1_;
      top_label[idx++] = box.bbox.x2_;
      top_label[idx++] = box.bbox.y2_;
      // LOG(INFO)<<"bbox_data_layer_top1:"<<box.bbox.x1_<<"|"<<box.bbox.y1_<<"|"<<box.bbox.x2_<<"|"<<box.bbox.y2_;
      count++;
      //top_label[1]=caffe_rng_rand() % 10;
    }
  }
  CHECK_EQ(count,num_gt) << "Size unmatched.";

}

template <typename Dtype>
void BBoxDataHandKeypointLayer<Dtype>::ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                                                  AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  // LOG(INFO)<<"img_path: "<<root_dir + '/' + pt.get<string>("Annotations.ImagePath");
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
      sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i+1);
    sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i+1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);

    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid);
    if (!Add_parts_) {
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
    //pose
    try {
      int num_keypoints = 20;
      pt.get<int>("Annotations.Object_1.kps.kp_1.x");
      ins.kps_included=true;
      //ins.mask_path = root_dir + '/' + pt.get<string>("Annotations.MaskMissPath");
      ins.joint.joints.resize(num_keypoints);
      ins.joint.isVisible.resize(num_keypoints);
      for(int j = 0; j < num_keypoints; ++j) {
          char temp_x[256], temp_y[256], temp_vis[256];
          sprintf(temp_x, "Annotations.Object_%d.kps.kp_%d.x", i+1,j+1);
          sprintf(temp_y, "Annotations.Object_%d.kps.kp_%d.y", i+1,j+1);
          sprintf(temp_vis, "Annotations.Object_%d.kps.kp_%d.vis", i+1,j+1);
          ins.joint.joints[j].x = pt.get<float>(temp_x);
          ins.joint.joints[j].y = pt.get<float>(temp_y);
          // ins.joint.joints[j] -= Point2f(1,1);
          int isVisible = pt.get<int>(temp_vis);
          ins.joint.isVisible[j] = (isVisible == 0) ? 0 : 1;
          if(ins.joint.joints[j].x <= 0 || ins.joint.joints[j].y <= 0 ||
             ins.joint.joints[j].x >= anno->img_width || ins.joint.joints[j].y >= anno->img_width) {
            ins.joint.isVisible[j] = 2;
          }
        }
        // LOG(INFO)<<ins.joint.joints[2].x<<"@"<<ins.joint.isVisible[2];
    } catch (const ptree_error &e) {
      ins.kps_included=false;
      //ins.mask_path = root_dir + '/' + pt.get<string>("Annotations.MaskMissPath");

    }
    anno->instances.push_back(ins);

    // LOG(INFO)<<ins.bbox.x1_<<"*"<<ins.bbox.y1_<<"*"<<ins.bbox.x2_<<"*"<<ins.bbox.y2_<<"bbox";
    // LOG(INFO)<<ins.kps_included<<"$"<<pt.get<int>("Annotations.Object_1.kps.kp_1.x")<<"!!";
//    LOG(INFO)<<pt.get<int>(temp_kps+"kp_1.x")<<"yes";

  }
}

INSTANTIATE_CLASS(BBoxDataHandKeypointLayer);
REGISTER_LAYER_CLASS(BBoxDataHandKeypoint);
}  // namespace caffe
