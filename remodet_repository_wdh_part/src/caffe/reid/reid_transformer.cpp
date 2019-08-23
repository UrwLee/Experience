#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "caffe/reid/reid_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {
using namespace boost::property_tree;
// 读入所有标注文件和数据
template<typename Dtype>
bool ReidTransformer<Dtype>::ReadMetaDataFromXml(const string& xml_file, const string& root_dir, MetaData<Dtype>& meta) {
  ptree pt;
  read_xml(xml_file, pt);
  // 读取路径
  string temp_img = pt.get<string>("Annotations.ImagePath");
  meta.img_path = root_dir + temp_img;
  // 读取Metadata
  cv::Mat img = cv::imread(root_dir + temp_img, CV_LOAD_IMAGE_COLOR);
  meta.img_width = img.cols;
  meta.img_height = img.rows;
  meta.nAppear = pt.get<int>("Annotations.nAppear");
  meta.persons.resize(meta.nAppear);
  for(int i = 0; i < meta.nAppear; ++i) {
    char box_xmin[256];
    char box_ymin[256];
    char box_width[256];
    char box_height[256];
    char box_id[256];
    sprintf(box_xmin, "Annotations.Bbox.box_%d.xmin", i+1);
    sprintf(box_ymin, "Annotations.Bbox.box_%d.ymin", i+1);
    sprintf(box_width, "Annotations.Bbox.box_%d.width", i+1);
    sprintf(box_height, "Annotations.Bbox.box_%d.height", i+1);
    sprintf(box_id, "Annotations.Bbox.box_%d.id", i+1);
    // get normalized value
    meta.persons[i].xmin = (Dtype)pt.get<int>(box_xmin) / (Dtype)meta.img_width;
    meta.persons[i].ymin = (Dtype)pt.get<int>(box_ymin) / (Dtype)meta.img_height;
    meta.persons[i].width = (Dtype)pt.get<int>(box_width) / (Dtype)meta.img_width;
    meta.persons[i].height = (Dtype)pt.get<int>(box_height) / (Dtype)meta.img_height;
    meta.persons[i].id = pt.get<int>(box_id);
  }
  return true;
}

template<typename Dtype>
ReidTransformer<Dtype>::ReidTransformer(const ReidTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {}

template<typename Dtype>
void ReidTransformer<Dtype>::getNumSamples(const string& xml_file, int* num) {
  MetaData<Dtype> meta;
  // 读入参数信息
  if (!ReadMetaDataFromXml(xml_file, param_.root_dir(), meta)) {
    LOG(FATAL) << "Error found in reading from: " << xml_file;
  }
  *num = meta.nAppear;
}

// 转换顶层API
template<typename Dtype>
void ReidTransformer<Dtype>::Transform(const string& xml_file,
                                const int batch_idx,
                                Dtype* transformed_data,
                                Dtype* transformed_label,
                                int* num) {
  MetaData<Dtype> meta;
  // 读入参数信息
  if (!ReadMetaDataFromXml(xml_file, param_.root_dir(), meta)) {
    LOG(FATAL) << "Error found in reading from: " << xml_file;
  }
  *num = meta.nAppear;
  Transform(meta,batch_idx,transformed_data,transformed_label);
}

// 转换调用
template<typename Dtype>
void ReidTransformer<Dtype>::Transform(MetaData<Dtype>& meta,
                            const int batch_idx,
                            Dtype* transformed_data,
                            Dtype* transformed_label) {
  int resized_width = param_.resized_width();
  int resized_height = param_.resized_height();
  cv::Mat image = cv::imread(meta.img_path, CV_LOAD_IMAGE_COLOR);

  cv::Mat tp;
  cv::Mat flip_90;
  cv::Mat aug;
  //Start transforming
  // rotate 90
  // if (meta.img_height > meta.img_width) {
  //   // flip 90
  //   transpose(image, tp);
  //   flip(tp,flip_90,1);
  //   resize(flip_90,aug,cv::Size(resized_width,resized_height),INTER_CUBIC);
  //   // update meta
  //   // update all boxes
  //   for (int i = 0; i < meta.nAppear; ++i) {
  //     Dtype temp = meta.persons[i].xmin;
  //     meta.persons[i].xmin = (Dtype)1.0 - meta.persons[i].ymin - meta.persons[i].height;
  //     meta.persons[i].ymin = temp;
  //     temp = meta.persons[i].height;
  //     meta.persons[i].height = meta.persons[i].width;
  //     meta.persons[i].width = temp;
  //   }
  //   // update image size
  //   int temp = meta.img_height;
  //   meta.img_height = meta.img_width;
  //   meta.img_width = temp;
  // } else {
  //   resize(image,aug,cv::Size(resized_width,resized_height),INTER_CUBIC);
  // }
  //
  resize(image,aug,cv::Size(resized_width,resized_height),INTER_CUBIC);
  // visualize
  if (param_.visual()) {
    visualize(aug,meta);
  }

  int offset = aug.rows * aug.cols;
  // data
  for (int i = 0; i < aug.rows; ++i) {
    for (int j = 0; j < aug.cols; ++j) {
      Vec3b& rgb = aug.at<Vec3b>(i, j);
      if (param_.normalize()) {
        transformed_data[i * aug.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[offset + i * aug.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2 * offset + i * aug.cols + j] = (rgb[2] - 128)/256.0;
      } else {
        // we use means to substract
        CHECK_EQ(param_.mean_value_size(), 3) << "Must provide mean_values, and mean_value should have length of 3.";
        transformed_data[i * aug.cols + j] = rgb[0] - param_.mean_value(0);
        transformed_data[offset + i * aug.cols + j] = rgb[1] - param_.mean_value(1);
        transformed_data[2 * offset + i * aug.cols + j] = rgb[2] - param_.mean_value(2);
      }
    }
  }
  // label [batch_idx, cls_id, id, xmin, ymin, xmax, ymax]
  for (int i = 0; i < meta.nAppear; ++i) {
    transformed_label[7*i] = batch_idx;
    transformed_label[7*i+1] = 1;
    transformed_label[7*i+2] = meta.persons[i].id;
    Dtype xmin = meta.persons[i].xmin;
    Dtype ymin = meta.persons[i].ymin;
    Dtype xmax = xmin + meta.persons[i].width;
    Dtype ymax = ymin + meta.persons[i].height;
    xmin = std::min(std::max(xmin,(Dtype)0),(Dtype)1);
    ymin = std::min(std::max(ymin,(Dtype)0),(Dtype)1);
    xmax = std::min(std::max(xmax,(Dtype)0),(Dtype)1);
    ymax = std::min(std::max(ymax,(Dtype)0),(Dtype)1);
    transformed_label[7*i+3] = xmin;
    transformed_label[7*i+4] = ymin;
    transformed_label[7*i+5] = xmax;
    transformed_label[7*i+6] = ymax;
  }
}

template<typename Dtype>
void ReidTransformer<Dtype>::visualize(cv::Mat& img, MetaData<Dtype>& meta) {
  cv::Mat img_vis = img.clone();
  static int counter = 0;
  for (int i = 0; i < meta.nAppear; ++i) {
    const Person<Dtype>& person = meta.persons[i];
    int id = person.id;
    int xmin = (int)(person.xmin * img_vis.cols);
    int ymin = (int)(person.ymin * img_vis.rows);
    int xmax = (int)((person.xmin + person.width) * img_vis.cols);
    int ymax = (int)((person.ymin + person.height) * img_vis.rows);
    xmin = std::max(std::min(xmin,img_vis.cols-1),0);
    xmax = std::max(std::min(xmax,img_vis.cols-1),0);
    ymin = std::max(std::min(ymin,img_vis.rows-1),0);
    ymax = std::max(std::min(ymax,img_vis.rows-1),0);
    cv::Point top_left_pt(xmin,ymin);
    cv::Point bottom_right_pt(xmax,ymax);
    cv::rectangle(img_vis, top_left_pt, bottom_right_pt, cv::Scalar(0,255,0), 3);
    cv::Point bottom_left_pt1(xmin+5,ymax-5);
    cv::Point bottom_left_pt2(xmin+3,ymax-3);
    char buffer[50];
    snprintf(buffer, sizeof(buffer), "%d", id);
    cv::putText(img_vis, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(img_vis, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
  }
  char imagename [256];
  sprintf(imagename, "%s%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

template <typename Dtype>
void ReidTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() || (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int ReidTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(ReidTransformer);

}  // namespace caffe
