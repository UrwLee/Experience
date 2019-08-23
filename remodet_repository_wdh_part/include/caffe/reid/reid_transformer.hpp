#ifndef CAFFE_REID_TRANSFORMER_HPP
#define CAFFE_REID_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * Re-Identification数据读入层的数据转换类。
 * 提供了对Re-ID任务的数据转换任务。
 * 禁止直接使用。
 * 请在熟练掌握源码基础上使用。
 */

template <typename Dtype>
struct Person {
  // id
  int id;
  // bounding_box - normalized
  Dtype xmin;
  Dtype ymin;
  Dtype width;
  Dtype height;
};

 // MetaData
 template <typename Dtype>
 struct MetaData {
   // image path
   string img_path;
   // size of image
   int img_width;
   int img_height;
   // num of people
   int nAppear;
   // persons
   vector<Person<Dtype> > persons;
 };

template <typename Dtype>
class ReidTransformer {
 public:
  explicit ReidTransformer(const ReidTransformationParameter& param, Phase phase);
  virtual ~ReidTransformer() {}

  void InitRand();
  void Transform(const string& xml_file,
                  const int batch_idx,
                  Dtype* transformed_data,
                  Dtype* transformed_label,
                  int* num);
  void getNumSamples(const string& xml_file, int* num);
  /**
   * Steps for Transform:
   * 1. rotate 90 degree -> if height > width
   * 2. resize to 512 x 288
   * 3. we do not random-crop & color-distortion & random-flip
   */
 protected:
  virtual int Rand(int n);
  void Transform(MetaData<Dtype>& meta,
                const int batch_idx,
                Dtype* transformed_data,
                Dtype* transformed_label);
  bool ReadMetaDataFromXml(const string& xml_file, const string& root_dir, MetaData<Dtype>& meta);
  void visualize(cv::Mat& img, MetaData<Dtype>& meta);
  // Tranformation parameters
  ReidTransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
};

}  // namespace caffe

#endif
