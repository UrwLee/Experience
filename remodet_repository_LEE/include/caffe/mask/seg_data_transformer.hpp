#ifndef CAFFE_SEG_DATA_TRANSFORMER_HPP
#define CAFFE_SEG_DATA_TRANSFORMER_HPP

#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SegDataTransformer {
public:
  explicit SegDataTransformer(const SegDataTransformationParameter& param, Phase phase);
  virtual ~SegDataTransformer() {}
  void InitRand();
#ifdef USE_OPENCV
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob, bool preserve_pixel_vals) ;
#endif  // USE_OPENCV

protected:
#ifdef USE_OPENCV
  virtual int Rand(int n);
  cv::Mat ApplyAugmentation(const cv::Mat& in_img, std::vector<Dtype> prob);
  void randomDistortion(cv::Mat* image);
  void arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y);
  void adjust_gama(Dtype gama, cv::Mat &image);
  void gama_com(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image);
#endif  // USE_OPENCV

  SegDataTransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
