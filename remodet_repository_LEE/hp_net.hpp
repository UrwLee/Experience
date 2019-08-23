#ifndef CAFFE_HP_NET_H_
#define CAFFE_HP_NET_H_

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/shared_ptr.hpp>

#include "caffe/tracker/bounding_box.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace caffe {

class HPNetWrapper {
public:
  HPNetWrapper(const std::string& proto, const std::string& model);

  int hpmode(const cv::Mat& image, const BoundingBox<float>& roi, float* score);

private:
  void getCropPatch(const cv::Mat& image, const BoundingBox<float>& roi, cv::Mat* patch);
  void load(const cv::Mat& image);

  boost::shared_ptr<caffe::Net<float> > net_;
};

}

#endif
