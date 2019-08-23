#ifndef CAFFE_SMILE_NET_H_
#define CAFFE_SMILE_NET_H_

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/shared_ptr.hpp>

#include "caffe/tracker/bounding_box.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace caffe {

class SmileNetWrapper {
public:
  SmileNetWrapper(const std::string& proto, const std::string& model);

  bool is_smile(const cv::Mat& image, const BoundingBox<float>& roi, float* score);
  void is_smile(const cv::Mat& image, const vector<BoundingBox<float> >& rois, vector<bool>* smile, vector<float>* scores);

private:
  void getCropPatch(const cv::Mat& image, const BoundingBox<float>& roi, cv::Mat* patch);
  void load(const cv::Mat& image);
  void load(const vector<cv::Mat>& images);

  boost::shared_ptr<caffe::Net<float> > net_;
};

}

#endif
