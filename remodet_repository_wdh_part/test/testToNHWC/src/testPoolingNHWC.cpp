#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/roi_align_layer.hpp"
#include "caffe/filler.hpp"

using namespace std;
using namespace caffe;

void NCHWToNHWC(const float* input, float* output, const int channels, const int height, const int width) {
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int input_offset = (c * height + h) * width + w;
        const int output_offset = (h * width + w) * channels + c;
        output[output_offset] = input[input_offset];
      }
    }
  }
}

void equal_NHWC_NCHW(const float* input, const float* output, const int channels, const int height, const int width) {
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int input_offset = (c * height + h) * width + w;
        const int output_offset = (h * width + w) * channels + c;
        const float input_val = input[input_offset];
        const float output_val = output[output_offset];
        if (fabs(input_val - output_val) > 1e-3) {
          LOG(INFO) << "[ERROR] (c,h,w) = " << c << "," << h << "," << w << ", "
                    << "Input is : " << input_val << ", while Output is : " << output_val;
        } else {
          LOG(INFO) << "[   OK] (c,h,w) = " << c << "," << h << "," << w << ", "
                    << "Input is : " << input_val << ", and Output is : " << output_val;
        }
      }
    }
  }
}

int main(int nargc, char** args) {
  // network NCHW
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  boost::shared_ptr<caffe::Net<float> > net_NCHW_, net_NHWC_;
  const std::string network_proto = "/home/zhangming/work/repository/test/testToNHWC/models/net_NCHW.prototxt";
  const std::string caffe_model = "/home/zhangming/work/repository/test/testToNHWC/models/net_NCHW.caffemodel";
  const std::string network_proto_NHWC = "/home/zhangming/work/repository/test/testToNHWC/models/net_NHWC.prototxt";
  const std::string caffe_model_NHWC = "/home/zhangming/work/repository/test/testToNHWC/models/net_NHWC.caffemodel";
  net_NCHW_.reset(new caffe::Net<float>(network_proto, caffe::TEST));
  net_NHWC_.reset(new caffe::Net<float>(network_proto_NHWC, caffe::TEST));
  net_NCHW_->CopyTrainedLayersFrom(caffe_model);
  net_NHWC_->CopyTrainedLayersFrom(caffe_model_NHWC);

  // input for NCHW
  const int input_channels = 3;
  const int input_height = 224;
  const int input_width = 224;
  Blob<float>* input_blob = net_NCHW_->input_blobs()[0];
  input_blob->Reshape(1,input_channels,input_height,input_width);
  const int input_width_NCHW = input_blob->shape(3);
  const int input_height_NCHW = input_blob->shape(2);
  const int input_channels_NCHW = input_blob->shape(1);
  LOG(INFO) << "Input dim (NCHW): " << input_height_NCHW << " x " << input_width_NCHW << ", Input channels: " << input_channels_NCHW;
  // input for NHWC
  Blob<float>* input_blob_NHWC = net_NHWC_->input_blobs()[0];
  input_blob_NHWC->Reshape(1,input_height,input_width,input_channels);
  const int input_width_NHWC = input_blob_NHWC->shape(2);
  const int input_height_NHWC = input_blob_NHWC->shape(1);
  const int input_channels_NHWC = input_blob_NHWC->shape(3);
  LOG(INFO) << "Input dim (NHWC): " << input_height_NHWC << " x " << input_width_NHWC << ", Input channels: " << input_channels_NHWC;

  // input for NCHW
  LOG(INFO) << "Input data for NCHW.";
  FillerParameter filler_param;
  filler_param.set_std(0.5);
  GaussianFiller<float> filler(filler_param);
  filler.Fill(input_blob);
  LOG(INFO) << "Input data for NHWC.";
  NCHWToNHWC(input_blob->cpu_data(), input_blob_NHWC->mutable_cpu_data(), input_channels, input_height, input_width);

  // Forward
  net_NCHW_->Forward();
  net_NHWC_->Forward();

  // get the feature blobs
  const std::string feature_name = "pool5";
  Blob<float> output_NCHW, output_NHWC;
  const boost::shared_ptr<Blob<float> > feature = net_NCHW_->blob_by_name(feature_name.c_str());
  const Blob<float>* f_b = feature.get();
  output_NCHW.ReshapeLike(*f_b);
  output_NCHW.ShareData(*f_b);
  const boost::shared_ptr<Blob<float> > feature_NHWC = net_NHWC_->blob_by_name(feature_name.c_str());
  const Blob<float>* f_b_NHWC = feature_NHWC.get();
  output_NHWC.ReshapeLike(*f_b_NHWC);
  output_NHWC.ShareData(*f_b_NHWC);

  // check if it's equal in [NCHW] & [NHWC]
  equal_NHWC_NCHW(output_NCHW.cpu_data(), output_NHWC.cpu_data(), output_NCHW.shape(1), output_NCHW.shape(2), output_NCHW.shape(3));

  LOG(INFO) << "Finished.";
  return 0;
}
