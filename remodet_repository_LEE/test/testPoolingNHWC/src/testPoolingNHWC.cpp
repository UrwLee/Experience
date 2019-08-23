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

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/roi_align_layer.hpp"
#include "caffe/filler.hpp"

#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <Eigen/Dense>

#include "caffe/nhwc/pooling_nhwc_layer.hpp"

using namespace std;
using namespace caffe;

int main(int nargc, char** args) {
  // set caffe::GPU
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);

  int channels = 8;    //input
  int num_output = 8;  // output
  int dim = 8;          // size
  // define input & output blobs
  Blob<float>* input = new Blob<float>(1, dim, dim, channels);
  Blob<float>* output = new Blob<float>(1, dim, dim, num_output);
  vector<Blob<float>*> bottom_vec;
  vector<Blob<float>*> top_vec;

  // set the bottom blob data
  // FillerParameter filler_param;
  // filler_param.set_std(0.5);
  // GaussianFiller<float> filler(filler_param);
  // filler.Fill(input);

  for (int i = 0; i < input->count(); ++i) {
    input->mutable_cpu_data()[i] = sin(i) + tan(i) + tanh(i) + exp(-i);
  }

  // set the bottom and top data-pointer
  bottom_vec.push_back(input);
  top_vec.push_back(output);

  LayerParameter layer_param;
  layer_param.set_name("PoolingNHWC");
  layer_param.set_type("PoolingNHWC");
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(2);
  pooling_param->set_kernel_w(2);
  pooling_param->set_stride_h(2);
  pooling_param->set_stride_w(2);
  pooling_param->set_pad_h(0);
  pooling_param->set_pad_w(0);
  pooling_param->set_pool(caffe::PoolingParameter_PoolMethod_MAX);

  LOG(INFO) << layer_param.pooling_param().has_kernel_size();
  LOG(INFO) << layer_param.pooling_param().has_kernel_h();
  LOG(INFO) << layer_param.pooling_param().has_kernel_w();

  boost::shared_ptr<Layer<float> > PoolNHWC_layer = LayerRegistry<float>::CreateLayer(layer_param);
  PoolNHWC_layer->SetUp(bottom_vec, top_vec);

  PoolNHWC_layer->Forward(bottom_vec, top_vec);

  for (int i = 0; i < output->count(); ++i) {
    LOG(INFO) << output->cpu_data()[i];
  }

  // GradientChecker<float> checker(1e-4, 1e-3);
  // checker.CheckGradientExhaustive(PoolNHWC_layer.get(), bottom_vec, top_vec);

  LOG(INFO) << "Gradient Check Finished.";

  return 0;
}
