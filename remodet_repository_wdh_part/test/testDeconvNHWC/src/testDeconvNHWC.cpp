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
#include "caffe/filler.hpp"

#include "caffe/nhwc/deconv_nhwc_layer.hpp"

#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using namespace std;
using namespace caffe;

int main(int nargc, char** args) {
  // set caffe::GPU
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);

  int input_channels = 4;
  int output_channels = 4;
  int input_dim = 4;
  int kernel = 4;
  int stride = 4;
  int pad = 0;
  // define input & output blobs
  Blob<float>* input = new Blob<float>(1, input_dim, input_dim, input_channels);
  Blob<float>* output = new Blob<float>(1, input_dim, input_dim, output_channels);
  vector<Blob<float>*> bottom_vec;
  vector<Blob<float>*> top_vec;

  // set the bottom blob data
  FillerParameter filler_param;
  filler_param.set_std(0.5);
  GaussianFiller<float> filler(filler_param);
  filler.Fill(input);

  // set the bottom and top data-pointer
  bottom_vec.push_back(input);
  top_vec.push_back(output);

  // create reorg layer
  LayerParameter layer_param;
  layer_param.set_name("DeconvNHWC");
  layer_param.set_type("DeconvolutionNHWC");
  ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
  conv_param->set_kernel_h(kernel);
  conv_param->set_kernel_w(kernel);
  conv_param->set_stride_h(stride);
  conv_param->set_stride_w(stride);
  conv_param->set_pad_h(pad);
  conv_param->set_pad_w(pad);
  conv_param->set_group(1);
  conv_param->set_num_output(output_channels);
  conv_param->set_engine(caffe::ConvolutionParameter_Engine_CAFFE);
  FillerParameter* weight_filler = conv_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.01);
  FillerParameter* bias_filler = conv_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_std(0);

  boost::shared_ptr<Layer<float> > deconvNHWC_layer = LayerRegistry<float>::CreateLayer(layer_param);
  deconvNHWC_layer->SetUp(bottom_vec, top_vec);
  deconvNHWC_layer->Forward(bottom_vec, top_vec);

  GradientChecker<float> checker(1e-4, 1e-3);
  checker.CheckGradientExhaustive(deconvNHWC_layer.get(), bottom_vec, top_vec);

  LOG(INFO) << "Gradient Check Finished.";
  return 0;
}
