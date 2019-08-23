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

using namespace std;
using namespace caffe;

int main(int nargc, char** args) {

  // set caffe::GPU
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // caffe::Caffe::set_mode(caffe::Caffe::CPU);

  // define input & output blobs
  Blob<float>* input = new Blob<float>(1, 32, 4, 4);
  Blob<float>* output = new Blob<float>(1, 32, 4, 4);
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

  // set the param of Align layer
  float dropout_ratio = 0.5;

  // create reorg layer
  LayerParameter layer_param;
  layer_param.set_name("test_Spatial_Dropout");
  layer_param.set_type("SpatialDropout");
  SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
  spatial_dropout_param->set_dropout_ratio(dropout_ratio);

  // RoiAlignLayer<float> roi_align_layer(layer_param);
  boost::shared_ptr<Layer<float> > spatial_dropout_layer = LayerRegistry<float>::CreateLayer(layer_param);
  // spatial_dropout_layer->phase_ == caffe::TRAIN;
  LOG(INFO) << "Step6...";
  spatial_dropout_layer->SetUp(bottom_vec, top_vec);
  //
  LOG(INFO) << "Start Running...";

  // Runing the layer
  // spatial_dropout_layer->Reshape(bottom_vec, top_vec);
  // spatial_dropout_layer->Forward(bottom_vec, top_vec);

  // RoiAlignLayer<float> spatial_dropout_layer(layer_param);
  /**
   * first one: stepsize
   * second one: threshold
   */
  GradientChecker<float> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(spatial_dropout_layer.get(), bottom_vec, top_vec, 0);
  // checker.CheckGradientExhaustive(&roi_align_layer, bottom_vec, top_vec);
  /**
   * [top_id, top_data_id, blob_id, feat_id] :
   * top_id -> top.size()[...]
   * top_data_id -> top[...].count()[...]
   * blob_id -> bottom.size()[...]
   * feat_id -> bottom[...].count()[...]
   * Note: the ROIData, with blob_id == 1 should be ignored.
   * So error with blob_id == 1 could be ignored cuz we do not backpropagate the gradient to roi input.
   */
  LOG(INFO) << "Gradient Check Finished.";
  return 0;
}
