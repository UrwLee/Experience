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

  // define input & output blobs
  Blob<float>* input_0 = new Blob<float>(1, 2, 12, 12);
  Blob<float>* input_1 = new Blob<float>(1, 1, 2, 7);
  Blob<float>* output = new Blob<float>(2, 2, 6, 6);
  vector<Blob<float>*> bottom_vec;
  vector<Blob<float>*> top_vec;

  // set the bottom blob data
  FillerParameter filler_param;
  filler_param.set_std(0.5);
  GaussianFiller<float> filler(filler_param);
  filler.Fill(input_0);

  // set the ROI data
  float* roi_data = input_1->mutable_cpu_data();
  caffe_set(input_1->count(), (float)0, roi_data);
  roi_data[0] = 0;
  roi_data[3] = 0.1;
  roi_data[4] = 0.17;
  roi_data[5] = 0.4;
  roi_data[6] = 0.85;
  roi_data[7+0] = 0;
  roi_data[7+3] = 0.38;
  roi_data[7+4] = 0.02;
  roi_data[7+5] = 0.79;
  roi_data[7+6] = 0.9;

  // set the bottom and top data-pointer
  bottom_vec.push_back(input_0);
  bottom_vec.push_back(input_1);
  top_vec.push_back(output);

  // set the param of Align layer
  int roiResizedWidth = output->width();
  int roiResizedHeight = output->height();
  int inter_times = 1;

  // create reorg layer
  LayerParameter layer_param;
  layer_param.set_name("test_Align");
  layer_param.set_type("RoiAlign");
  RoiAlignParameter* roi_align_param = layer_param.mutable_roi_align_param();
  roi_align_param->set_roi_resized_width(roiResizedWidth);
  roi_align_param->set_roi_resized_height(roiResizedHeight);
  roi_align_param->set_inter_times(inter_times);

  // RoiAlignLayer<float> roi_align_layer(layer_param);
  boost::shared_ptr<Layer<float> > roi_align_layer = LayerRegistry<float>::CreateLayer(layer_param);
  // LOG(INFO) << "Step6...";
  roi_align_layer->SetUp(bottom_vec, top_vec);
  //
  // LOG(INFO) << "Start Running...";

  // Runing the layer
  // roi_align_layer->Reshape(bottom_vec, top_vec);
  // roi_align_layer->Forward(bottom_vec, top_vec);

  // RoiAlignLayer<float> roi_align_layer(layer_param);
  /**
   * first one: stepsize
   * second one: threshold
   */
  GradientChecker<float> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(roi_align_layer.get(), bottom_vec, top_vec);
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
