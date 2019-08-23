#include <string>
#include <vector>
#include <utility>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/visualize_boxpose_layer.hpp"

namespace caffe {

template <typename Dtype>
double VisualizeBoxposeLayer<Dtype>::get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time,NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const VisualizeBoxposeParameter &visualize_boxpose_param =
      this->layer_param_.visualize_boxpose_param();
  // draw type
  drawtype_ = visualize_boxpose_param.type();
  pose_threshold_ = visualize_boxpose_param.pose_threshold();
  write_frames_ = visualize_boxpose_param.write_frames();
  output_directory_ = visualize_boxpose_param.output_directory();
  visualize_ = visualize_boxpose_param.visualize();
  print_score_ = visualize_boxpose_param.print_score();
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // bottom[0]: image
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 3);
  // bottom[1]: heatmaps + vecmaps
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 52);
  // bottom[2]: proposals
  CHECK_EQ(bottom[2]->num(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->width(), 61);
  // not used
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // no cpu method.
  return;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(VisualizeBoxposeLayer, Forward);
#endif

INSTANTIATE_CLASS(VisualizeBoxposeLayer);
REGISTER_LAYER_CLASS(VisualizeBoxpose);

} // namespace caffe
