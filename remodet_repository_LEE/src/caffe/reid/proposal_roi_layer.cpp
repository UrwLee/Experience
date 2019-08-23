#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/reid/proposal_roi_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ProposalRoiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /**
   * NOTE: it's shared with roi_data_layer. use the param of roi_data_layer instead.
   */
  net_input_width_ = this->layer_param_.roi_data_param().net_input_width();
  net_input_height_ = this->layer_param_.roi_data_param().net_input_height();
}

template <typename Dtype>
void ProposalRoiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(3), 61);
  vector<int> shape(4,1);
  shape[1] = 5;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ProposalRoiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int N = bottom[0]->shape(2);
  const Dtype* proposal = bottom[0]->cpu_data();

  // default: use [0.25,0.25,0.75,0.75]
  if ((N == 1) && ((int)proposal[60] < 0)) {
    top[0]->Reshape(1,5,1,1);
    Dtype* top_data = top[0]->mutable_cpu_data();
    top_data[0] = 0;
    top_data[1] = (Dtype)0.25 * net_input_width_;
    top_data[2] = (Dtype)0.25 * net_input_height_;
    top_data[3] = (Dtype)0.75 * net_input_width_;
    top_data[4] = (Dtype)0.75 * net_input_height_;
    return;
  }
  // output
  top[0]->Reshape(N,5,1,1);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < N; ++i) {
    top_data[5*i] = 0;
    top_data[5*i+1] = proposal[61*i] * net_input_width_;
    top_data[5*i+2] = proposal[61*i+1] * net_input_height_;
    top_data[5*i+3] = proposal[61*i+2] * net_input_width_;
    top_data[5*i+4] = proposal[61*i+3] * net_input_height_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ProposalRoiLayer);
#endif

INSTANTIATE_CLASS(ProposalRoiLayer);
REGISTER_LAYER_CLASS(ProposalRoi);

}  // namespace caffe
