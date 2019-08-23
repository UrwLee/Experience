#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/reid/roi_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RoiDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), 7) << "bottom[0] must have dimension of 7 within each sample.";
  net_input_width_ = this->layer_param_.roi_data_param().net_input_width();
  net_input_height_ = this->layer_param_.roi_data_param().net_input_height();
}

template <typename Dtype>
void RoiDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(3), 7) << "bottom[0] must have dimension of 7 within each sample.";
  vector<int> shape(2);
  shape[0] = bottom[0]->shape(2);
  shape[1] = 5;
  // [N, 5]
  top[0]->Reshape(shape);
  shape.resize(1);
  shape[0] = bottom[0]->shape(2);
  // [N]
  top[1]->Reshape(shape);
}

template <typename Dtype>
void RoiDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int N = bottom[0]->shape(2);
  const Dtype* gt_data = bottom[0]->cpu_data();
  Dtype* roi_data = top[0]->mutable_cpu_data();
  Dtype* label_data = top[1]->mutable_cpu_data();
  /**
   * batch_index, cls_id, person_id, x1,y1,x2,y2
   */
  for (int i = 0; i < N; ++i) {
    // ROI_DATA {batch_idx, X1,Y1,X2,Y2 -> RealScale}
    roi_data[5*i]   = gt_data[7*i];    // batch_id
    roi_data[5*i+1] = gt_data[7*i+3] * net_input_width_;  // x1
    roi_data[5*i+2] = gt_data[7*i+4] * net_input_height_;  // y1
    roi_data[5*i+3] = gt_data[7*i+5] * net_input_width_;  // x2
    roi_data[5*i+4] = gt_data[7*i+6] * net_input_height_;  // y2
    // LABEL
    label_data[i] = gt_data[7*i+2];    // person_id
  }
}

#ifdef CPU_ONLY
STUB_GPU(RoiDataLayer);
#endif

INSTANTIATE_CLASS(RoiDataLayer);
REGISTER_LAYER_CLASS(RoiData);

}  // namespace caffe
