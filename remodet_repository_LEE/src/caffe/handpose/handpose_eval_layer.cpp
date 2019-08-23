#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/handpose/handpose_eval_layer.hpp"

namespace caffe {

template <typename Dtype>
void HandPoseEvalLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void HandPoseEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  int batchsize = bottom[0]->shape(0);
  std::vector<int> top_shape;
  //{N, pred_label/gt_label}
  top_shape.push_back(batchsize);
  top_shape.push_back(2);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void HandPoseEvalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* logits_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* out_data = top[0]->mutable_cpu_data();
  int batchsize = bottom[0]->shape(0);
  for(int ib = 0; ib < batchsize;ib++){
    const Dtype* logits_i = logits_data + ib * 10;
   /* LOG(INFO)<<logits_i[0]<<" "
            <<logits_i[1]<<" "
           <<logits_i[2]<<" "
		<<logits_i[3]<<" "
		<<logits_i[4]<<" "
		<<logits_i[5]<<" "
		<<logits_i[6]<<" "
		<<logits_i[7]<<" "
		<<logits_i[8]<<" "
		<<logits_i[9]<<" ";*/
    int max_id = -1;
    float max_value = -10;
    for (int ic = 0; ic < 10; ic++){
      if(logits_i[ic]>max_value){
        max_value = logits_i[ic];
        max_id = ic;
      }
    }
    out_data[2*ib] = gt_data[ib];
    out_data[2*ib + 1] = max_id;
    LOG(INFO)<<(int)(gt_data[ib])<<" "<<max_id<<" "<<max_value;
  }
}

INSTANTIATE_CLASS(HandPoseEvalLayer);
REGISTER_LAYER_CLASS(HandPoseEval);

}  // namespace caffe
