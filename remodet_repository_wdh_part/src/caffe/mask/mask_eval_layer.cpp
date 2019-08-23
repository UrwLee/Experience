#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/mask/mask_eval_layer.hpp"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

template <typename Dtype>
void MaskEvalLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void MaskEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0]: [N,1,H,W]
  CHECK_EQ(bottom[0]->channels(), 1);
  // bottom[1]: [N,1,H,W]
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  // bottom[2]: [1,1,N,7]
  CHECK_EQ(bottom[0]->num(), bottom[2]->height());
  CHECK_EQ(bottom[2]->width(), 7);
  // bottom[3]: [1,1,1,N]
  CHECK_EQ(bottom[3]->count(), bottom[0]->num());
  // Top[0]
  vector<int> shape(2,1);
  // [1,1,N,3]
  shape.push_back(bottom[0]->num());
  shape.push_back(3);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MaskEvalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* gt = bottom[1]->cpu_data();
  const Dtype* roi = bottom[2]->cpu_data();
  const Dtype* flags = bottom[3]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int offs = bottom[0]->height() * bottom[0]->width();
  for (int n = 0; n < num; ++n) {
    if (flags[n] < 0) {
      top_data[0] = -1;
      top_data[1] = -1;
      top_data[2] = -1;
      continue;
    }
    // get cid
    top_data[0] = roi[n * 7 + 1];
    // get size
    BoundingBox<Dtype> bbox;
    bbox.x1_ = roi[n * 7 + 3];
    bbox.y1_ = roi[n * 7 + 4];
    bbox.x2_ = roi[n * 7 + 5];
    bbox.y2_ = roi[n * 7 + 6];
    top_data[1] = bbox.compute_area();
    // get accuracy
    int tp = 0;
    int fp = 0;
    for (int i = 0; i < offs; ++i) {
      int pred_p = pred[i];
      int gt_p = gt[i];
      if (pred_p == gt_p) tp++;
      else fp++;
    }
    Dtype accuracy = Dtype(tp) / Dtype(tp+fp);
    top_data[2] = accuracy;
    // pointer to next one
    pred += offs;
    gt += offs;
    top_data += 3;
  }
}

INSTANTIATE_CLASS(MaskEvalLayer);
REGISTER_LAYER_CLASS(MaskEval);

}
