#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/mask/kps_eval_layer.hpp"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

template <typename Dtype>
void KpsEvalLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  conf_thre_ = this->layer_param_.kps_eval_param().conf_thre();
  // other params
  distance_thre_ = this->layer_param_.kps_eval_param().distance_thre();
}

template <typename Dtype>
void KpsEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0]: [1,N,18,3]
  // bottom[1]: [1,N,18,3]
  // bottom[2]: [1,1,N,7]
  // bottom[3]: [1,1,1,N]
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->height());
  CHECK_EQ(bottom[2]->width(), 7);
  CHECK_EQ(bottom[3]->count(), bottom[0]->channels());
  // Top[0] [1,1,N,3]
  vector<int> shape(2,1);
  shape.push_back(bottom[2]->height());
  // cid, size, 18
  shape.push_back(20);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void KpsEvalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* gt = bottom[1]->cpu_data();
  const Dtype* roi = bottom[2]->cpu_data();
  const Dtype* flags = bottom[3]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    if (flags[n] < 0) {
      caffe_set(20,Dtype(-1),top_data);
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
    for (int i = 0; i < 18; ++i) {
      Dtype x = pred[i*3];
      Dtype y = pred[i*3+1];
      Dtype v = pred[i*3+2];
      Dtype gx = gt[i*3];
      Dtype gy = gt[i*3+1];
      Dtype gv = gt[i*3+2];
      if (gv > 0) {
        if (v > conf_thre_) {
          Dtype distance = (x-gx)*(x-gx) + (y-gy)*(y-gy);
          if (distance < distance_thre_) {
            top_data[i+2] = 1;
          } else {
            top_data[i+2] = 0;
          }
        } else {
          top_data[i+2] = 0;
        }
      } else {
        if (v > conf_thre_) {
          top_data[i+2] = 0;
        } else {
          top_data[i+2] = 1;
        }
      }
    }
    // pointer to next one
    pred += 54;
    gt += 54;
    top_data += 20;
  }
}

INSTANTIATE_CLASS(KpsEvalLayer);
REGISTER_LAYER_CLASS(KpsEval);

}
