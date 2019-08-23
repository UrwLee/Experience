#include <vector>
#include "caffe/mask/kps_gen_layer.hpp"

namespace caffe {

template <typename Dtype>
void KpsGenLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const KpsGenParameter& kps_gen_param = this->layer_param_.kps_gen_param();
  resized_height_ = kps_gen_param.resized_height();
  resized_width_ = kps_gen_param.resized_width();
  // default: false
  use_softmax_ = kps_gen_param.use_softmax();
}

template <typename Dtype>
void KpsGenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  // check for anchors
  if (bottom[0]->width() != 7 && bottom[0]->width() != 9) {
    LOG(INFO) << "ROI-Instance must has a width of 7 or 9.";
  }
  const int num_mat = bottom[0]->height();
  // bottom[1] -> 61
  CHECK_EQ(bottom[1]->width(), 61);
  if (use_softmax_) {
    vector<int> shape;
    shape.push_back(num_mat);
    shape.push_back(18);
    top[0]->Reshape(shape);
  } else {
    top[0]->Reshape(num_mat,18,resized_height_,resized_width_);
  }
  // each anchor will have 18 flags to indicate if the specified channel is active (control the grad-flow)
  top[1]->Reshape(1,1,num_mat,18);
}

template <typename Dtype>
void KpsGenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype* anchor_data = bottom[0]->cpu_data();
  const Dtype* kps_data = bottom[1]->cpu_data();
  // offs = 61
  const int rw = bottom[0]->width();
  const int offs = bottom[1]->width();
  const int num_mat = bottom[0]->height();
  const int num_gt = bottom[1]->height();
  if ((num_mat == 1) && (anchor_data[0] < 0)) {
    caffe_set<Dtype>(top[0]->count(), (Dtype)0, top[0]->mutable_cpu_data());
    caffe_set<Dtype>(top[1]->count(), (Dtype)0, top[1]->mutable_cpu_data());
  } else {
    for (int i = 0; i < num_mat; ++i) {
      const int bindex  = anchor_data[i * rw];
      const int cid     = anchor_data[i * rw + 1];
      const int pid     = anchor_data[i * rw + 2];
      const Dtype xmin  = anchor_data[i * rw + rw-4];
      const Dtype ymin  = anchor_data[i * rw + rw-3];
      const Dtype xmax  = anchor_data[i * rw + rw-2];
      const Dtype ymax  = anchor_data[i * rw + rw-1];
      // search this image within the mini-batch
      CHECK_EQ(cid, 0) << "Only cid == 0 (person) is support in current version.";
      int matched_idx = -1;
      for (int j = 0; j < num_gt; ++j) {
        const int gt_bindex = kps_data[offs * j];
        const int gt_cid    = kps_data[offs * j + 1];
        const int gt_pid    = kps_data[offs * j + 2];
        if ((gt_bindex == bindex) && (gt_cid == cid) && (gt_pid == pid)) {
          matched_idx = j;
          break;
        }
      }
      if (matched_idx < 0) {
        LOG(FATAL) << "Found No Matching-Kps for the proposals (Anchors).";
      }
      // output
      const int matched_is_diff  = kps_data[offs * matched_idx + 3];
      const int matched_iscrowd  = kps_data[offs * matched_idx + 4];
      const int matched_has_kps  = kps_data[offs * matched_idx + 5];
      // const int matched_num_kps  = kps_data[offs * matched_idx + 6];
      // is_diff or iscrowd or not has_mask
      int offs_item;
      int offs_chan;
      if (use_softmax_) {
        offs_item = 18;
        offs_chan = 1;
      } else {
        offs_item = 18 * resized_height_ * resized_width_;
        offs_chan = resized_height_ * resized_width_;
      }
      // default: All are zeros
      caffe_set<Dtype>(offs_item, (Dtype)0, top[0]->mutable_cpu_data() + i * offs_item);
      caffe_set<Dtype>(18, (Dtype)0, top[1]->mutable_cpu_data() + i * 18);
      if (matched_is_diff || matched_iscrowd || (matched_has_kps == 0)) {
        // switch to the next anchor
        continue;
      } else {
        Dtype* top_flags = top[1]->mutable_cpu_data() + i * 18;
        Dtype* top_maps = top[0]->mutable_cpu_data() + i * offs_item;
        // judge size
        if ((xmax - xmin) * (ymax - ymin) < 0.001) {
          continue;
        }
        // handle 18 maps
        for (int k = 0; k < 18; ++k) {
          // joint - k
          Dtype px = kps_data[offs * matched_idx + 7 + k * 3];
          Dtype py = kps_data[offs * matched_idx + 8 + k * 3];
          int pv   = kps_data[offs * matched_idx + 9 + k * 3];
          if (pv <= 1) {
            // modify
            px = (px - xmin) / (xmax - xmin);
            py = (py - ymin) / (ymax - ymin);
            if (px > 0. && px < 1. && py > 0. && py < 1.) {
              top_flags[k] = 1;
              int cx = round(px * resized_width_);
              int cy = round(py * resized_height_);
              int idx = cy * resized_width_ + cx;
              idx = std::min(idx,resized_height_ * resized_width_ - 1);
              if (use_softmax_) {
                top_maps[k] = idx;
              } else {
                top_maps[k * offs_chan + idx] = 1;
              }
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(KpsGenLayer);
REGISTER_LAYER_CLASS(KpsGen);

}
