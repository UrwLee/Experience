#include <vector>
#include "caffe/mask/split_label_layer.hpp"

namespace caffe {

template <typename Dtype>
void SplitLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int offset = bottom[0]->width();
  // NOTE: dim should be 9 + (1/0)*56 + (1/0)*(1+HW)
  spatial_dim_ = this->layer_param_.split_label_param().spatial_dim();
  add_parts_ = this->layer_param_.split_label_param().add_parts();
  int num_top_blobs = 0;
  if (offset == 9) {
    add_kps_ = false;
    add_mask_ = false;
    num_top_blobs = 1 + add_parts_;
  } else if (offset == 65) {
    add_kps_ = true;
    add_mask_ = false;
    num_top_blobs = 2 + add_parts_;
  } else if (offset == (10 + spatial_dim_)) {
    add_kps_ = false;
    add_mask_ = true;
    num_top_blobs = 2 + add_parts_;
  } else if (offset == (66 + spatial_dim_)) {
    add_kps_ = true;
    add_mask_ = true;
    num_top_blobs = 3 + add_parts_;
  } else {
    LOG(FATAL) << "bottom width check failed.";
  }
  CHECK_EQ(num_top_blobs, top.size());  
  int top_id = 0;
  top[top_id]->Reshape(1,1,1,9);
  if (add_parts_) {
    top_id++;
    top[top_id]->Reshape(1,1,1,9);
  }
  if (add_kps_) {
    top_id++;
    top[top_id]->Reshape(1,1,1,61);
  }
  if (add_mask_) {
    top_id++;
    top[top_id]->Reshape(1,1,1,6+spatial_dim_);
  }
}

template <typename Dtype>
void SplitLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int offset = bottom[0]->width();
  const int num_gt = bottom[0]->height();
  int num_p = 0;
  int num_part = 0;
  for (int i = 0; i < num_gt; ++i) {
    if (bottom_data[offset * i + 1] == 0) {
      num_p++;
    } else if (bottom_data[offset * i + 1] > 0) {
      num_part++;
    } else;
  }
  // top[0]
  if (num_p == 0 && num_part == 0) {
    LOG(FATAL) << "Error: num_p == 0 && num_part == 0, it's not allowed.";
  }
  // PERSON & KPS & MASK
  if (num_p == 0) {
    int top_id = 0;
    top[top_id]->Reshape(1,1,1,9);
    caffe_set(top[top_id]->count(),Dtype(-1),top[top_id]->mutable_cpu_data());
    if (add_parts_) {
      top_id++;
    }
    if (add_kps_) {
      top_id++;
      top[top_id]->Reshape(1,1,1,61);
      caffe_set(top[top_id]->count(),Dtype(0),top[top_id]->mutable_cpu_data());
    }
    if (add_mask_) {
      top_id++;
      top[top_id]->Reshape(1,1,1,6+spatial_dim_);
      caffe_set(top[top_id]->count(),Dtype(0),top[top_id]->mutable_cpu_data());
    }
  } else {
    int top_id = 0;
    top[top_id]->Reshape(1,1,num_p,9);
    if (add_parts_) {
      top_id++;
    }
    if (add_kps_) {
      top_id++;
      top[top_id]->Reshape(1,1,num_p,61);
    }
    if (add_mask_) {
      top_id++;
      top[top_id]->Reshape(1,1,num_p,6+spatial_dim_);
    }
  }
  // Parts
  if (add_parts_) {
    if (num_part == 0) {
      top[1]->Reshape(1,1,1,9);
      caffe_set(top[1]->count(),Dtype(-1),top[1]->mutable_cpu_data());
    } else {
      top[1]->Reshape(1,1,num_part,9);
    }
  }
  // pointer
  Dtype* box_data = top[0]->mutable_cpu_data();
  int top_id = 0;
  int pidx = 0;
  Dtype* pbox_data = NULL;
  if (add_parts_) {
    top_id++;
    pbox_data = top[top_id]->mutable_cpu_data();
  }
  int part_idx = 0;
  Dtype* kps_data = NULL;
  if (add_kps_) {
    top_id++;
    kps_data = top[top_id]->mutable_cpu_data();
  }
  int kps_idx = 0;
  Dtype* mask_data = NULL;
  if (add_mask_) {
    top_id++;
    mask_data = top[top_id]->mutable_cpu_data();
  }
  int mask_idx = 0;
  // process
  for (int i = 0; i < num_gt; ++i) {
    int idx = 0;
    const Dtype* bottom_data_ptr = bottom_data + offset * i;
    int bindex  = bottom_data_ptr[idx++];
    if (bindex < 0) continue;
    int cid     = bottom_data_ptr[idx++];
    if (cid < 0) continue;
    int pid     = bottom_data_ptr[idx++];
    int is_diff = bottom_data_ptr[idx++];
    int iscrowd = bottom_data_ptr[idx++];
    Dtype xmin =  bottom_data_ptr[idx++];
    Dtype ymin =  bottom_data_ptr[idx++];
    Dtype xmax =  bottom_data_ptr[idx++];
    Dtype ymax =  bottom_data_ptr[idx++];
    // bbox
    if (cid == 0) {
      box_data[pidx++] = bindex;
      box_data[pidx++] = cid;
      box_data[pidx++] = pid;
      box_data[pidx++] = is_diff;
      box_data[pidx++] = iscrowd;
      box_data[pidx++] = xmin;
      box_data[pidx++] = ymin;
      box_data[pidx++] = xmax;
      box_data[pidx++] = ymax;
    } else if (cid > 0 && add_parts_) {
      pbox_data[part_idx++] = bindex;
      pbox_data[part_idx++] = cid;
      pbox_data[part_idx++] = pid;
      pbox_data[part_idx++] = is_diff;
      pbox_data[part_idx++] = iscrowd;
      pbox_data[part_idx++] = xmin;
      pbox_data[part_idx++] = ymin;
      pbox_data[part_idx++] = xmax;
      pbox_data[part_idx++] = ymax;
    }
    // kps
    if (add_kps_ && (cid == 0)) {
      int has_kps = bottom_data_ptr[idx++];
      int num_kps = bottom_data_ptr[idx++];
      kps_data[kps_idx++] = bindex;
      kps_data[kps_idx++] = cid;
      kps_data[kps_idx++] = pid;
      kps_data[kps_idx++] = is_diff;
      kps_data[kps_idx++] = iscrowd;
      kps_data[kps_idx++] = has_kps;
      kps_data[kps_idx++] = num_kps;
      for (int k = 0; k < 18; ++k) {
        kps_data[kps_idx++] = bottom_data_ptr[idx++];
        kps_data[kps_idx++] = bottom_data_ptr[idx++];
        kps_data[kps_idx++] = bottom_data_ptr[idx++];
      }
    }
    // mask
    if (add_mask_ && (cid == 0)) {
      int has_mask = bottom_data_ptr[idx++];
      mask_data[mask_idx++] = bindex;
      mask_data[mask_idx++] = cid;
      mask_data[mask_idx++] = pid;
      mask_data[mask_idx++] = is_diff;
      mask_data[mask_idx++] = iscrowd;
      mask_data[mask_idx++] = has_mask;
      for (int p = 0; p < spatial_dim_; ++p) {
        mask_data[mask_idx++] = bottom_data_ptr[idx++];
      }
    }
  }
 
}

INSTANTIATE_CLASS(SplitLabelLayer);
REGISTER_LAYER_CLASS(SplitLabel);

}
