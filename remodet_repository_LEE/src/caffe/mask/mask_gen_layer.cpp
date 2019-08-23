#include <vector>
#include "caffe/mask/mask_gen_layer.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void MaskGenLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const MaskGenParameter& mask_gen_param = this->layer_param_.mask_gen_param();
  height_ = mask_gen_param.height();
  width_ = mask_gen_param.width();
  resized_height_ = mask_gen_param.resized_height();
  resized_width_ = mask_gen_param.resized_width();
}

template <typename Dtype>
void MaskGenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  // check for anchors
  if (bottom[0]->width() != 7 && bottom[0]->width() != 9) {
    LOG(INFO) << "ROI-Instance must has a width of 7 or 9.";
  }
  const int num_mat = bottom[0]->height();
  // bottom[1] -> 6 + map(h,w)
  CHECK_EQ(bottom[1]->width(), 6+height_*width_);
  // each anchor will have a mask-label with size -> (resized_height/width)
  top[0]->Reshape(num_mat,1,resized_height_,resized_width_);
  // each anchor will have a active-flag (1/0) to indicate if the mask is real
  top[1]->Reshape(1,1,num_mat,1);
}


template <typename Dtype>
void MaskGenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype* anchor_data = bottom[0]->cpu_data();
  const Dtype* mask_data = bottom[1]->cpu_data();
  // offs = 6 + h*w
  const int rw = bottom[0]->width();
  const int offs = bottom[1]->width();
  const int num_mat = bottom[0]->height();
  const int num_gt = bottom[1]->height();
  if ((num_mat == 1) && (anchor_data[0] < 0)) {
    // no anchors -- we use zero masks & false-active flags
    top[1]->mutable_cpu_data()[0] = 0;
    caffe_set<Dtype>(top[0]->count(), (Dtype)0, top[0]->mutable_cpu_data());
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
        const int gt_bindex = mask_data[offs * j];
        const int gt_cid = mask_data[offs * j + 1];
        const int gt_pid = mask_data[offs * j + 2];
        if ((gt_bindex == bindex) && (gt_cid == cid) && (gt_pid == pid)) {
          matched_idx = j;
          break;
        }
      }
      if (matched_idx < 0) {
        LOG(FATAL) << "Found No Matching-Masks for the proposals (Anchors).";
      }
      // output
      const int matched_is_diff  = mask_data[offs * matched_idx + 3];
      const int matched_iscrowd  = mask_data[offs * matched_idx + 4];
      const int matched_has_mask = mask_data[offs * matched_idx + 5];
      // is_diff or iscrowd or not has_mask
      if (matched_is_diff || matched_iscrowd || (matched_has_mask == 0)) {
        top[1]->mutable_cpu_data()[i] = 0;
        caffe_set<Dtype>(resized_height_*resized_width_, (Dtype)0, top[0]->mutable_cpu_data()+i*resized_height_*resized_width_);
      } else {
        // copy the mask to cv::Mat
        cv::Mat mask = cv::Mat::zeros(height_, width_, CV_8UC1);
        for (int y = 0; y < height_; ++y) {
          uchar* ptr = mask.ptr<uchar>(y);
          int index = 0;
          for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            ptr[index++] = static_cast<uchar>(mask_data[offs * matched_idx + 6 + idx]);
          }
        }
        // get roi
        int w_off_int = (int)(xmin * width_);
        int h_off_int = (int)(ymin * height_);
        int roi_w_int = (int)((xmax-xmin) * width_);
        int roi_h_int = (int)((ymax-ymin) * height_);
        roi_w_int = std::min(std::max(roi_w_int,1),width_-w_off_int);
        roi_h_int = std::min(std::max(roi_h_int,1),height_-h_off_int);
        cv::Rect roi(w_off_int, h_off_int, roi_w_int, roi_h_int);
        cv::Mat roi_mask = mask(roi);
        // resized
        cv::Mat roi_resized_mask;
        cv::resize(roi_mask, roi_resized_mask, cv::Size(resized_width_, resized_height_), cv::INTER_LINEAR);
        // copy to top_mask_label
        Dtype* top_mask_label = top[0]->mutable_cpu_data() + i*resized_height_*resized_width_;
        for (int y = 0; y < resized_height_; ++y) {
          for (int x = 0; x < resized_width_; ++x) {
            int val = roi_resized_mask.at<uchar>(y,x);
            top_mask_label[y * resized_width_ + x] = val > 127 ? 1 : 0;
          }
        }
        top[1]->mutable_cpu_data()[i] = 1;
      }
    }
  }
}

INSTANTIATE_CLASS(MaskGenLayer);
REGISTER_LAYER_CLASS(MaskGen);

}
