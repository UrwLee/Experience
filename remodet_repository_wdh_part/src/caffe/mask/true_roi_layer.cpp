#include <vector>
#include "caffe/mask/true_roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void TrueRoiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  type_ = this->layer_param_.true_roi_param().type();
}

template <typename Dtype>
void TrueRoiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1,1,1,7);
  top[1]->Reshape(1,1,1,1);
}

template <typename Dtype>
void TrueRoiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int offset = bottom[0]->width();
  const int num_gt = bottom[0]->height();
  vector<vector<Dtype> > rois;
  for (int i = 0; i < num_gt; ++i) {
    // header
    int bindex  = bottom_data[offset * i];
    int cid     = bottom_data[offset * i + 1];
    int pid     = bottom_data[offset * i + 2];
    // int is_diff = bottom_data[offset * i + 3];
    // int iscrowd = bottom_data[offset * i + 4];
    // bbox
    Dtype xmin =  bottom_data[offset * i + 5];
    Dtype ymin =  bottom_data[offset * i + 6];
    Dtype xmax =  bottom_data[offset * i + 7];
    Dtype ymax =  bottom_data[offset * i + 8];
    // kps-mask
    int has_kps = bottom_data[offset * i + 9];
    // int num_kps = bottom_data[offset * i + 10];
    int has_mask = bottom_data[offset * i + 65];
    // output
    if (type_ == "mask") {
      if (has_mask) {
        vector<Dtype> roi;
        roi.push_back(bindex);
        roi.push_back(cid);
        roi.push_back(pid);
        roi.push_back(xmin);
        roi.push_back(ymin);
        roi.push_back(xmax);
        roi.push_back(ymax);
        rois.push_back(roi);
      }
    } else if (type_ == "pose") {
      if (has_kps) {
        vector<Dtype> roi;
        roi.push_back(bindex);
        roi.push_back(cid);
        roi.push_back(pid);
        roi.push_back(xmin);
        roi.push_back(ymin);
        roi.push_back(xmax);
        roi.push_back(ymax);
        rois.push_back(roi);
      }
    } else {
      LOG(FATAL) << "Found Unknown type: " << type_;
    }
  }
  // output
  if (rois.size() == 0) {
    top[0]->Reshape(1,1,1,7);
    Dtype* top_data = top[0]->mutable_cpu_data();
    // we use the first one instead
    top_data[0] = bottom_data[0];
    top_data[1] = bottom_data[1];
    top_data[2] = bottom_data[2];
    top_data[3] = bottom_data[5];
    top_data[4] = bottom_data[6];
    top_data[5] = bottom_data[7];
    top_data[6] = bottom_data[8];
    top[1]->Reshape(1,1,1,1);
    top[1]->mutable_cpu_data()[0] = -1;
  } else {
    top[0]->Reshape(1,1,rois.size(),7);
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int n = 0; n < rois.size(); ++n) {
      top_data[n*7]   = rois[n][0];
      top_data[n*7+1] = rois[n][1];
      top_data[n*7+2] = rois[n][2];
      top_data[n*7+3] = rois[n][3];
      top_data[n*7+4] = rois[n][4];
      top_data[n*7+5] = rois[n][5];
      top_data[n*7+6] = rois[n][6];
    }
    top[1]->Reshape(1,1,1,rois.size());
    caffe_set(top[1]->count(), Dtype(1), top[1]->mutable_cpu_data());
  }
}

INSTANTIATE_CLASS(TrueRoiLayer);
REGISTER_LAYER_CLASS(TrueRoi);

}
