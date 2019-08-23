#include <vector>

#include "caffe/layers/trackerprior_layer.hpp"

namespace caffe {


template <typename Dtype>
void TrackerPriorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batchsize = bottom[0]->num() / 2; //bottom[0]:featuremap; bottom[1]:image
  int h = bottom[0]->height();
  int w = bottom[0]->width();
  step_ = this->layer_param_.trackerprior_param().step();
  extent_scale_ = this->layer_param_.trackerprior_param().extent_scale();
  CHECK_LE(extent_scale_,1.0);
  float edge = 1.0 - 0.5*(1 + extent_scale_);
  numprior_h_ = int(h *edge/step_);
  numprior_w_ = int(w *edge/step_);
  vector<int> top_shape0(2, 1);
  top_shape0[0] = batchsize*numprior_h_*numprior_w_*2;
  top_shape0[1] = 5;
  top[0]->Reshape(top_shape0); 
   // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  vector<int> top_shape1(3, 1);
  top_shape1[0] = 1;
  top_shape1[1] = 2;
  top_shape1[2] = numprior_h_*numprior_w_*4;
  top[1]->Reshape(top_shape1); 
}

template <typename Dtype>
void TrackerPriorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   int batchsize = bottom[0]->num() / 2; //bottom[0]:featuremap; bottom[1]:image
   vector<int> top_shape0(2, 1);
  top_shape0[0] = batchsize*numprior_h_*numprior_w_*2;
  top_shape0[1] = 5;
  top[0]->Reshape(top_shape0); 
   // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  vector<int> top_shape1(3, 1);
  top_shape1[0] = 1;
  top_shape1[1] = 2;
  top_shape1[2] = numprior_h_*numprior_w_*4;
  top[1]->Reshape(top_shape1); 
}

template <typename Dtype>
void TrackerPriorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  int img_w = bottom[1]->width();
  int img_h = bottom[1]->height();
  int feat_w = bottom[0]->width();
  int feat_h = bottom[0]->height();
  Dtype* top_data0 = top[0]->mutable_cpu_data();//
  int batchsize = bottom[0]->num() / 2;
  int idx = 0;
  float w_extened = 0.5*(1 + extent_scale_);
  float mid_start = (1 - w_extened)/2.0;
  float mid_end =  mid_start + w_extened;
  for(int ibatch=0;ibatch<batchsize;ibatch++){
    for(int ih=0;ih<numprior_h_;ih++){
      for(int iw=0;iw<numprior_w_;iw++){
        top_data0[idx++] = ibatch;
        top_data0[idx++] = (int)(mid_start*img_w);
        top_data0[idx++] = (int)(mid_start*img_h);
        top_data0[idx++] = (int)(mid_end*img_w);
        top_data0[idx++] = (int)(mid_end*img_h);
        top_data0[idx++] = ibatch+batchsize;
        top_data0[idx++] = (int)((float)iw*step_/(float)feat_w*img_w);
        top_data0[idx++] = (int)((float)ih*step_/(float)feat_h*img_h);
        top_data0[idx++] = (int)(((float)iw*step_/(float)feat_w + w_extened)*img_w);
        top_data0[idx++] = (int)(((float)ih*step_/(float)feat_h + w_extened)*img_h);
        //LOG(INFO)<<idx<<" "<<batchsize*numprior_h_*numprior_w_*2*5;
      }
    }
  }
  Dtype* top_data1 = top[1]->mutable_cpu_data();//
  idx = 0;
  float prior_edge = (w_extened - 0.5)/2.0;
  for(int ih=0;ih<numprior_h_;ih++){
    for(int iw=0;iw<numprior_w_;iw++){
      top_data1[idx++] = (float)iw*step_/(float)feat_w + prior_edge;
      top_data1[idx++] = (float)ih*step_/(float)feat_h + prior_edge;
      top_data1[idx++] = (float)iw*step_/(float)feat_w + 0.5;
      top_data1[idx++] = (float)ih*step_/(float)feat_h + 0.5;
    }
  }
  for(int ih=0;ih<numprior_h_;ih++){
    for(int iw=0;iw<numprior_w_;iw++){
      top_data1[idx++] = 0.1;
      top_data1[idx++] = 0.1;
      top_data1[idx++] = 0.2;
      top_data1[idx++] = 0.2;
    }
  }
}

INSTANTIATE_CLASS(TrackerPriorLayer);
REGISTER_LAYER_CLASS(TrackerPrior);

}  // namespace caffe
