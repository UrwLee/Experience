#include <vector>

#include "caffe/layers/imagedata_histgram_layer.hpp"

namespace caffe {


template <typename Dtype>
void ImageDataHistgramLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batchsize = bottom[0]->num(); //bottom[0]:featuremap; bottom[1]:image
  vector<int> top_shape0(2, 1);
  top_shape0[0] = batchsize;
  top_shape0[1] = 256*3;
  top[0]->Reshape(top_shape0); 
}

template <typename Dtype>
void ImageDataHistgramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   int batchsize = bottom[0]->num(); //bottom[0]:featuremap; bottom[1]:image
  vector<int> top_shape0(2, 1);
  top_shape0[0] = batchsize;
  top_shape0[1] = 256*3;
  top[0]->Reshape(top_shape0); 
}

template <typename Dtype>
void ImageDataHistgramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  int batchsize = bottom[0]->num();
  int img_w = bottom[0]->width();
  int img_h = bottom[0]->height();
  const Dtype* imagedata = bottom[0]->cpu_data();
  vector<vector<Dtype> > hist_gram;
  hist_gram.resize(batchsize);
  for(int i=0;i<batchsize;i++){
    for(int j=0;j<256*3;j++){
      hist_gram[i].push_back(0);
    }
  }
  vector<float> vec_mean;
  vec_mean.push_back(104.0);
  vec_mean.push_back(117.0);
  vec_mean.push_back(123.0);

  int offset_c = img_h*img_w;
  int offset_b = 3*img_h*img_w;
  for(int ibatch=0;ibatch<batchsize;ibatch++){
    for(int ic=0;ic<3;ic++){
      for(int ih=0;ih<img_h;ih++){
        for(int iw=0;iw<img_w;iw++){
          float v = imagedata[ibatch*offset_b+ic*offset_c+ih*img_w+iw]+vec_mean[ic];
          int id = (int)v;
          hist_gram[ibatch][ic*256 + id]++;
        }
      }
    }
  }
  Dtype* top_data = top[0]->mutable_cpu_data();//
  int idx = 0;
  for(int ibatch=0;ibatch<batchsize;ibatch++){
    for(int iv=0;iv<256*3;iv++){
      top_data[idx++] = hist_gram[ibatch][iv]/512.0/288.0;
    }
  }  
}

INSTANTIATE_CLASS(ImageDataHistgramLayer);
REGISTER_LAYER_CLASS(ImageDataHistgram);

}  // namespace caffe
