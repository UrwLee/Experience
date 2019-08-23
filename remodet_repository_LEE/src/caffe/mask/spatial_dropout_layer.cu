#include "caffe/mask/spatial_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// template <typename Dtype>
// __global__ void caffe_gpu_axinpbxout(int n, int offset, Dtype a, const Dtype* in, Dtype b, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = a * in[offset + index] + b * out[offset + index];
//   }
// }

// template <typename Dtype>
// __global__ void caffe_gpu_axinpout(int n, int offset, Dtype a, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = a * in[offset + index] + out[offset + index];
//   }
// }

// template <typename Dtype>
// __global__ void caffe_gpu_axoutpin(int n, int offset, Dtype a, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = a * out[offset + index] + in[offset + index];
//   }
// }

// template <typename Dtype>
// __global__ void caffe_gpu_axin(int n, int offset, Dtype a, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = a * in[offset + index];
//   }
// }

template <typename Dtype>
__global__ void caffe_gpu_a(int n, int offset, Dtype a, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[offset + index] = a;
  }
}

// template <typename Dtype>
// __global__ void caffe_gpu_ineout(int n, int offset, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = in[offset + index];
//   }
// }

// template <typename Dtype>
// __global__ void SpatialDropoutForward(const int n, const int img_size, Dtype scale, 
//   unsigned int* mask, const Dtype* bottom_data, Dtype* top_data) {
//   CUDA_KERNEL_LOOP(index, n) {
    
//   }
// }

// template <typename Dtype>
// __global__ void caffe_gpu_ineout(int n, int offset, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[offset + index] = in[offset + index];
//   }
// }

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int img_size = bottom[0]->height() * bottom[0]->width();
  if (this->phase_ == TRAIN) {
    caffe_rng_bernoulli(num * channel, 1. - threshold_, mask);
    for(int n = 0; n < num; ++n) {
      for(int c = 0; c < channel; ++c) {
        int index = (n * channel + c);
        int offset =  index * img_size;
        if (mask[index] == 1) {
          caffe_gpu_axpby(img_size, scale_, bottom_data + offset, (Dtype)0, top_data + offset);
        }else{
          caffe_gpu_a<Dtype><<<CAFFE_GET_BLOCKS(img_size), CAFFE_CUDA_NUM_THREADS>>>(
            img_size, offset, (Dtype)0, top_data);
        }
      }
    }
  } else {
    caffe_gpu_memcpy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int num = bottom[0]->num();
    const int channel = bottom[0]->channels();
    const int img_size = bottom[0]->height() * bottom[0]->width();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      for(int n = 0; n < num; ++n) {
        for(int c = 0; c < channel; ++c) {
          int index = (n * channel + c);
          int offset =  index* img_size;
          if (mask[index] == 1) {
            caffe_gpu_axpby(img_size, scale_, top_diff + offset, (Dtype)0, bottom_diff + offset);
          } else {
            caffe_gpu_a<Dtype><<<CAFFE_GET_BLOCKS(img_size), CAFFE_CUDA_NUM_THREADS>>>(
            img_size, offset, (Dtype)0, bottom_diff);
          }
        }
      }
    } else {
      caffe_gpu_memcpy(bottom[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialDropoutLayer);

}  // namespace caffe
