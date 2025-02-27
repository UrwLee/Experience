#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/layers/softmax_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FocalLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, float alpha_, float gama_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
      loss[index] = -alpha_ * powf(1 - pt, gama_) * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithFocusLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* counts = prob_.mutable_gpu_diff();
  FocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
      alpha_, gama_);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
      normalization_, outer_num_, inner_num_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
}

template <typename Dtype>
__global__ void FocalLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts, float alpha_, float gama_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype pt = bottom_diff[n * dim + label_value * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        if(c == label_value){
            bottom_diff[n * dim + c * spatial_dim + s] = alpha_ *
            powf(1 - pt, gama_) * (gama_ * pt * log(max(pt, Dtype(FLT_MIN))) + pt - 1);
        }
        else{
            Dtype pc = bottom_diff[n * dim + c * spatial_dim + s];
            bottom_diff[n * dim + c * spatial_dim + s] = alpha_ *
                (powf(1 - pt, gama_ - 1) * (-gama_ * log(max(pt, Dtype(FLT_MIN))) * pt * pc) +
                powf(1 - pt, gama_) * pc);
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithFocusLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    Dtype* counts = prob_.mutable_gpu_diff();
    FocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
        alpha_, gama_);

    Dtype valid_count = -1;
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, valid_count);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithFocusLossLayer);

}  // namespace caffe
