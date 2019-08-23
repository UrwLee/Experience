#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/nhwc/pooling_nhwc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingNHWCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PoolingParameter& pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingNHWCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, height, width, channels)";
  channels_ = bottom[0]->shape(3);
  height_ = bottom[0]->shape(1);
  width_ = bottom[0]->shape(2);
  if (global_pooling_) {
    kernel_h_ = bottom[0]->shape(1);
    kernel_w_ = bottom[0]->shape(2);
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->shape(0), pooled_height_, pooled_width_, channels_);
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    max_idx_.Reshape(bottom[0]->shape(0), pooled_height_, pooled_width_, channels_);
  }
}

template <typename Dtype>
void PoolingNHWCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // we use Eigen::Matrix to compute Pooling
  typedef Eigen::Map<const Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> > ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> > EigenMatrixMap;
  // input and output data pointer
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int input_spatial_dim = bottom[0]->count() / channels_;
  const int output_spatial_dim = top[0]->count() / channels_;
  // Eigen Matrix for input & output
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX: {
      caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
      ConstEigenMatrixMap input_mat(bottom_data, channels_, input_spatial_dim);
      EigenMatrixMap output_mat(top_data, channels_, output_spatial_dim);
      int out_col = 0;
      for (int n = 0; n < bottom[0]->shape(0); ++n) {
        const int input_base = n * height_;
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int input_col = (input_base + h) * width_ + w;
                output_mat.col(out_col) =
                  output_mat.col(out_col).cwiseMax(input_mat.col(input_col));
              }
            }
            out_col++;
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE: {
    ConstEigenMatrixMap input_mat(bottom_data, channels_, input_spatial_dim);
    EigenMatrixMap output_mat(top_data, channels_, output_spatial_dim);
    output_mat.setZero();
    int out_col = 0;
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      const int input_base = n * height_;
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_ + pad_h_);
          int wend = min(wstart + kernel_w_, width_ + pad_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, height_);
          wend = min(wend, width_);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_col = (input_base + h) * width_ + w;
              output_mat.col(out_col) += input_mat.col(input_col);
            }
          }
          output_mat.col(out_col) /= pool_size;
          out_col++;
        }
      }
    }
  }
  break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(PoolingNHWCLayer);
#endif

INSTANTIATE_CLASS(PoolingNHWCLayer);
REGISTER_LAYER_CLASS(PoolingNHWC);
}
