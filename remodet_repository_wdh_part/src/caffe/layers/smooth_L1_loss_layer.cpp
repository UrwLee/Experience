// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //  调用子类的设置函数
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // 输入为3,则True,否则为False
  has_weights_ = (bottom.size() == 3);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //  调用子类的Reshape方法
  LossLayer<Dtype>::Reshape(bottom, top);
  // 检查输入0和输入1的维度
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  // 如果有第三个输入,继续检查
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  // 梯度尺寸
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  //  误差尺寸
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // 计算梯度: 输入0 - 输入 1
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // 如果有第三个输入,则输入的是权重Blob (Win)
  // 需要再做1次乘法
  if (has_weights_) {
    caffe_mul(
        count,
        bottom[2]->cpu_data(),
        diff_.cpu_data(),
        diff_.mutable_cpu_data());  // d := w * (b0 - b1)
  }
  // 获取diff的数据指针
  const Dtype* diff_data = diff_.cpu_data();
  // 获取err的数据指针
  Dtype* error_data = errors_.mutable_cpu_data();
  // 误差err是diff的abs值
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    Dtype abs_val = fabs(val);
    /**
     * 如果误差在1内,使用1/2* square代替err
     */
    if (abs_val < 1.) {
      error_data[i] = 0.5 * val * val;
    } else {
      /**
       * 如果大于1, 则使用-0.5代替
       */
      error_data[i] = abs_val - 0.5;
    }
  }
  /**
   * 计算输出平均损失
   */
  top[0]->mutable_cpu_data()[0] =
      caffe_cpu_asum(count, errors_.cpu_data()) / bottom[0]->num();
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  /**
   * 首先根据损失函数,计算相对于diff的导数
   * 即: smoothl1(x)的求导
   */
  Dtype* diff_data = diff_.mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise
    if (fabs(val) < 1.) {
      diff_data[i] = val;
    } else {
      diff_data[i] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
  /**
   * diff进一步传播
   */
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      /**
       * 输入0和输入1的符号相反
       */
      const Dtype sign = (i == 0) ? 1 : -1;
      // 误差缩放,除以Num_
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),               // count
          alpha,                            // alpha
          diff_.cpu_data(),                 // a
          Dtype(0),                         // beta
          bottom[i]->mutable_cpu_diff());   // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
