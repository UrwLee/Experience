#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //  调用损失层的Reshape方法
  LossLayer<Dtype>::Reshape(bottom, top);
  // 检查维度,channel必须相同
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  //  误差与输入0相同
  diff_.ReshapeLike(*bottom[0]);
}

/**
 * 正向计算:
 * 首先计算误差,作为向后传播的源
 * 然后计算MSE,作为输出损失
 */
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // 计算误差的平方和
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  // Loss = 1/2 * {sum of square error} / N
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  // 损失层来说,输出就一个值
  top[0]->mutable_cpu_data()[0] = loss;
}

/**
 * 欧几里得损失对两个输入都会进行反向传播,方向相反
 */
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      // 输入0取+1
      // 输入1取-1
      const Dtype sign = (i == 0) ? 1 : -1;
      // +/-1/N
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      // 误差均衡传播
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
