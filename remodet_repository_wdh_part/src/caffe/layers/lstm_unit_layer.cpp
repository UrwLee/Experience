#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

/**
 * Sigmoid函数
 */
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

/**
 * Tanh函数
 */
template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

/**
 * Reshape
 */
template <typename Dtype>
void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 样本数量
  const int num_instances = bottom[0]->shape(1);
  // 检查维度
  // bottom[2]->num_axis() == 2  [cont_{ts} -> only 2 axis]
  // others: bottom[i]->num_axis() == 3 [1, N_, num_output]
  // bottom[i]->shape(0) == 1 -> single Ts
  // bottom[i]->shape(1) == N_
  // 除了bottom[2] -> [1, N_]
  // 其他的bottom的shape全部是： [1, N_, num_output]
  // bottom[0] -> c_{ts-1}
  // bottom[1] -> gate_{ts}
  // bottom[2] -> cont_{ts}
  for (int i = 0; i < bottom.size(); ++i) {
    if (i == 2) {
      CHECK_EQ(2, bottom[i]->num_axes());
    } else {
      CHECK_EQ(3, bottom[i]->num_axes());
    }
    CHECK_EQ(1, bottom[i]->shape(0));
    CHECK_EQ(num_instances, bottom[i]->shape(1));
  }
  // 隐藏单元数
  hidden_dim_ = bottom[0]->shape(2);
  // bottom[1] -> gated_{ts} -> [1, N_, 4*num_output]
  // {i_{ts}, f_{ts}, i_{ts}, g_{ts}}
  CHECK_EQ(num_instances, bottom[1]->shape(1));
  CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
  // top[0] -> c_{ts}
  // top[1] -> h_{ts}
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);

  // X_acts_ -> gated
  X_acts_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // num samples
  const int num = bottom[0]->shape(1);
  // gate_dim
  const int x_dim = hidden_dim_ * 4;
  // c_{ts-1}
  const Dtype* C_prev = bottom[0]->cpu_data();
  // gate_{ts}
  const Dtype* X = bottom[1]->cpu_data();
  // cont_{ts}
  const Dtype* cont = bottom[2]->cpu_data();
  // c_{ts}
  Dtype* C = top[0]->mutable_cpu_data();
  // h_{ts}
  Dtype* H = top[1]->mutable_cpu_data();
  // 计算所有样本
  for (int n = 0; n < num; ++n) {
    // 计算所有gate信号
    for (int d = 0; d < hidden_dim_; ++d) {
      // 计算i/f/o/g的激活
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      // 取第d个cell
      const Dtype c_prev = C_prev[d];
      // 计算cell更新值
      const Dtype c = f * c_prev + i * g;
      C[d] = c;
      // 计算h更新值，用于输出
      const Dtype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    // 计算完一个样本后，C_prev跳过一个样本
    C_prev += hidden_dim_;
    // X跳过一个样本
    X += x_dim;
    // C跳过一个样本
    C += hidden_dim_;
    // H跳过一个样本
    H += hidden_dim_;
    // cont跳过一个样本
    ++cont;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // cont不支持反向传播
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  // 如果0/1都不需要，则直接返回
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  // 样本数
  const int num = bottom[0]->shape(1);
  // gate的dim： i/f/o/g
  const int x_dim = hidden_dim_ * 4;
  // c_{ts-1}
  const Dtype* C_prev = bottom[0]->cpu_data();
  // gate_{ts}
  const Dtype* X = bottom[1]->cpu_data();
  // cont_{ts}
  const Dtype* cont = bottom[2]->cpu_data();
  // c_{ts}
  const Dtype* C = top[0]->cpu_data();
  // h_{ts}
  const Dtype* H = top[1]->cpu_data();
  // cdiff_{ts}
  const Dtype* C_diff = top[0]->cpu_diff();
  // hdiff_{ts}
  const Dtype* H_diff = top[1]->cpu_diff();
  // cdiff_{ts-1}
  Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
  // gate_diff_{ts}
  Dtype* X_diff = bottom[1]->mutable_cpu_diff();
  // 遍历所有样本
  for (int n = 0; n < num; ++n) {
    // 遍历所有隐层单元
    for (int d = 0; d < hidden_dim_; ++d) {
      // 计算gate的激活
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      // 获取cell过去值
      const Dtype c_prev = C_prev[d];
      // 获取当前cell
      const Dtype c = C[d];
      // 获取cell的激活值
      const Dtype tanh_c = tanh(c);
      // 反馈的cell误差
      Dtype* c_prev_diff = C_prev_diff + d;
      // 反馈的gate误差
      Dtype* i_diff = X_diff + d;
      Dtype* f_diff = X_diff + 1 * hidden_dim_ + d;
      Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
      Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;

      /**
       * cell的返回误差计算
       */
      const Dtype c_term_diff =
          C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
      // 返回cell的误差
      *c_prev_diff = c_term_diff * f;
      // 返回gates的误差
      *i_diff = c_term_diff * g * i * (1 - i);
      *f_diff = c_term_diff * c_prev * f * (1 - f);
      *o_diff = H_diff[d] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
    // 指向下一个样本
    C_prev += hidden_dim_;
    // x
    X += x_dim;
    // c和h
    C += hidden_dim_;
    H += hidden_dim_;
    // c/h/x/c_prev
    C_diff += hidden_dim_;
    H_diff += hidden_dim_;
    X_diff += x_dim;
    C_prev_diff += hidden_dim_;
    // cont
    ++cont;
  }
}

#ifdef CPU_ONLY
STUB_GPU(LSTMUnitLayer);
#endif

INSTANTIATE_CLASS(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
