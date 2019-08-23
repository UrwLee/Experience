#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_)); // vector 1, c
  sum_multiplier_.Reshape(mult_dims);
  Dtype *multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  /**
   * 定义softmax轴之前和之后的维度累积
   */
  outer_num_ = bottom[0]->count(0, softmax_axis_); // n
  inner_num_ = bottom[0]->count(softmax_axis_ + 1); // h*w
  /**
   * scale_在softmax轴上的维度是1，为一个scalar
   * 而在其他轴上，保留输入的维度尺寸
   */
  vector<int> scale_dims = bottom[0]->shape(); // vector : n, c, h, w
  scale_dims[softmax_axis_] = 1; // c = 1
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  /**
   * 输入数据指针
   * 输出数据指针
   * scale数据指针
   */
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *scale_data = scale_.mutable_cpu_data();
  /**
   * channals -> softmax轴上的维度，即分类数
   */
  int channels = bottom[0]->shape(softmax_axis_); // c
  /**
   * dim -> softmax轴及其下方的轴的累积
   */
  int dim = bottom[0]->count() / outer_num_; // c*h*w
  /**
   * 复制输入到输出
   */
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  /**
   * outer_num_可以认为是需要分类的所有样本数，样本数×网格数（boxes数）...
   */
  for (int i = 0; i < outer_num_; ++i) {
    // scale是一个最大值平面，初始化为输入的第一个平面
    // 平面也有可能是一个点，如果softmax是最后的一个轴
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data); // h*w
    /**
     * 遍历所有softmax轴上的平面
     */
    for (int j = 0; j < channels; j++) { // c
      /**
       * 遍历当前平面上所有点，选取最大点到平面上，构成新的最大平面
       */
      for (int k = 0; k < inner_num_; k++) { // h*w
        scale_data[k] =
            std::max(scale_data[k], bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    /**
     * sum_multiplier_ -> [channels]  (1, c)
     * scale_data -> [b,1,h,w]
     * top_data -> [b,c,h,w]
     * sum_multiplier_将softmax轴上的维度扩展成原状
     * 输入平面上的每个点都减去该softmax轴上的最大值
     */
    // A.T, B.T, M, N, K, alpha, A, B, beta, C
    // M = c,   A, C  h
    // N = h*w, B, C  w
    //  k = 1,  A, B  h
    // alpha=-1,  A: (1,c) sum_multiplier_.cpu_data()
              //  B: (b,1,h,w), C:1    scale_data
              //  C:  top_data
    // C = alpha * A * B + beta * C 
    // top_data = - (1,c) * (b,1,h,w) + (b,c,h,w)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                          -1., sum_multiplier_.cpu_data(), scale_data, 1.,
                          top_data);  // 
    // 指数
    // dim: c*h*w
    // top_data 
    caffe_exp<Dtype>(dim, top_data, top_data);

    // A.T , M, N, alpha, A, x, beta,y
    // M = c,   A,  h
    // N = h*w, A   w
    // alpha=1 , top_data: A , (h*w, c)
    //  x: sum_multiplier_  (1,c)
    // y=alpha*A*x+beta*y
    //  scale_data = (h*w, c) * (c) + 0 
    // 沿softmax方向求和
    // [b,1,h,w] -> 每个点上的累加和
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1., top_data,
                          sum_multiplier_.cpu_data(), 0., scale_data);
    // 除法，完成softmax计算
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  /**
   *  输入误差
   *  输出数据
   *  输出误差
   *  scale_data -> 指数累加和【沿着softmax轴】
   */
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *top_data = top[0]->cpu_data();
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype *scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  /**
   * 将输入误差直接复制到输出误差
   */
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  /**
   * 遍历所有外部轴
   */
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    /**
     * 计算间隔：inner_num_
     * 遍历inner_num_，累加channels上的结果：scale_data
     * top_data * top_diff
     */
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(
          channels, bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    /**
     *  sum_multiplier_ [channels]
     *  scale_data [h,w]
     *  输出误差
     */
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                          -1., sum_multiplier_.cpu_data(), scale_data, 1.,
                          bottom_diff + i * dim);
  }
  // elementwise multiplication
  /**
   * 乘以输出，完成误差传播
   * elementwise计算
   */
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

} // namespace caffe
