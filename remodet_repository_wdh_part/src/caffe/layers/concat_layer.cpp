#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const ConcatParameter &concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  /**
   * 获取拼接的axis
   * 检查除拼接轴之外的所有尺度信息，严格匹配！
   * 对top的尺寸进行定义，拼接轴上的尺寸进行累加！
   */
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter &concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0)
        << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  /**
   * num_concats_： 拼接轴之前的size (outerSize)
   * concat_input_size_: 拼接轴之后的size (innerSize)
   */
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  // count
  int bottom_count_sum = bottom[0]->count();
  // 检查每个输入维度
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    /**
     * 检查每个轴的维度信息
     * 除了拼接轴，其他轴上的Size一定要匹配！
     */
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) {
        continue;
      }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    //累计尺寸
    bottom_count_sum += bottom[i]->count();
    // 拼接轴上的尺寸累加
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  // 定义输出的尺寸
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  /**
   * 如果只有一个输入，直接共享数据
   */
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  if (bottom.size() == 1) {
    return;
  }
  // get the output data
  Dtype *top_data = top[0]->mutable_cpu_data();
  // 从拼接轴上的0开始进行数据复制
  int offset_concat_axis = 0;
  // get the size of concat axis
  const int top_concat_axis = top[0]->shape(concat_axis_);
  // concat all the input bottom blobs
  for (int i = 0; i < bottom.size(); ++i) {
    // get the input data
    const Dtype *bottom_data = bottom[i]->cpu_data();
    // get the size of concat axis
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    /**
     * 单次复制数量：concat_input_size_ * bottom_concat_axis
     * 复制次数：  num_concats_
     * 源地址：bottom_data + n * bottom_concat_axis * concat_input_size_
     * 目的地址：top_data + n*top_concat_axis*concat_input_size_
     *         + offset_concat_axis*concat_input_size_
     * offset_concat_axis表示的是拼接轴上的序号
     */
    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_,
                 bottom_data + n * bottom_concat_axis * concat_input_size_,
                 top_data +
                     (n * top_concat_axis + offset_concat_axis) *
                         concat_input_size_);
    }
    /**
     * 拼接完一个bottom blob后，将拼接轴上的对应尺寸往上累加
     */
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  if (bottom.size() == 1) {
    return;
  }
  // 输入误差
  const Dtype *top_diff = top[0]->cpu_diff();
  // 拼接轴偏移
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      /**
       * 获取对应输入的误差指针
       * 将输入的误差数据进行分配
       * 源地址：top_diff+n*top_concat_axis*concat_input_size_
       *       + offset_concat_axis*concat_input_size_
       * 目的地址：bottom_diff + n * bottom_concat_axis * concat_input_size_
       */
      Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < num_concats_; ++n) {
        caffe_copy(bottom_concat_axis * concat_input_size_,
                   top_diff +
                       (n * top_concat_axis + offset_concat_axis) *
                           concat_input_size_,
                   bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
    /**
     * 拼接轴上的偏移向上累加
     */
    offset_concat_axis += bottom_concat_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);

} // namespace caffe
