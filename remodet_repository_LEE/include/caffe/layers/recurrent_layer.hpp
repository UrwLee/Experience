#ifndef CAFFE_RECURRENT_LAYER_HPP_
#define CAFFE_RECURRENT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype> class RecurrentLayer;

/**
 * @brief An abstract class for implementing recurrent behavior inside of an
 *        unrolled network.  This Layer type cannot be instantiated -- instead,
 *        you should use one of its implementations which defines the recurrent
 *        architecture, such as RNNLayer or LSTMLayer.
 */
template <typename Dtype>
class RecurrentLayer : public Layer<Dtype> {
 public:
  explicit RecurrentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reset();

  virtual inline const char* type() const { return "Recurrent"; }
  virtual inline int MinBottomBlobs() const {
    // x and cont (continuation indicators)
    int min_bottoms = 2;
    // 如果需要暴露隐层，表明隐层的输入Blobs由bottom输入
    if (this->layer_param_.recurrent_param().expose_hidden()) {
      vector<string> inputs;
      // 对于RNN，只有1个： h_0
      this->RecurrentInputBlobNames(&inputs);
      // 将隐层的input Blobs加入到bottom之中
      min_bottoms += inputs.size();
    }
    return min_bottoms;
  }
  // 最大bottom为最小bottom+1
  // +1： 如果隐层（h_{i}）的来源中包含外部固定的输入，则包含之：x_static
  virtual inline int MaxBottomBlobs() const { return MinBottomBlobs() + 1; }
  virtual inline int ExactNumTopBlobs() const {
    // 默认包含输出"o":是每个timestamp的输出拼接
    int num_tops = 1;
    // 如果隐层暴露的话，则将隐层的状态输出也包含
    if (this->layer_param_.recurrent_param().expose_hidden()) {
      vector<string> outputs;
      // 对于RNN来说，只包括：h_T
      this->RecurrentOutputBlobNames(&outputs);
      num_tops += outputs.size();
    }
    return num_tops;
  }

  // RNN只对于cont输入不进行BP
  // 对于其他Blob，默认进行反向传播
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Can't propagate to sequence continuation indicators.
    return bottom_index != 1;
  }

 protected:
  /**
   * @brief Fills net_param with the recurrent network architecture.  Subclasses
   *        should define this -- see RNNLayer and LSTMLayer for examples.
   */
  // RNN网络内部构建，即通过Unroll的方式进行网络展开
  // 纯虚函数，需要在子类中定义该方法
  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;

  /**
   * @brief Fills names with the names of the 0th timestep recurrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  // 定义RNN的隐层输入Blobs：h_0
  virtual void RecurrentInputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills shapes with the shapes of the recurrent input Blob&s.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  // 定义RNN的隐层状态Blobs的Shape: [1(single ts), N_, outputs]
  // 注意：RNN的隐层状态，无论是输入还是输出，都是等同的
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const = 0;

  /**
   * @brief Fills names with the names of the Tth timestep recurrent output
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  // 定义RNN的隐层输出Blobs: h_T
  virtual void RecurrentOutputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all timesteps.  Should return a name for each top Blob.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
   *        examples.
   */
  // 定义RNN的输出： o
  virtual void OutputBlobNames(vector<string>* names) const = 0;

  /**
   * @param bottom input Blob vector (length 2-3)
   *
   *   -# @f$ (T \times N \times ...) @f$
   *      the time-varying input @f$ x @f$.  After the first two axes, whose
   *      dimensions must correspond to the number of timesteps @f$ T @f$ and
   *      the number of independent streams @f$ N @f$, respectively, its
   *      dimensions may be arbitrary.  Note that the ordering of dimensions --
   *      @f$ (T \times N \times ...) @f$, rather than
   *      @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$
   *      independent input streams must be "interleaved".
   *
   *   -# @f$ (T \times N) @f$
   *      the sequence continuation indicators @f$ \delta @f$.
   *      These inputs should be binary (0 or 1) indicators, where
   *      @f$ \delta_{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
   *      @f$ n @f$ is the beginning of a new sequence, and hence the previous
   *      hidden state @f$ h_{t-1} @f$ is multiplied by @f$ \delta_t = 0 @f$
   *      and has no effect on the cell's output at timestep @f$ t @f$, and
   *      a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
   *      stream @f$ n @f$ is a continuation from the previous timestep
   *      @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
   *      updated hidden state and output.
   *
   *   -# @f$ (N \times ...) @f$ (optional)
   *      the static (non-time-varying) input @f$ x_{static} @f$.
   *      After the first axis, whose dimension must be the number of
   *      independent streams, its dimensions may be arbitrary.
   *      This is mathematically equivalent to using a time-varying input of
   *      @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
   *      across the @f$ T @f$ timesteps and concatenating with the time-varying
   *      input.  Note that if this input is used, all timesteps in a single
   *      batch within a particular one of the @f$ N @f$ streams must share the
   *      same static input, even if the sequence continuation indicators
   *      suggest that difference sequences are ending and beginning within a
   *      single batch.  This may require padding and/or truncation for uniform
   *      length.
   *
   * @param top output Blob vector (length 1)
   *   -# @f$ (T \times N \times D) @f$
   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
   *      <code>recurrent_param.num_output()</code>.
   *      Refer to documentation for particular RecurrentLayer implementations
   *      (such as RNNLayer and LSTMLayer) for the definition of @f$ y @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief A Net to implement the Recurrent functionality.
  //  RNN的实际内部网络
  shared_ptr<Net<Dtype> > unrolled_net_;

  /// @brief The number of independent streams to process simultaneously.
  // RNN的独立对象数
  int N_;

  /**
   * @brief The number of timesteps in the layer's input, and the number of
   *        timesteps over which to backpropagate through time.
   */
  // RNN的TimeStamp数
  int T_;

  /// @brief Whether the layer has a "static" input copied across all timesteps.
  // 是否包含x_static: [N_, ...]，每个对象有一个独立的输入向量
  bool static_input_;

  /**
   * @brief The last layer to run in the network. (Any later layers are losses
   *        added to force the recurrent net to do backprop.)
   */
  // 最后一层非损失层的参数，用于前向和反向计算
  int last_layer_index_;

  /**
   * @brief Whether the layer's hidden state at the first and last timesteps
   *        are layer inputs and outputs, respectively.
   */
  // 是否将隐层的状态暴露：状态输入由bottom输入，状态输出至top
  bool expose_hidden_;

  // 隐层的状态输入blobs列表
  vector<Blob<Dtype>* > recur_input_blobs_;
  // 隐层的状态输出blobs列表
  vector<Blob<Dtype>* > recur_output_blobs_;
  // RNN的输出blobs列表
  vector<Blob<Dtype>* > output_blobs_;
  // RNN的输入x
  Blob<Dtype>* x_input_blob_;
  // RNN的输入x_static
  Blob<Dtype>* x_static_input_blob_;
  // RNN的输入cont
  Blob<Dtype>* cont_input_blob_;
};

}  // namespace caffe

#endif  // CAFFE_RECURRENT_LAYER_HPP_
