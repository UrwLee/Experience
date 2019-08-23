#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * 定义了RNN的隐层输入Blobs： 只有一个，为"h_0"
 */
template <typename Dtype>
void RNNLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_0";
}

/**
 * 定义了RNN的隐层输出Blobs： 只有一个，为"h_T"
 */
template <typename Dtype>
void RNNLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_" + format_int(this->T_);
}

/**
 * 定义了RNN的隐层状态shape： 只要一个，为 [1, N_, num_outputs]
 * num_outputs: 外部参数定义的隐藏层状态的数量，即隐层的神经元数
 */
template <typename Dtype>
void RNNLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  shapes->resize(1);
  (*shapes)[0].Clear();
  (*shapes)[0].add_dim(1);  // a single timestep
  (*shapes)[0].add_dim(this->N_);
  (*shapes)[0].add_dim(num_output);
}

/**
 * 定义了RNN的输出Blobs： 只有一个，为“o”
 */
template <typename Dtype>
void RNNLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "o";
}

/**
 * 创建RNN的网络主体架构
 */
template <typename Dtype>
void RNNLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  // 获取隐藏层的单元数
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  // 获取权重和bias的初始化方式
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  // 定义不含有bias的IP层参数
  LayerParameter hidden_param;
  // 全连接层
  hidden_param.set_type("InnerProduct");
  // 设置输出数
  hidden_param.mutable_inner_product_param()->set_num_output(num_output);
  // 不含有bias
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  // 全连接计算的axis：2， 0-TS，1-N_， 2-特征向量
  hidden_param.mutable_inner_product_param()->set_axis(2);
  // 参数初始化
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  // 定义包含bias的IP层参数
  LayerParameter biased_hidden_param(hidden_param);
  // 含有bias，并初始化
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  // 定义累加层SUM
  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  // 定义激活层Tanh
  LayerParameter tanh_param;
  tanh_param.set_type("TanH");

  // 定义Scale层
  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  // Slice层，对axis-0进行slice操作： 按顺序进行序列化输出top[0] - top[T_]
  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  // 定义隐层的状态shape：[1, N_, num_outputs]
  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(1, input_shapes.size());

  // 添加隐层状态的输入层：h_0
  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  // 添加序列时序的slice层：cont_slice
  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xh_x = W_xh * x + b_h
  // 添加输入x的转换层： 通过{W_xh / b_h} 完成第一步计算
  // 输出W_xh_x
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xh");
    x_transform_param->add_param()->set_name("b_h");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xh_x");
    x_transform_param->add_propagate_down(true);
  }

  // 定义x_static的状态输入
  if (this->static_input_) {
    // Add layer to transform x_static to the hidden state dimension.
    //     W_xh_x_static = W_xh_static * x_static
    // x_static: [N_, ...]
    // W_xh_x_static_preshape: [N_, ...]
    // W_xh_x_static: [1, N_, num_outputs]
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    // 从axis-1进行计算
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xh_x_static");
    x_static_transform_param->add_param()->set_name("W_xh_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xh_x_static_preshape");
    x_static_transform_param->add_propagate_down(true);

    // 定义reshape层: 需要与隐层的状态shape进行匹配
    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    // 1
    new_shape->add_dim(1);  // One timestep.
    // Should infer this->N as the dimension so we can reshape on batch size.
    new_shape->add_dim(-1);
    // 最后一个维度要与隐层的数量一致
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->set_name("W_xh_x_static_reshape");
    reshape_param->add_bottom("W_xh_x_static_preshape");
    reshape_param->add_top("W_xh_x_static");
  }

  // 定义输入x的序列化层： slice
  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->set_name("W_xh_x_slice");
  x_slice_param->add_bottom("W_xh_x");

  // 定义输出的拼接层： Concat, 将每个stamp的输出全部进行拼接
  LayerParameter output_concat_layer;
  output_concat_layer.set_name("o_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("o");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  // 构建所有的内部blobs和layers： 累计T个layers
  for (int t = 1; t <= this->T_; ++t) {
    // 定义时间戳
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    // 1. 序列展开： W_xh_x和cont
    // 每个时间戳对应一个slice序列
    // 也就是将一个连续T个样本展开成T个top[i] -> 依次送入到指定的i_layer的x
    // : cont_{ts}
    // : W_xh_x_{ts}
    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xh_x_" + ts);

    // Add layer to flush the hidden state when beginning a new sequence,
    // as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    /**
     * 2. 状态门控： cont_{ts} -> 控制状态连接的强度
     * 注意： cont[t]控制着前一个隐层状态是否需要接入到当前的隐层状态中
     * 相当于一个门的作用，如果为0， 则不接入，两层之间的联系是断开的；
     * 如果为1， 则全连接，两层之间的的状态是直接传递的；
     * 如果是0-1，部分状态会传递到下一个ts
     * h_conted_{ts-1} = cont_{ts} * h_{ts-1}
     */
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    /**
     * 3. 状态连接： 通过{W_hh}完成状态计算
     * W_hh_h_{ts-1} = W_hh * h_conted_{ts-1}
     */
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("W_hh_h_" + tm1s);
      w_param->add_param()->set_name("W_hh");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hh_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    /**
     * 4. 状态累加
     *  ： 状态连接的传递值
     *  ： 输入的传递值
     *  ： x_static传递值
     *  h_neu_{ts} = W_xh_x_{ts} + W_hh_h_{ts-1} + W_xh_x_static
     */
    {
      LayerParameter* h_input_sum_param = net_param->add_layer();
      h_input_sum_param->CopyFrom(sum_param);
      h_input_sum_param->set_name("h_input_sum_" + ts);
      h_input_sum_param->add_bottom("W_hh_h_" + tm1s);
      h_input_sum_param->add_bottom("W_xh_x_" + ts);
      if (this->static_input_) {
        h_input_sum_param->add_bottom("W_xh_x_static");
      }
      h_input_sum_param->add_top("h_neuron_input_" + ts);
    }

    /**
     * 5. 隐层状态激活
     *    h_{ts} = Tanh(h_neu_{ts})
     */
    {
      LayerParameter* h_neuron_param = net_param->add_layer();
      h_neuron_param->CopyFrom(tanh_param);
      h_neuron_param->set_name("h_neuron_" + ts);
      h_neuron_param->add_bottom("h_neuron_input_" + ts);
      h_neuron_param->add_top("h_" + ts);
    }

    /**
     * 6. 输出计算：
     *    W_ho_h_{ts} = W_ho * h_{ts} + b_o
     */
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(biased_hidden_param);
      w_param->set_name("W_ho_h_" + ts);
      w_param->add_param()->set_name("W_ho");
      w_param->add_param()->set_name("b_o");
      w_param->add_bottom("h_" + ts);
      w_param->add_top("W_ho_h_" + ts);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    /**
     * 7. 输出激活：
     *    o_{ts} = Tanh(W_ho_h_{ts})
     */
    {
      LayerParameter* o_neuron_param = net_param->add_layer();
      o_neuron_param->CopyFrom(tanh_param);
      o_neuron_param->set_name("o_neuron_" + ts);
      o_neuron_param->add_bottom("W_ho_h_" + ts);
      o_neuron_param->add_top("o_" + ts);
    }

    /**
     * 8. 输出向量拼接
     *   o += concat(o_{ts})
     */
    output_concat_layer.add_bottom("o_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  // 将输出拼接层加入网络架构之中
  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(RNNLayer);
REGISTER_LAYER_CLASS(RNN);

}  // namespace caffe
