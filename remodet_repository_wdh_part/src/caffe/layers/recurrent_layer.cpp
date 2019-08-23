#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RecurrentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0]至少包含2个维度: T和N
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  // 第一个维度是T
  T_ = bottom[0]->shape(0);
  // 第二个维度是N
  N_ = bottom[0]->shape(1);
  // 包含T个ts,以及N个独立的streams
  LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
            << T_ << " timesteps of " << N_ << " independent streams.";
  // bottom[1]也至少包含2个axis,且与bottom[0]的前两个维度相等
  // bottom[1]表示连续的索引序列，与bottom[1]严格对应
  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));

  // If expose_hidden is set, we take as input and produce as output
  // the hidden state blobs at the first and last timesteps.
  expose_hidden_ = this->layer_param_.recurrent_param().expose_hidden();

  // Get (recurrent) input/output names.
  // 首先定义RNN的输入输出Blobs名称
  vector<string> output_names;
  // RNN的输出: 只有一个，为"o"
  OutputBlobNames(&output_names);
  vector<string> recur_input_names;
  // RNN状态输入：只有一个，为“h_0”
  RecurrentInputBlobNames(&recur_input_names);
  vector<string> recur_output_names;
  // RNN的状态输出：只有一个，为“h_(T_)”
  RecurrentOutputBlobNames(&recur_output_names);
  // RNN：1
  const int num_recur_blobs = recur_input_names.size();
  CHECK_EQ(num_recur_blobs, recur_output_names.size());

  // If provided, bottom[2] is a static input to the recurrent net.
  // 隐层的暴露
  const int num_hidden_exposed = expose_hidden_ * num_recur_blobs;
  // 是否有static输入
  static_input_ = (bottom.size() > 2 + num_hidden_exposed);
  if (static_input_) {
    // 为每个stream分配一个static输入
    // bottom[2]-> [N_, ...]
    CHECK_GE(bottom[2]->num_axes(), 1);
    CHECK_EQ(N_, bottom[2]->shape(0));
  }

  // 构建网络参数
  NetParameter net_param;
  // 1. 首先添加一个数据输入层，bottom: x, cont, x_static (if has)
  // x: -> bottom[0]
  // cont: -> bottom[1]
  // x_static: -> bottom[2] (if has)
  LayerParameter* input_layer_param = net_param.add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  input_param->add_shape()->CopyFrom(input_shape);
  // top[1]：cont，维度与bottom[1]相同
  input_shape.Clear();
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    input_shape.add_dim(bottom[1]->shape(i));
  }
  input_layer_param->add_top("cont");
  input_param->add_shape()->CopyFrom(input_shape);
  // 如果定义了static，则每个stream都有一个static值
  // top[2]: x_static, 维度与bottom[2]相同
  if (static_input_) {
    input_shape.Clear();
    for (int i = 0; i < bottom[2]->num_axes(); ++i) {
      input_shape.add_dim(bottom[2]->shape(i));
    }
    input_layer_param->add_top("x_static");
    input_param->add_shape()->CopyFrom(input_shape);
  }

  // 2. 构建RNN的网络主体部分，即将网络展开为T_层
  // 每一层{i}的输出都拼接到最后的输出Blob中
  this->FillUnrolledNet(&net_param);

  // 将层名添加到网络中所有层名的头部_
  const string& layer_name = this->layer_param_.name();
  if (layer_name.size()) {
    for (int i = 0; i < net_param.layer_size(); ++i) {
      LayerParameter* layer = net_param.mutable_layer(i);
      layer->set_name(layer_name + "_" + layer->name());
    }
  }

  // Add "pseudo-losses" to all outputs to force backpropagation.
  // 3. 在最后的输出层后为每一个输出blob添加pseudo-losses损失层，强制反向传播
  // 加入该层没有本质含义，只是为了让该网络具有反向传播能力
  // 在网络构建的时候，需要网络顶层具有损失层
  // 实际计算时： 只计算到非损失层的顶部
  // (Setting force_backward is too aggressive as we may not need to backprop to
  // all inputs, e.g., the sequence continuation indicators.)
  vector<string> pseudo_losses(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    LayerParameter* layer = net_param.add_layer();
    pseudo_losses[i] = output_names[i] + "_pseudoloss";
    layer->set_name(pseudo_losses[i]);
    layer->set_type("Reduction");
    layer->add_bottom(output_names[i]);
    layer->add_top(pseudo_losses[i]);
    layer->add_loss_weight(1);
  }

  // 4. 根据网络参数创建网络，并设置调试信息
  unrolled_net_.reset(new Net<Dtype>(net_param));
  unrolled_net_->set_debug_info(
      this->layer_param_.recurrent_param().debug_info());

  // 5. 指向网络的输入Blobs接口： x / cont / x_static
  x_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("x").get());
  cont_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("cont").get());
  if (static_input_) {
    x_static_input_blob_ =
        CHECK_NOTNULL(unrolled_net_->blob_by_name("x_static").get());
  }

  // 6. 指向网络的输入/输出隐层Blobs接口： recur_input_blobs_ / recur_output_blobs_
  recur_input_blobs_.resize(num_recur_blobs);
  recur_output_blobs_.resize(num_recur_blobs);
  for (int i = 0; i < recur_input_names.size(); ++i) {
    recur_input_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_input_names[i]).get());
    recur_output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_output_names[i]).get());
  }

  // 7. 指向网络的输出Blobs接口： o
  CHECK_EQ(top.size() - num_hidden_exposed, output_names.size())
      << "OutputBlobNames must provide an output blob name for each top.";
  output_blobs_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
  }

  // We should have 2 inputs (x and cont), plus a number of recurrent inputs,
  // plus maybe a static input.
  // 检查网络的输入Blobs数量与外部提供的Blobs数量是否匹配？？
  CHECK_EQ(2 + num_recur_blobs + static_input_,
           unrolled_net_->input_blobs().size());

  // This layer's parameters are any parameters in the layers of the unrolled
  // net. We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  // 将网络中的所有参数映射到本地的blobs_
  this->blobs_.clear();
  for (int i = 0; i < unrolled_net_->params().size(); ++i) {
    if (unrolled_net_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << unrolled_net_->param_display_names()[i];
      this->blobs_.push_back(unrolled_net_->params()[i]);
    }
  }
  // Check that param_propagate_down is set for all of the parameters in the
  // unrolled net; set param_propagate_down to true in this layer.
  // 默认内部的所有参数都需要反向传播
  for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
    for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
      CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }

  // 本地参数设置为都需要反向传播
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Set the diffs of recurrent outputs to 0 -- we can't backpropagate across
  // batches.
  // 将隐层最后状态输出的误差初始化为0
  for (int i = 0; i < recur_output_blobs_.size(); ++i) {
    caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
              recur_output_blobs_[i]->mutable_cpu_diff());
  }

  // Check that the last output_names.size() layers are the pseudo-losses;
  // set last_layer_index so that we don't actually run these layers.
  // 检查： 网络的最后几层全部都应该是损失层
  const vector<string>& layer_names = unrolled_net_->layer_names();
  last_layer_index_ = layer_names.size() - 1 - pseudo_losses.size();
  for (int i = last_layer_index_ + 1, j = 0; i < layer_names.size(); ++i, ++j) {
    CHECK_EQ(layer_names[i], pseudo_losses[j]);
  }
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  CHECK_EQ(T_, bottom[0]->shape(0)) << "input number of timesteps changed";
  N_ = bottom[0]->shape(1);
  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));
  // 网络的x输入与bottom[0]尺寸相同
  x_input_blob_->ReshapeLike(*bottom[0]);
  vector<int> cont_shape = bottom[1]->shape();
  // 网络的cont输入与bottom[1]尺寸相同
  cont_input_blob_->Reshape(cont_shape);
  // 网络的x_static输入与bottom[2]尺寸相同
  if (static_input_) {
    x_static_input_blob_->ReshapeLike(*bottom[2]);
  }
  // 定义隐层的shape
  vector<BlobShape> recur_input_shapes;
  // 产生隐层的shape[1, N_, outputs]
  RecurrentInputShapes(&recur_input_shapes);
  // shapes数与blobs要严格一致，每个blob对应一个shape
  CHECK_EQ(recur_input_shapes.size(), recur_input_blobs_.size());
  for (int i = 0; i < recur_input_shapes.size(); ++i) {
    recur_input_blobs_[i]->Reshape(recur_input_shapes[i]);
  }
  // 将整个网络依据输入的shape和隐层的状态shape完成全部reshape
  unrolled_net_->Reshape();

  // 网络的输入blob映射到bottom[0]
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  // 网络的cont-blob映射到bottom[1]
  cont_input_blob_->ShareData(*bottom[1]);
  // 网络的x_static-blob映射到bottom[2]
  if (static_input_) {
    x_static_input_blob_->ShareData(*bottom[2]);
    x_static_input_blob_->ShareDiff(*bottom[2]);
  }
  // 如果隐层接口暴露，还要将该状态的输入/输出映射到bottom/top
  if (expose_hidden_) {
    // 先进行shape检查，然后完成数据映射
    const int bottom_offset = 2 + static_input_;
    for (int i = bottom_offset, j = 0; i < bottom.size(); ++i, ++j) {
      CHECK(recur_input_blobs_[j]->shape() == bottom[i]->shape())
          << "bottom[" << i << "] shape must match hidden state input shape: "
          << recur_input_blobs_[j]->shape_string();
      recur_input_blobs_[j]->ShareData(*bottom[i]);
    }
  }
  // 完成本层top输出与网络的输出Blobs之间的映射
  for (int i = 0; i < output_blobs_.size(); ++i) {
    top[i]->ReshapeLike(*output_blobs_[i]);
    top[i]->ShareData(*output_blobs_[i]);
    top[i]->ShareDiff(*output_blobs_[i]);
  }
  // 如果隐层输出暴露，则将隐层的状态输出映射到top，注意这里只是做了reshape，映射部分在
  // 前向计算中实现
  if (expose_hidden_) {
    const int top_offset = output_blobs_.size();
    for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
      top[i]->ReshapeLike(*recur_output_blobs_[j]);
    }
  }
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Reset() {
  // "Reset" the hidden state of the net by zeroing out all recurrent outputs.
  // Reset只是将隐层的状态输出全部初始化为0
  for (int i = 0; i < recur_output_blobs_.size(); ++i) {
    caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
              recur_output_blobs_[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time: reshare all the internal shared blobs, which may
  // currently point to a stale owner blob that was dropped when Solver::Test
  // called test_net->ShareTrainedLayersWith(net_.get()).
  // TODO: somehow make this work non-hackily.
  // 处于测试状态时，对于共享其它的参数进行设置，一般无用。
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeights();
  }

  // 检查： 隐层的输入输出的数量应该严格一致
  DCHECK_EQ(recur_input_blobs_.size(), recur_output_blobs_.size());
  /**
   * 注意： 如果隐层的状态接口没有暴露，则需要将上一次计算的状态输入作为本次计算的状态输入
   * 如果隐层状态接口暴露，则隐层的输入是由bottom提供的，因此直接进行前向计算即可
   */
  if (!expose_hidden_) {
    for (int i = 0; i < recur_input_blobs_.size(); ++i) {
      const int count = recur_input_blobs_[i]->count();
      DCHECK_EQ(count, recur_output_blobs_[i]->count());
      const Dtype* timestep_T_data = recur_output_blobs_[i]->cpu_data();
      Dtype* timestep_0_data = recur_input_blobs_[i]->mutable_cpu_data();
      caffe_copy(count, timestep_T_data, timestep_0_data);
    }
  }

  // 前向计算。
  unrolled_net_->ForwardTo(last_layer_index_);

  // 如果隐层暴露，将隐层的最后状态输出即可
  if (expose_hidden_) {
    const int top_offset = output_blobs_.size();
    for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
      top[i]->ShareData(*recur_output_blobs_[j]);
    }
  }
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // cont是序列编号，无需传播
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";

  // TODO: skip backpropagation to inputs and parameters inside the unrolled
  // net according to propagate_down[0] and propagate_down[2]. For now just
  // backprop to inputs and parameters unconditionally, as either the inputs or
  // the parameters do need backward (or Net would have set
  // layer_needs_backward_[i] == false for this layer).
  // 误差从输出传播过来，网络的输出接入到其他的外部layer，因此：
  // 直接从该层开始计算即可。
  // 外部反馈的误差最后传播到last_layer_index_层的diff处，因此从该层向前传播即可
  // 最后误差会传播至输入层： x / x_static / recur_input_blobs_ ...
  unrolled_net_->BackwardFrom(last_layer_index_);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RecurrentLayer, Forward);
#endif

INSTANTIATE_CLASS(RecurrentLayer);

}  // namespace caffe
