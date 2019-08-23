#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// 两个构造函数， 
template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages,
    const Net* root_net)
    : root_net_(root_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // LOG(INFO) << "1: =============param_file name: " << param_file; 
  // Set phase, stages and level
  // 
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  CHECK(Caffe::root_solver() || root_net_) 
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param; 
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver()) // 将不符合 net stage规则的 和 exclude的layer 去掉， 
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param; 
  InsertSplits(filtered_param, &param); //此函数作用是，对于底层一个输出blob对应多个上层的情况，则要在加入分裂层，形成新的网络。这么做的主要原因是多个层反传给该blob的梯度需要累加。
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  // 建立blob的名称和索引的映射表
  map<string, int> blob_name_to_idx; // net 中的blob  层之间的结果
  //可用blobs的集合
  set<string> available_blobs; // 可用 blobs？？ 
  // 使用内存统计
  memory_used_ = 0;
  // For each layer, set up its input and output
  // 定义layers层数
  bottom_vecs_.resize(param.layer_size()); //vector<vector<Blob<Dtype>*> > bottom_vecs_， [layer_size][]
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  // 遍历所有层
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) { 
    // For non-root solvers, whether this layer is shared from root_net_.
    // 对于非root_solver,检查其是否共享自root_net_
    // 检查root_net_->layer[id]是否是ShareInParallel
    // LOG(FATAL);
    bool share_from_root = !Caffe::root_solver() // ??? 什么东西
        && root_net_->layers_[layer_id]->ShareInParallel(); // vector<shared_ptr<Layer<Dtype> > >layers_ 共享指针的函数 
    // Inherit phase from net if unset.
    // 对于没有设置phase的layer,一律设置为网络的phase_参数
    // 即:其余层的phase默认与net的phase相同
    if (!param.layer(layer_id).has_phase()) { //如果net 中layer 没有phase ， 设置为net 默认phase_
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // 获取该层的参数
    const LayerParameter& layer_param = param.layer(layer_id); 
    // 如果该层定义了反向传播参数,则必须与bottom的size相同， layer 进行反传， 则每个bottom 都要进行反传？ 
    if (layer_param.propagate_down_size() > 0) { 
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    // 检查该层是否共享至'root_solver'
    // 否则,需要创建该layer
    if (share_from_root) {
      // 与根网络的指定层共享
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]); // 如果与root_net
      // 将该层的共享标记设置为True,表示为共享层
      layers_[layer_id]->SetShared(true);
    } else {
      // 创建新层
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param)); // 创造新layer 存储在共享指针中，  LayerRegistry 与shared_ptr相关
    }
    // 定义层名
    layer_names_.push_back(layer_param.name()); // 层的基本属性 layer_names_ type: vector<string>
    // net 中layer 基本属性 phase, shared_ptr , layer_name, propagate_down
    // 打印层创建信息
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    // 默认layer不参与反向传播过程 本层是否反向传播 , 与层中对应参数相关
    bool need_backward = false; 

    // Figure out this layer's input and output
    // 遍历输入blobs
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); // for layer_id 每layer 中的 bottom 进行遍历
         ++bottom_id) {
      /**
       * param:网络参数
       * layer_id:层号
       * bottom_id:层的输入blob-ID
       * available_blobs: 可用Blobs type: set<string>
       * blob_name_to_idx:为每个输入blobs分配blob索引ID type: map<string, int>
       bottom 初始化 
       */
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      // 该输入blob是否需要反向传播
      need_backward |= blob_need_backward_[blob_id];
    }
    // 查找输出top-blobs数量
    int num_top = layer_param.top_size(); 
    // 遍历该层的top-blobs
    for (int top_id = 0; top_id < num_top; ++top_id) {
      /**
       * param: net 网络参数
       * layerid: 层号
       * top_id: 该层的输出编号
       * available_blobs: 可用blobs集合, 由blob名称组成的集合
       * blob_name_to_idx: 为blob分配其编号,并返回
       */
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx); //  available_blobs 在哪里进行了初始化？
      // Collect Input layer tops as Net inputs.
      // 对于类型为INPUT的layer,其输出被定义为网络的输入
      if (layer_param.type() == "Input") {
        // Input层被定义为最先输入层
        // 其ID从0开始
        // net_input_blob_indices_: 输入blobs的id
        // net_input_blobs_: 输入blobs
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    // 获取本层的对象
    Layer<Dtype>* layer = layers_[layer_id].get();
    // 如果本层需要AutoTopBlobs定义,则需要对输出进行扩展
    if (layer->AutoTopBlobs()) { // default false , loss layer 中自动加层
      // 获取其需要的输出数:max(最小输出数,应该输出数)
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      // 为剩下还需要输出的top-blobs进行添加
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        // 继续添加剩下的top-blobs
        // 注意:只是添加,但并不加入可用blobs列表,也不为其分配blobs索引
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    // 如果使用共享层,则进行如下设置: 设置与root_net_相同的shape
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      // 否则调用layer自身的setup方法
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]); // 调用layer的 SetUp 函数
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    // 遍历该层的输出blobs列表
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) { 
      // 注意:默认的top-blobs的损失权重都是0
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) { 
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      // 设置top-blobs的损失权重,注意每一层的每个输出blob都会有这个值
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      // 输出每一个top-blob的shape信息
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      // 如果该top-blob的算是不为零,则打印其信息
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      // 获取每个top-blob的存储器信息
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    // 打印存储器信息
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    // 获取参数param的数量 
    const int param_size = layer_param.param_size(); 
    /* param 在layer中的形式, 每个param中设置对应可学习参数的 param
    param {
      name: "conv1_paramconv0"
      lr_mult: 0
      decay_mult: 1
    }
    */
    // 获取该层的参数blobs的数量
    const int num_param_blobs = layers_[layer_id]->blobs().size(); // layer 的blobs 是w, b可学习参数
    // 注意,参数的数量一定不能超过本层使用的blobs的数量,否则出错
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    // 默认的参数描述 初始化在构造函数中 设置lr, decy...
    ParamSpec default_param_spec; 
    // 在proto中描述的优先使用描述的参数信息
    // 否则,之外的部分,使用默认参数信息
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) { // 对每个blob (top, bottom)进行分析
      const ParamSpec* param_spec = (param_id < param_size) ? // param_id 表示 layer中存在的 w,b, param_size 表示设置的w, b的对应参数
          &layer_param.param(param_id) : &default_param_spec; // 如果设置了w,b 但没有设置相对应的参数. 就使用默认参数代替
      // 注意,该参数的lr_mult不能为0,否则该参数不参与反向计算
      const bool param_need_backward = param_spec->lr_mult() != 0; 
      // 如果参数需要反向计算,本层也默认需要反向计算
      need_backward |= param_need_backward; // 
      // 设置参数的反向传播开关
      // param_id: 该层的第几个参数
      // param_need_backward: 该参数是否需要更新 
      // 一个layer中 有多个 反向传播flag,
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    // 对所有的参数,添加到参数列表中去
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    // 设置该层的反向传播标记
    layer_need_backward_.push_back(need_backward);
    // 如果需要反向传播,则该层的所有输出blobs都设置为需要反向传播
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }//所有layer遍历完毕

  // layer 中的blob与 net中的blob 不同, layer中 blob为可学习参数, net中为 层间传递的 blob

  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  // 贡献损失的blobs
  set<string> blobs_under_loss;
  // 不需要反向传播的blobs
  set<string> blobs_skip_backp;
  // 从最后一层向前遍历所有的层
  // 第一个查找到的损失层就是net的损失来源
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) { //索引从0 开始, 5个layer, 开始与id=4
    // 默认该层不贡献损失
    bool layer_contributes_loss = false;
    // 默认该层跳过反向传播
    bool layer_skip_propagate_down = true;
    // 遍历该层的输出blobs
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) { // 
      // 获取该top blob的名字, net 中blobWie 层之间blob
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]]; 
      // 该top-blob需要损失,有损失参数, 或者在贡献损失的blobs 中查找到该blob(对损失有贡献), 则对该层的loss有贡献
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        // 贡献损失
        layer_contributes_loss = true;
      }
      // 不跳过反向传播,则该层也不跳过
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      // 只要查找到损失层,且需要反向传播,则立即退出,整个网络从后至前,只允许一个损失层
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    // 如果前面分析该层需要反向传播,但此处得到的该层可以跳过,则重新设置该层的相关信息
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      // 该layer不需发现传播
      layer_need_backward_[layer_id] = false;
      // 该层的所有输入blobs都不需要
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    // 如果不在损失层之下,也就是层在损失层上面,则肯定也不要反向传播
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; } // data-conv1-loss-acc  acc 在loss下方,不反传
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        // 对于需要的层,打印其反向传播信息
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        // 否则,打印不需要的传播信息
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    // 遍历所有输入blobs
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      // 如果在损失层之下,push其名称
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        // 否则,将其设置为无需反向
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      // 对于不需要反向计算的blobs,push到跳过的blobs列表中
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }

  // Handle force_backward if needed.
  // 处理强制反向传播
  // 网络具有强制反向传播的参数
  if (param.force_backward()) {
    // 遍历所有层
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      // 强制设置为true
      layer_need_backward_[layer_id] = true;
      // 输入的blobs进行设置
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id); // 具体层中设置 哪些层可以被强制反传, 哪些层不可以强制反传
        //  设置
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      // 强制所有的参数为反向
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }//网络强制反向传播

  // In the end, all remaining blobs are considered output blobs.
  // 所有剩下保持的blobs都设置为输出blobs  //不为 层之间的 blob 就是剩下的 输出
  // 迭代遍历所有可用的blobs
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    // 将剩下的push到网络输出blobs中
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    // push其索引
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  // 创建blobs名称到索引的map映射
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id; // blob_name_index_ type: map<string, int> 
  }
  // 创建layers名称到索引的map映射 创建新的 name 和index 索引
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;  // layer_names_index_ type: map<string, int> 
  }
  // 共享权重信息
  ShareWeights(); // ? 
  // 获取网络参数的调试状态信息
  debug_info_ = param.debug_info();
  // 初始化结束
  // 注意:LOG_IF(INFO,...) ->条件状态,只有作为root_solver的才会输出
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  // 获取state
  NetState net_state(param.state());
  // 复制所有参数
  param_filtered->CopyFrom(param);
  // 清除其层信息
  param_filtered->clear_layer();
  // 遍历层信息
  for (int i = 0; i < param.layer_size(); ++i) {
    // 获取第i层的参数
    const LayerParameter& layer_param = param.layer(i);
    // 获取名称
    const string& layer_name = layer_param.name();
    // include和exclude不能同时为1
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    // 默认是include: 不包含include字段
    bool layer_included = (layer_param.include_size() == 0);
    //将符合exclude条件的layer去掉
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) { // layer_param.exclude(i) 是NetStateRule 类型
        layer_included = false;
      }
    }
    // 将符合include的layer加上
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    // 如果该层需要被包含
    // 则将其添加到filter
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  // 默认使用定义的描述名称,否则自动添加名称
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  // 直接替换
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    // top_vecs_ type: vector<vector<Blob<Dtype>*> >
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get()); 
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    // blobs重名,出错
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) { // 层输出 blob
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);  // 某一层layer  
  const string& blob_name = layer_param.bottom(bottom_id); // 某一个bottom 的name 
  if (available_blobs->find(blob_name) == available_blobs->end()) { // 最开始 将所有blob 都初始化进入available_blobs_
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name]; // 获取对应名字的blob 存入blob_id
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name; // 层的输入 指向 层
  // bottom_vecs_ type:vector<vector<Blob<Dtype>*> > 
  // blobs type: vector<shared_ptr<Blob<Dtype> > > , 
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());  // 共享指针 shared_ptr get函数 返回保存的指针
  
  bottom_id_vecs_[layer_id].push_back(blob_id); // vector<vector<int> >
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward); // 该层 反穿哪些bottom需要反传
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  // 新的参数,前面尚未出现过!!
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    // Add by ZhangM
    // if (param_id == 0) {
    //   learnable_params_satvalues_.push_back(layer_param.weight_satvalue());
    // } else {
    //   learnable_params_satvalues_.push_back(Dtype(-1));
    // }

    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    // 前面已经出现过的参数
    // 获取索引
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    // used for acceleration
    Timer timer;
    timer.Start();
    Dtype temp = timer.MicroSeconds();
    if (temp > 10) {}
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

/**
 * 将网络中的权值参数转换为NHWC模型．
 * 将网络中的卷积层/反卷积层转换为NHWC模型
 * 将名称包含"conv_fc"的全连接(IP)层转换为NHWC模型
 * 注意：网络中bottom输入为卷积层输出Blob的全连接层，其layer_name中务必包含"conv_fc"字段，否则模型转换将会失败．
 * 输出：将处理完的网络权值保存为指定的输出文件．
 */
template <typename Dtype>
void Net<Dtype>::ToNHWC(const std::string& model_name) {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->ToNHWC();
    LOG(INFO) << "ToNHWC for layer: " << layer_names()[i];
  }
  // save
  LOG(INFO) << "Preparing to save to " << model_name + ".caffemodel ...";
  NetParameter net_param;
  ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, model_name + ".caffemodel");
  LOG(INFO) << "Done.";
}

/**
 * 共享其他网络中的层及参数
 * 1. 遍历src网络的所有层, 查找本网络中是否有对应的同名层,如果找到,则继续,否则,继续遍历src网络的下一层
 * 2. 找到同名层,则检查参数的blobs数是否等同,如果等同,继续,否则报错;
 * 3. 遍历目标网络中该层的所有参数blobs,检查其shape与src中的对应参数blob是否等同,否则报错;
 * 4. 完成对应blob的数据共享
 */
template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  // 查找所有的目标层
  int num_source_layers = other->layers().size();
  // 遍历这些layer
  for (int i = 0; i < num_source_layers; ++i) {
    // 获取source-layer
    Layer<Dtype>* source_layer = other->layers()[i].get();
    // 获取名称
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    // 查看本网络中哪一层的名称与之相同
    // 1. 层号越界,表示本网络中不含有该层
    // 2. 匹配到同名层
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    // 如果是越界,则忽略之,结束本次循环
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    // 否则:复制本层
    LOG(INFO) << "Copying source layer " << source_layer_name;
    // 获取本网络中对应层的参数
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    // 检查参数的大小size是否等同:必须具有相同的参数blobs数量
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    //遍历所有的参数
    for (int j = 0; j < target_blobs.size(); ++j) {
      // 获取source的参数blob指针
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      // 注意:两者必须具有相同的尺寸
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      // 完成参数共享
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    // 遍历所有可学习参数
    // 累计其data和diff的asum和sumsq值
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    // 统计L2norm值:data和diff
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    // 打印
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  // 层数
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    // 获取参数
    const LayerParameter& source_layer = param.layer(i);
    // 名称
    const string& source_layer_name = source_layer.name();
    // 查找目标层
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    // 获得目标层参数Blobs
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    // 遍历所有的参数Blob
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        // 参数Blob尺寸不匹配
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      // 复制
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
    // Add by ZhangM
    // learnable_params_[i]->Update(learnable_params_satvalues_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
