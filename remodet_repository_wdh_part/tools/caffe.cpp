#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <string>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"


using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using std::map;
using std::pair;

// 11.7 wdh 
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(layertest, "no",
    "The speed test control bit.");
DEFINE_string(ap_version, "Integral",
    "The method for ap calculation.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)(); // 函数指针, B是指针, (*B) 是个函数, 返回值是 int
typedef std::map<caffe::string, BrewFunction> BrewMap; 
BrewMap g_brew_map; // map<str, 函数指针>

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

/* RegisterBrewFunction(train)
  namespace{
    class __Registerer_train{
      public:
        __Registerer_train()  // func: 类的初始化函数 注册器中注册 func, 通过g_brew_map[func] 进行调用
          {g_brew_map["train"] = &train;}  // g_brew_map type: BrewMap, 
    };
    __Registerer_train g_registerer_train;
  }
*/

static BrewFunction GetBrewFunction(const caffe::string& name) { //func: 返回 对应name的 brewFunction
  if (g_brew_map.count(name)) { // type: std::map<caffe::string, BrewFunction>  返回name 的个数
    return g_brew_map[name]; // 
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
// model_list: caffemodel模型文件中加载参数,用于finetune
// 加载同名layer
// 如果不希望加载,应定义不同的layer-name
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  // 获取model_list
  boost::split(model_names, model_list, boost::is_any_of(",") );
  // 遍历所有提供的model
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    // 从caffemodel中加载数据
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    // 将所有的test网络都进行加载
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  // 指定训练用的solver.prototxt文件
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  // 快照和权值文件不能同时提供!!!
  // --snapshot:从该快照中恢复训练, resume训练
  // --weights:从该网络中加载同名层权值进行训练, finetune训练
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  // 获取stages
  vector<string> stages = get_stages_from_flags(); 

  // 定义解析器,并从solver文件中获取
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param); // func: 读取solver param

  // 设置level和stages
  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  // 如果定义了快照,则恢复训练
  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    // 从快照中恢复
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    // 从caffemodel中获取加载参数
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() { // func:
  // 定义model文件:使用proto描述网络
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  // 定义权值文件:使用caffemodel描述训练好的网络
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // 获取stages
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  // 使用由proto描述的网络,phase为TEST
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  // 从权值文件中复制网络参数
  // .caffemodel文件实际上也是一个网络
  // .proto描述的网络文件会从.caffemodel文件中加载同名的layer,只要两者的layer名称
  // 匹配,则从.caffemodel中复制相应的layer参数
  // 如果不希望从.caffemodel中复制参数
  // 需要将proto对应的layer的名称进行修改
  // 注意:只要是layer名称相同,就会复制!!!
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  // 迭代
  for (int i = 0; i < FLAGS_iterations; ++i) {
    // 定义本次损失
    float iter_loss;
    // 获取输出Blobs以及损失
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    // 损失累加
    loss += iter_loss;
    //
    int idx = 0;
    // 遍历每个输出blob
    for (int j = 0; j < result.size(); ++j) {
      // 获取输出数据
      const float* result_vec = result[j]->cpu_data();
      // 遍历输出结果
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        // 获取输出结果
        const float score = result_vec[k];
        // test_score: 每个输出结果的多次迭代累加
        // test_score_output_id: 对应输出结果的输出编号j
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        // 获取第j个输出blob的name
        // const std::string& output_name = caffe_net.blob_names()[
        //     caffe_net.output_blob_indices()[j]];
        // 打印信息
        // Batch编号,输出blob名称,最后一个score结果: 往往是全局的评分
        // LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }

  // 求取平均损失
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;

  // 遍历输出blob的所有点
  for (int i = 0; i < test_score.size(); ++i) {
    // 获取名称
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    // 获取损失权重
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

// speed: test the fps of the network
int speed(){
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to speed.";
  // caffe::Phase phase = get_phase_from_flags(caffe::TEST);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  if (FLAGS_weights.size()) {
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  }
  // Do a clean forward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  // caffe_net.Forward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();

  std::vector<double> forward_time_per_layer(layers.size(), 0.0);

  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";

  Timer total_timer;
  total_timer.Start();
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    float loss;
    if (FLAGS_layertest == "yes") {
      for (int i = 0; i < layers.size(); ++i) {
        Timer timer;
        timer.Start();
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
        forward_time_per_layer[i] += timer.MicroSeconds();
      }
    } else{
      caffe_net.Forward(&loss);
    }
    LOG(INFO) << "Iteration: " << j + 1 << " forward time: "
      << iter_timer.MicroSeconds() << " us.";
  }
  if (FLAGS_layertest == "yes") {
    // print each layer time
    for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
        "\tforward: " << forward_time_per_layer[i] / FLAGS_iterations << " us.";
    }
  }
  total_timer.Stop();
  int avg_iter_us = total_timer.MicroSeconds() / FLAGS_iterations;
  LOG(INFO) << "Average Forward pass: " << avg_iter_us << " us.";
  float fps = 1000000.0 / avg_iter_us;
  LOG(INFO) << "Average Running Speed is: " << std::setiosflags(std::ios::fixed)
            << std::setprecision(2) << fps << " FPS.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(speed);

int score() {
  // 定义model文件:使用proto描述网络
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  // 定义权值文件:使用caffemodel描述训练好的网络
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // 获取stages
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  // 使用由proto描述的网络,phase为TEST
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  // 从权值文件中复制网络参数
  // .caffemodel文件实际上也是一个网络
  // .proto描述的网络文件会从.caffemodel文件中加载同名的layer,只要两者的layer名称
  // 匹配,则从.caffemodel中复制相应的layer参数
  // 如果不希望从.caffemodel中复制参数
  // 需要将proto对应的layer的名称进行修改
  // 注意:只要是layer名称相同,就会复制!!!
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  std::map<int, map<int, map<int, vector<pair<float, int> > > > > all_true_pos;
  std::map<int, map<int, map<int, vector<pair<float, int> > > > > all_false_pos;
  std::map<int, map<int, map<int, int> > > all_num_pos;

  float loss = 0;

  for (int i = 0; i < FLAGS_iterations; ++i) {
    const vector<Blob<float>*>& result = caffe_net.Forward(&loss);
    CHECK_LT(result.size(), 3) << "To much output blobs output.";

    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      if (result[out_idx]->count() > 3) {
        // accuracy
        CHECK(result[out_idx]);
        CHECK_EQ(result[out_idx]->width(),7);
        int num_det = result[out_idx]->height();
        const float* top_data = result[out_idx]->cpu_data();
        for (int i = 0; i < num_det; ++i) {
          int diff = static_cast<int>(top_data[i * 7]);
          int level = static_cast<int>(top_data[i * 7 + 1]);
          CHECK_LT(diff,3);
          CHECK_GE(diff,0);
          CHECK_LT(level,7);
          CHECK_GE(level,0);
          int item = static_cast<int>(top_data[i * 7 + 2]);
          int label = static_cast<int>(top_data[i * 7 + 3]);
          if (item < 0) {
            // stat gts
            if (all_num_pos[diff][level].find(label) == all_num_pos[diff][level].end()){
              all_num_pos[diff][level][label] = static_cast<int>(top_data[i * 7 + 4]);
            } else {
              all_num_pos[diff][level][label] += static_cast<int>(top_data[i * 7 + 4]);
            }
          } else {
            // stat tps/fps
            float score = top_data[i * 7 + 4];
            int tp = static_cast<int>(top_data[i * 7 + 5]);
            int fp = static_cast<int>(top_data[i * 7 + 6]);
            if (tp == 0 && fp == 0) continue;
            all_true_pos[diff][level][label].push_back(std::make_pair(score, tp));
            all_false_pos[diff][level][label].push_back(std::make_pair(score, fp));
          }
        }
      }
    }
  }
  std::map<int, map<int, map<int, float> > > APs;
  std::map<int, map<int, float> > mAPs;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      float mAP = 0;
      const std::map<int, vector<pair<float, int> > >& true_pos = all_true_pos[i][j];
      const std::map<int, vector<pair<float, int> > >& false_pos = all_false_pos[i][j];
      const std::map<int, int>& num_pos = all_num_pos[i][j];
      for (std::map<int, int>::const_iterator it = num_pos.begin();
           it != num_pos.end(); ++it) {
        int label = it->first;
        int label_num_pos = it->second;

        if (true_pos.find(label) == true_pos.end()) {
          LOG(WARNING) << "Missing true_pos for label: " << label;
          continue;
        }
        const vector<pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;

        if (false_pos.find(label) == false_pos.end()) {
          LOG(WARNING) << "Missing false_pos for label: " << label;
          continue;
        }
        const vector<pair<float, int> >& label_false_pos =
            false_pos.find(label)->second;

        vector<float> prec, rec;

        caffe::ComputeAP(label_true_pos, label_num_pos, label_false_pos,
          FLAGS_ap_version, &prec, &rec, &(APs[i][j][label]));
        mAP += APs[i][j][label];
      }
      mAP /= num_pos.size();
      mAPs[i][j] = mAP;
    }
  }
  // display names
  string names[1]  = {"person"};
  string levels[7] = {"0.00","0.01","0.05","0.10","0.15","0.20","0.25"};
  string diffs[3]  = {"0.90","0.75","0.50"};
  // output Class APs INFO
  // LOG(INFO) << "[Test All] After Train Iteration " << iter_ << " : ";
  // LOG(INFO) << "[Test Category Accuracy (APs)]: ";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      map<int,float>& class_APs = APs[i][j];
      for(map<int,float>::const_iterator it = class_APs.begin();
          it != class_APs.end(); ++it) {
        int label = it->first;
        if(label != 1) continue;
        string type = "AP";
        ostringstream keystream;
        keystream << "IOU@" << diffs[i]
                  << "/SIZE@" << levels[j]
                  << "/CAT@" << names[label-1];
        string key = keystream.str();
        LOG(INFO) << "${"
                  << "\"Type\": " << "\"" << type << "\", "
                  // << "\"Iteration\": " << "\"" << iter_ << "\", "
                  << "\"Key\": " << "\"" << key << "\", "
                  << "\"Value\": " << "\"" << it->second << "\""
                  << "}";
      }
    }
  }
  // output mAP INFO
  // LOG(INFO) << "[Test All Accuracy (mAP)]: ";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      float map = mAPs[i][j];
      string type = "mAP";
      ostringstream keystream;
      keystream << "IOU@" << diffs[i]
                << "/SIZE@" << levels[j];
      string key = keystream.str();
      LOG(INFO) << "${"
                << "\"Type\": " << "\"" << type << "\", "
                // << "\"Iteration\": " << "\"" << iter_ << "\", "
                << "\"Key\": " << "\"" << key << "\", "
                << "\"Value\": " << "\"" << map << "\""
                << "}";
    }
  }
  return 0;
}
RegisterBrewFunction(score);

int main(int argc, char** argv) {  // func: 
  // Print output to stderr (while still logging). 
  FLAGS_alsologtostderr = 1; // func: log打印到屏幕
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION)); 
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time\n"
      "  speed           benchmark running time and fps\n"
      "  score           get the testing AP and mAP of a model");
  // Run tool or show usage. // argc 参数个数, argv[0] 是程序运行路径, argv[]是参数, 
  caffe::GlobalInit(&argc, &argv);  // func: 
  if (argc == 2) { // func: 如果有2个参数, 第一个参数是程序本身路径
#ifdef WITH_PYTHON_LAYER // func: 如果宏已经定义，则编译下面代码
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))(); // func: GetBrewFunction 函数返回 一个函数, 又进行调用
#ifdef WITH_PYTHON_LAYER // func: 如果定义 使用python layer层, 
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
