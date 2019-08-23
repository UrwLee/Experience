#include <cstdio>
#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/tracker/tracker_tester.hpp"
#include "caffe/tracker/tracker_video.hpp"
#include "caffe/tracker/video_loader.hpp"
#include "caffe/tracker/vot_loader.hpp"
#include "caffe/tracker/alov_loader.hpp"
#include "caffe/tracker/regressor.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;

  // 统计输出count()为1的blobs数量
  // blob名称中包含"loss/Loss/LOSS"被认为是损失.
  // 损失vector: 用于smoothed
  vector<vector<Dtype> > multi_losses_array;
  // smoothed losses
  vector<Dtype> multi_smoothed_losses;
  // 名称
  vector<string> multi_losses_names;
  int num_out_losses = 0;
  const vector<Blob<Dtype>*>& out_blobs = net_->output_blobs();
  for (int j = 0; j < out_blobs.size(); ++j) {
    if (out_blobs[j]->count() == 1) {
      std::string blob_name = net_->blob_names()[net_->output_blob_indices()[j]];
      if ((blob_name.find("loss") != string::npos) ||
          (blob_name.find("Loss") != string::npos) ||
          (blob_name.find("LOSS") != string::npos)) {
        ++num_out_losses;
        multi_smoothed_losses.push_back(Dtype(0));
        multi_losses_names.push_back(blob_name);
      }
    }
  }
  multi_losses_array.resize(num_out_losses);
  //　下面开始进行迭代
  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    vector<Dtype> multi_losses(num_out_losses, Dtype(0));
    for (int i = 0; i < param_.iter_size(); ++i) {
      Dtype iter_loss;
      const vector<Blob<Dtype>*>& results = net_->ForwardBackward(&iter_loss);
      loss += iter_loss;
      int temp = 0;
      for (int r = 0; r < results.size(); ++r) {
        Blob<Dtype>* res = results[r];
        if (res->count() == 1) {
          std::string blob_name = net_->blob_names()[net_->output_blob_indices()[r]];
          if ((blob_name.find("loss") != string::npos) ||
              (blob_name.find("Loss") != string::npos) ||
              (blob_name.find("LOSS") != string::npos)) {
            multi_losses[temp++] += res->cpu_data()[0];
          }
        }
      }
    }
    // 损失平均
    loss /= param_.iter_size();
    for (int p = 0; p < multi_losses.size(); ++p) {
      multi_losses[p] /= param_.iter_size();
    }
    // 对所有损失进行光滑处理
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    for (int p = 0; p < multi_smoothed_losses.size(); ++p) {
      UpdateSmoothedLoss(multi_losses[p],&multi_losses_array[p],&multi_smoothed_losses[p],start_iter,average_loss);
    }
    if (display) {
      // train_smoothed_loss
      LOG_IF(INFO, Caffe::root_solver()) << "${"
                << "\"Type\": " << "\"" << "Loss" << "\", "
                << "\"Iteration\": " << "\"" << iter_ << "\", "
                << "\"Key\": " << "\"" << "train_smoothed_loss" << "\", "
                << "\"Value\": " << "\"" << smoothed_loss_ << "\""
                << "}";
      // train_loss
      LOG_IF(INFO, Caffe::root_solver()) << "${"
                << "\"Type\": " << "\"" << "Loss" << "\", "
                << "\"Iteration\": " << "\"" << iter_ << "\", "
                << "\"Key\": " << "\"" << "train_loss" << "\", "
                << "\"Value\": " << "\"" << loss << "\""
                << "}";
      for (int p = 0; p < multi_smoothed_losses.size(); ++p) {
        LOG_IF(INFO, Caffe::root_solver()) << "${"
                  << "\"Type\": " << "\"" << "Loss" << "\", "
                  << "\"Iteration\": " << "\"" << iter_ << "\", "
                  << "\"Key\": " << "\"" << multi_losses_names[p] << "\", "
                  << "\"Value\": " << "\"" << multi_smoothed_losses[p] << "\""
                  << "}";

       const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      /*or (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }*/

      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    if (param_.test_net_type(test_net_id) == "classification") {
      TestClassification(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "detection") {
      TestDetection(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "pose") {
      TestPose(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "tracker") {
      TestTracker(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "reid") {
      TestReid(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "mask") {
      TestMask(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "pose_ins") {
      TestPoseIns(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "handpose"){
      TestHandPose(test_net_id);
    } else if (param_.test_net_type(test_net_id) == "none"){
      continue;
    }else {
      LOG(FATAL) << "Unknown evaluation type: " << param_.eval_type();
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::TestReid(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  int tps = 0;
  int count = 0;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  // cal. (iters)
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      break;
    }
    // Forward pass
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    // get result
    for (int j = 0; j < result.size(); ++j) {
      if (result[j]->count() != 2) continue;
      const Dtype* result_vec = result[j]->cpu_data();
      tps += result_vec[0];
      count += result_vec[1];
    }
  }
  // post process
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  // loss
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  // print info.
  LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(3)
            << "[Test] TP/ALL -> " << tps << "/" << count
            << ", Accuracy: " << (Dtype)tps/count;
  LOG(INFO) << "Test Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestClassification(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::TestTracker(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  // 获得训练好的网络权值
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  // 获取数据集
  const int num_video_loaders = param_.tracker_test_vottype_dir_size()
                            + param_.tracker_test_alovtype_image_dir_size();
  CHECK_GT(num_video_loaders,0);
  CHECK_EQ(param_.tracker_test_alovtype_image_dir_size(),
           param_.tracker_test_alovtype_annos_dir_size());
  shared_ptr<VideoLoader<Dtype> > video_loader;
  // 加载视频
  if (param_.tracker_test_vottype_dir_size() > 0) {
    for (int i = 0; i < param_.tracker_test_vottype_dir_size(); ++i) {
      if (i == 0) {
        video_loader.reset(new VOTLoader<Dtype>(param_.tracker_test_vottype_dir(i)));
      } else {
        VOTLoader<Dtype> vot_loader(param_.tracker_test_vottype_dir(i));
        video_loader->merge_from(&vot_loader);
      }
    }
  }
  if (param_.tracker_test_vottype_dir_size() > 0) {
    for (int i = 0; i < param_.tracker_test_alovtype_image_dir_size(); ++i) {
      ALOVLoader<Dtype> alov_loader(param_.tracker_test_alovtype_image_dir(i),
                                    param_.tracker_test_alovtype_annos_dir(i));
      video_loader->merge_from(&alov_loader);
    }
  } else {
    video_loader.reset(new ALOVLoader<Dtype>(param_.tracker_test_alovtype_image_dir(0),
                                             param_.tracker_test_alovtype_annos_dir(0)));
    for (int i = 1; i < param_.tracker_test_alovtype_image_dir_size(); ++i) {
      ALOVLoader<Dtype> alov_loader(param_.tracker_test_alovtype_image_dir(i),
                                    param_.tracker_test_alovtype_annos_dir(i));
      video_loader->merge_from(&alov_loader);
    }
  }
  // 视频
  const std::vector<Video<Dtype> >& videos = video_loader->get_videos();
  // 回归器
  Regressor<Dtype> regressor(test_nets_[test_net_id], param_.device_id());
  // Tracker
  TrackerBase<Dtype> tracker(false);
  //TrackerVideo
  if(param_.test_use_camera())
  {
    VideoTracker<Dtype> video_tracker(param_.video_tracker_parameter(),
                                   &regressor,
                                   &tracker);
    video_tracker.Tracking();
  }

  // TrackerTester
  TrackerTester<Dtype> tracker_tester(videos,&regressor,&tracker,param_.show_tracking(),param_.save_tracking(),
                                      param_.tracker_test_out_folder(), param_.tracker_test_save_folder());
  // TrackAll()
  tracker_tester.TrackAll();
}

template <typename Dtype>
void Solver<Dtype>::TestDetection(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  // [diff][level][label]<pair>
  map<int, map<int, map<int, vector<pair<float, int> > > > > all_true_pos;
  map<int, map<int, map<int, vector<pair<float, int> > > > > all_false_pos;
  // [diff][level][label](num_gts)
  map<int, map<int, map<int, int> > > all_num_pos;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  int check_TP_id = 1; //1--for hand
  int check_TP_level = 2;
  int check_TP_level_1 = 3;
  int num_TP_id = 0;
  int num_TP_id_1 = 0;

  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      std::string blob_name = test_net->blob_names()[test_net->output_blob_indices()[out_idx]];
      if ((blob_name.find("accu") != string::npos) ||
          (blob_name.find("Accu") != string::npos) ||
          (blob_name.find("ACCU") != string::npos)) {
        int num_det = result[out_idx]->height();
        const Dtype* top_data = result[out_idx]->cpu_data();
        for (int i = 0; i < num_det; ++i) {
          int diff = static_cast<int>(top_data[i * 7]);
          int level = static_cast<int>(top_data[i * 7 + 1]);
          int cid = static_cast<int>(top_data[i * 7 + 3]);
          int tp_flag1 = static_cast<int>(top_data[i * 7 + 5]);
          int tp_flag2 = static_cast<int>(top_data[i * 7 + 6]);
          if(cid == check_TP_id && tp_flag1 == 1 && tp_flag2 == 0 && level <= check_TP_level){
            num_TP_id ++;
          }
          if(cid == check_TP_id && tp_flag1 == 1 && tp_flag2 == 0 && level <= check_TP_level_1){
            num_TP_id_1 ++;
          }
          CHECK_LT(diff,3);
          CHECK_GE(diff,0);
          CHECK_LT(level,7);
          CHECK_GE(level,0);
          int item = static_cast<int>(top_data[i * 7 + 2]);
          int label = static_cast<int>(top_data[i * 7 + 3]);

          // LOG(INFO)<<"hzw label"<<label;
          if (item < 0) {
            // stat gts
            // LOG(INFO)<<"hzw item<0 label:"<<label;
            if (all_num_pos[diff][level].find(label) == all_num_pos[diff][level].end()) {
              all_num_pos[diff][level][label] = static_cast<int>(top_data[i * 7 + 4]);
            } else {
              all_num_pos[diff][level][label] += static_cast<int>(top_data[i * 7 + 4]);
            }
          } else {
            // stat tps/fps
            // LOG(INFO)<<"hzw item>0 label:"<<label;
            float score = top_data[i * 7 + 4];
            int tp = static_cast<int>(top_data[i * 7 + 5]);
            // LOG(INFO)<<"hzw tp:"<<tp;
            int fp = static_cast<int>(top_data[i * 7 + 6]);
            // LOG(INFO)<<"hzw fp:"<<fp;
            if (tp == 0 && fp == 0) continue;
            // LOG(INFO)<<"hzw tp:"<<tp<<"fp:"<<fp<<"label";
            // LOG(INFO)<<"hzw score:"<<score<<"tp:"<<tp;
            all_true_pos[diff][level][label].push_back(std::make_pair(score, tp));
            all_false_pos[diff][level][label].push_back(std::make_pair(score, fp));
          }
        }
      }
    }
  }
  // ALL test iterations has been finished
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  // cal ap
  map<int, map<int, map<int, float> > > APs;
  map<int, map<int, float> > mAPs;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      float mAP = 0;
      const map<int, vector<pair<float, int> > >& true_pos = all_true_pos[i][j];
      const map<int, vector<pair<float, int> > >& false_pos = all_false_pos[i][j];
      const map<int, int>& num_pos = all_num_pos[i][j];
      for (map<int, int>::const_iterator it = num_pos.begin();
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
        // LOG(INFO)<<"hzw label"<<label;
        ComputeAP(label_true_pos, label_num_pos, label_false_pos,
          param_.ap_version(), &prec, &rec, &(APs[i][j][label]));
        mAP += APs[i][j][label];
      }
      mAP /= num_pos.size();
      mAPs[i][j] = mAP;
    }
  }
  // display names
  string names[4]  = {"person","hand","head","face"};
  string levels[7] = {"ALL","S","M","M+","L","L+","L++"};
  string diffs[3]  = {"0.90","0.75","0.50"};
  // output Class APs INFO
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      map<int,float>& class_APs = APs[i][j];
      for(map<int,float>::const_iterator it = class_APs.begin();
          it != class_APs.end(); ++it) {
        int label = it->first;

        string type = "AP";
        const int start_idx = 0;
        ostringstream keystream;
        keystream << "IOU@" << diffs[i]
                  << "/SIZE@" << levels[start_idx + j]
                  << "/CAT@" << names[label];
        string key = keystream.str();
        LOG(INFO) << "${"
                  << "\"Type\": " << "\"" << type << "\", "
                  << "\"Iteration\": " << "\"" << iter_ << "\", "
                  << "\"Key\": " << "\"" << key << "\", "
                  << "\"Value\": " << "\"" << it->second << "\""
                  << "}";
      }
    }
  }
  // output mAP INFO
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
                << "\"Iteration\": " << "\"" << iter_ << "\", "
                << "\"Key\": " << "\"" << key << "\", "
                << "\"Value\": " << "\"" << map << "\""
                << "}";
    }
  }
  LOG(INFO)<<"check_TP_id: "<<check_TP_id<<"; num_TP_id: "<<num_TP_id<<"(level:"<<check_TP_level<<");num_TP_id_1: "<<num_TP_id_1<<"(level:"<<check_TP_level_1<<")";
}

template <typename Dtype>
void Solver<Dtype>::TestPose(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  int gts_all = 0;
  vector<int> tps_all, fps_all;
  vector<Dtype> oks_thre;
  int num_thre = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }

    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      // 只统计eval的输出
      if (result[out_idx]->width() >= 4) {
        int width = result[out_idx]->width();
        num_thre = (width - 1) / 3;
        const Dtype* res_data = result[out_idx]->cpu_data();
        // 统计
        if (i == 0) {
          tps_all.resize(num_thre);
          fps_all.resize(num_thre);
          oks_thre.resize(num_thre);
          gts_all += res_data[0];
          for(int j = 0; j < num_thre; ++j) {
            tps_all[j] = res_data[1 + 3*j];
            fps_all[j] = res_data[2 + 3*j];
            oks_thre[j] = res_data[3 + 3*j];
          }
        } else {
          gts_all += res_data[0];
          for(int j = 0; j < num_thre; ++j) {
            tps_all[j] += res_data[1 + 3*j];
            fps_all[j] += res_data[2 + 3*j];
            oks_thre[j] = res_data[3 + 3*j];
          }
        }
      }
    }
  }
  // ALL test iterations has been finished
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "${"
              << "\"Type\": " << "\"" << "Loss" << "\", "
              << "\"Iteration\": " << "\"" << iter_ << "\", "
              << "\"Key\": " << "\"" << "test_loss" << "\", "
              << "\"Value\": " << "\"" << loss << "\""
              << "}";
  }
  // cal
  LOG(INFO) << "[Test] Test Instances -> " << param_.test_iter(test_net_id);
  for (int i = 0; i < num_thre; ++i) {
    LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(3)
            << "[Test] OKS:" << oks_thre[i] << ", TP/GT:" << tps_all[i] << "/" << gts_all << ", Accuracy="
            << (Dtype)tps_all[i] / gts_all << ", TP/PR:" << tps_all[i] << "/" << tps_all[i]+fps_all[i] << ", Recall="
            << (Dtype)tps_all[i] / (tps_all[i]+fps_all[i]);
  }
  LOG(INFO) << "[Test] Evaluation Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestMask(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  Dtype accu = 0;
  int num_ins = 0;
  int tp_75 = 0;
  int tp_90 = 0;
  Dtype size_s = 0.01;
  Dtype size_m = 0.05;
  Dtype size_l = 0.1;
  int num_s = 0;
  int num_m = 0;
  int num_l = 0;
  Dtype accu_s = 0;
  Dtype accu_m = 0;
  Dtype accu_l = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      break;
    }
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      if (result[out_idx]->width() == 3) {
        const Dtype* res = result[out_idx]->cpu_data();
        for (int p = 0; p < result[out_idx]->height(); ++p) {
          int cid = res[3*p];
          if (cid < 0) continue;
          num_ins++;
          accu += res[3*p+2];
          if (res[3*p+2] > 0.75) tp_75++;
          if (res[3*p+2] > 0.90) tp_90++;
          if (res[3*p+1] > size_s) { num_s++; accu_s += res[3*p+2]; }
          if (res[3*p+1] > size_m) { num_m++; accu_m += res[3*p+2]; }
          if (res[3*p+1] > size_l) { num_l++; accu_l += res[3*p+2]; }
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
  }
  // Output Info.
  LOG(INFO) << "Found " << num_ins << " Instances.";
  LOG(INFO) << "MASK AP       : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu / num_ins;
  LOG(INFO) << "MASK AP(0.75) : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << Dtype(tp_75) / num_ins;
  LOG(INFO) << "MASK AP(0.90) : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << Dtype(tp_90) / num_ins;
  LOG(INFO) << "MASK AP(S)    : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_s / num_s;
  LOG(INFO) << "MASK AP(M)    : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_m / num_m;
  LOG(INFO) << "MASK AP(L)    : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_l / num_l;
  LOG(INFO) << "Mask Evaluation Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestPoseIns(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  Dtype size_m = 0.05;
  Dtype size_l = 0.1;
  // each
  Dtype accu[18];
  int num_ins = 0;
  int tp_ins[18];
  for (int i = 0; i < 18; ++i) {
    tp_ins[i] = 0;
  }
  // size_m
  Dtype accu_m[18];
  int num_ins_m = 0;
  int tp_ins_m[18];
  for (int i = 0; i < 18; ++i) {
    tp_ins_m[i] = 0;
  }
  // size_l
  Dtype accu_l[18];
  int num_ins_l = 0;
  int tp_ins_l[18];
  for (int i = 0; i < 18; ++i) {
    tp_ins_l[i] = 0;
  }
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      break;
    }
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      if (result[out_idx]->width() == 20) {
        const Dtype* res = result[out_idx]->cpu_data();
        for (int p = 0; p < result[out_idx]->height(); ++p) {
          int cid = res[20*p];
          if (cid < 0) continue;
          // stat ap
          num_ins++;
          for (int k = 0; k < 18; ++k) {
            tp_ins[k] += res[20*p+k+2];
          }
          // stat size_m
          if (res[20*p+1] > size_m) {
            num_ins_m++;
            for (int k = 0; k < 18; ++k) {
              tp_ins_m[k] += res[20*p+k+2];
            }
          }
          // stat size_l
          if (res[20*p+1] > size_l) {
            num_ins_l++;
            for (int k = 0; k < 18; ++k) {
              tp_ins_l[k] += res[20*p+k+2];
            }
          }
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
  }
  // Output Info.
  Dtype accu_all_sum = 0;
  Dtype accu_m_all_sum = 0;
  Dtype accu_l_all_sum = 0;
  for (int k = 0; k < 18; ++k) {
    accu[k] = Dtype(tp_ins[k]) / num_ins;
    accu_m[k] = Dtype(tp_ins_m[k]) / num_ins_m;
    accu_l[k] = Dtype(tp_ins_l[k]) / num_ins_l;
    accu_all_sum += accu[k];
    accu_m_all_sum += accu_m[k];
    accu_l_all_sum += accu_l[k];
  }
  // avg ap
  Dtype accu_all = accu_all_sum / 18;
  Dtype accu_m_all = accu_m_all_sum / 18;
  Dtype accu_l_all = accu_l_all_sum / 18;
  LOG(INFO) << "STAT Average POSE AP.";
  LOG(INFO) << "Found " << num_ins << " Instances.";
  for (int k = 0; k < 18; ++k) {
    LOG(INFO) << "[mAP]    Part " << k << " : AP = " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu[k];
  }
  LOG(INFO) << "[mAP]    mAP : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_all;
  // Sized AP
  LOG(INFO) << "STAT Medium-Size-Instances POSE AP.";
  LOG(INFO) << "STAT Instances for normalized-size > " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << size_m;
  LOG(INFO) << "Found " << num_ins_m << " Instances.";
  for (int k = 0; k < 18; ++k) {
    LOG(INFO) << "[Medium] Part " << k << " : AP = " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_m[k];
  }
  LOG(INFO) << "[Medium] mAP : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_m_all;
  // Sized AP
  LOG(INFO) << "STAT Large-Size-Instances POSE AP.";
  LOG(INFO) << "STAT Instances for normalized-size > " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << size_l;
  LOG(INFO) << "Found " << num_ins_l << " Instances.";
  for (int k = 0; k < 18; ++k) {
    LOG(INFO) << "[Large]  Part " << k << " : AP = " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_l[k];
  }
  LOG(INFO) << "[Large]  mAP : " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << accu_l_all;
  LOG(INFO) << "Pose Evaluation Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestHandPose(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  // [diff][level][label]<pair>
  vector<vector<float> > confusion_matrix;
  vector<float> counter;
  for(int ic = 0; ic <10;ic++){
      vector<float> tmpv(10,0);
      confusion_matrix.push_back(tmpv);
      counter.push_back(0);
  }

  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  Dtype acc = 0;


  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    // if (param_.test_compute_loss()) {
    //   loss += iter_loss;
    // }
    //LOG(INFO)<<"result.size() "<<result.size();
    //LOG(INFO)<<"confusion_matrix.size "<<confusion_matrix.size();
    for (int out_idx = 0; out_idx < result.size(); ++out_idx) {
      std::string blob_name = test_net->blob_names()[test_net->output_blob_indices()[out_idx]];
      if ((blob_name.find("accu") != string::npos) ||
          (blob_name.find("Accu") != string::npos) ||
          (blob_name.find("ACCU") != string::npos)) {
        int batchsize = result[out_idx]->num();
        //LOG(INFO)<<"batchsize "<<batchsize;
        const Dtype* top_data = result[out_idx]->cpu_data();
        for (int ib = 0; ib < batchsize; ++ib) {
          int gt_label = (int)top_data[ib*2];
          int pred_label = (int)top_data[ib*2 + 1];
          //LOG(INFO)<<gt_label<<" "<<pred_label;
          counter[gt_label] += 1;
          confusion_matrix[gt_label][pred_label] += 1;


        }
      } else if (blob_name.find("loss") != string::npos){
        const Dtype* top_data = result[out_idx]->cpu_data();
        loss += top_data[0];
      } else if (blob_name.find("acc") != string::npos){
        const Dtype* top_data = result[out_idx]->cpu_data();
        acc += top_data[0];
      }
    }
  }
  // ALL test iterations has been finished
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    acc /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss<<"; acc: "<<acc;
  }
  for(int ic=0;ic<10;ic++){
    float num = counter[ic];
    for(int jc=0;jc<10;jc++){
      confusion_matrix[ic][jc] /= num;
    }
  }
  // output mAP INFO
  for (int ic = 0; ic < 10; ++ic) {
    vector<float> conf=confusion_matrix[ic];
    char str_show[10][50];
    for(int jc=0;jc<10;jc++){
      sprintf(str_show[jc],"%4f ",conf[jc]);
    }
    /*LOG(INFO)<<str_show[0]<<conf[0]<<" "
             <<str_show[1]<<conf[1]<<" "
             <<str_show[2]<<conf[2]<<" "
             <<str_show[3]<<conf[3]<<" "
             <<str_show[4]<<conf[4]<<" "
             <<str_show[5]<<conf[5]<<" "
             <<str_show[6]<<conf[6]<<" "
             <<str_show[7]<<conf[7]<<" "
             <<str_show[8]<<conf[8]<<" "
             <<str_show[9]<<conf[9]<<" ";*/
    LOG(INFO)<<str_show[0]
             <<str_show[1]
             <<str_show[2]
             <<str_show[3]
             <<str_show[4]
             <<str_show[5]
             <<str_show[6]
             <<str_show[7]
             <<str_show[8]
             <<str_show[9];
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, vector<Dtype>* loss_array,
                    Dtype* smoothed_loss, int start_iter, int average_loss) {
  if (loss_array->size() < average_loss) {
    loss_array->push_back(loss);
    int size = loss_array->size();
    *smoothed_loss = ((*smoothed_loss) * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    *smoothed_loss += (loss - (*loss_array)[idx]) / average_loss;
    (*loss_array)[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
