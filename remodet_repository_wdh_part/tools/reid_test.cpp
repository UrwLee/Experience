#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

#include <boost/shared_ptr.hpp>

/**
 * 该程序是为了测试Re-ID任务的性能。
 * 目前不建议使用该程序。
 */

using namespace caffe;

int main(int argc, char** argv) {
  // GPU id
  const int gpu_id = 0;
  // quiry net
  const std::string quiry_proto = "/home/zhangming/Models/Results/Reid_Train/RemNet_R3_0/Proto/test_quiry.prototxt";
  const std::string quiry_model = "/home/zhangming/Models/Results/Reid_Train/RemNet_R3_0/Models/RemNet_R3_0_iter_10000.caffemodel";
  const int quiry_num = 2900;
  boost::shared_ptr<caffe::Net<float> > quiry_net;
  // test net
  const std::string test_proto = "/home/zhangming/Models/Results/Reid_Train/RemNet_R3_0/Proto/test_test.prototxt";
  const int test_num = 6978;
  boost::shared_ptr<caffe::Net<float> > test_net;

  // quiry Models
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  quiry_net.reset(new Net<float>(quiry_proto, caffe::TEST));
  quiry_net->CopyTrainedLayersFrom(quiry_model);
  // Forward pass for 2900
  for (int i = 0; i < quiry_num; ++i) {
    quiry_net->Forward();
  }
  LOG(INFO) << "Quiry features updated.";

  // Test Models: copy from QUIRY_NET
  test_net.reset(new Net<float>(test_proto, caffe::TEST));
  (test_net.get())->ShareTrainedLayersWith(quiry_net.get());

  // now we start test
  int tps = 0;
  int count = 0;
  for (int i = 0; i < test_num; ++i) {
    const vector<Blob<float>*>& result = test_net->Forward();
    for (int j = 0; j < result.size(); ++j) {
      if (result[j]->count() != 2) continue;
      const float* result_vec = result[j]->cpu_data();
      tps += result_vec[0];
      count += result_vec[1];
      LOG(INFO) << "Test index: " << i
                << ", TP/GT -> " << (int)result_vec[0]
                << "/" << (int)result_vec[1];
    }
  }
  LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(3)
            << "[Test] TP/ALL -> " << tps << "/" << count
            << ", Accuracy: " << (float)tps/count;
  LOG(INFO) << "Test Done.";
}
