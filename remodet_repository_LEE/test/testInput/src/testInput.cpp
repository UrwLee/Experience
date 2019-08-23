#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include <boost/shared_ptr.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/roi_align_layer.hpp"
#include "caffe/filler.hpp"

#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using namespace std;
using namespace caffe;

int main(int nargc, char** args) {
  while(1) {
    std::cout << "type ENTER to save: ";
    // std::string line;
    // cin >> line;
    char line;
    line = cin.get();
    if (line == '\n') {
      std::cout << "YES" << '\n';
    } else {
      std::cout << "NO" << "\n\n";
    }
    while(cin.get() != '\n');
  }
  return 0;
}
