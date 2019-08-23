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

#include <Eigen/Dense>

using namespace std;
using namespace caffe;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

template <typename Dtype>
void FilterPad(const Dtype* filter, Dtype* padded_filter, const int out_channels, const int filter_spatial_dim) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  for (int i = 0; i < filter_spatial_dim; ++i) {
    const int input_base = i * out_channels;
    const int output_base = i * out_channels;
    for (int j = 0; j < out_channels; j+=kPacketSize) {
      const Packet v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
      Eigen::internal::pstoreu<Dtype>(padded_filter + output_base + j, v);
    }
  }
}

template <typename Dtype>
void InputCopy(const Dtype* input, Dtype* input_buffer, const int channels,
  const int out_r, const int out_c, const int stride, const int pad,
  const int kernel, const int bwidth, const int bheight) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  // float * in_buf = input_buffer;
  // 行列起点
  const int in_r_start = out_r * stride - pad;
  const int in_c_start = out_c * stride - pad;
  // 遍历所有kernel
  for (int fr = 0; fr < kernel; ++fr) {
    const int in_r = in_r_start + fr;
    for (int fc = 0; fc < kernel; ++fc) {
      const int in_c = in_c_start + fc;
      if (in_r >= 0 && in_r < bheight && in_c >= 0 && in_c < bwidth) {
        const Dtype* in = input + (in_r * bwidth + in_c) * channels;
        for (int d = 0; d < channels; d+=kPacketSize) {
          Packet v = Eigen::internal::ploadu<Packet>(in + d);
          Eigen::internal::pstoreu<Dtype>(input_buffer + d, v);
        }
      } else {

      }
      input_buffer += channels;
    }
  }
}

template <typename Dtype>
void DepthWiseConv2DKernel(const Dtype* filter, const Dtype* input_buffer, Dtype* output,
                           const int channels, const int out_r, const int out_c,
                           const int top_height, const int top_width, const int kernel_size) {
  typedef typename Eigen::internal::packet_traits<Dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(Dtype);
  const int output_base_index = (out_r * top_width + out_c) * channels;
  for (int i = 0; i < channels; i+=kPacketSize) {
    Packet vaccum = Eigen::internal::pset1<Packet>(0);
    for (int j = 0; j < kernel_size; ++j) {
      const int index = i + j * channels;
      const Packet filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      const Packet data_block = Eigen::internal::ploadu<Packet>(input_buffer + index);
      vaccum = Eigen::internal::pmadd<Packet>(filter_block,data_block,vaccum);
    }
    Eigen::internal::pstoreu<Dtype>(output + output_base_index + i, vaccum);
  }
}

template <typename Dtype>
void DepthWiseConv2D(const Dtype* filter, const Dtype* input, Dtype* output,
                     const int num, const int channels, const int height, const int width,
                     const int stride, const int pad, const int kernel, Blob<Dtype>* buf) {
  for (int n = 0; n < num; ++n) {
    int in_base = n * height * width * channels;
    int out_base = n * height * width * channels;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // copy input buffer
        InputCopy(input + in_base, buf->mutable_cpu_data(), channels, h, w, stride, pad, kernel, width, height);
        // dw conv
        DepthWiseConv2DKernel(filter, buf->mutable_cpu_data(), output + out_base, channels, h, w, height, width, kernel * kernel);
      }
    }
  }
}

int main(int nargc, char** args) {

  const int input_height = 48;
  const int input_width = 48;
  const int input_channels = 128;

  const int kernel_size = 3;
  const int pad = 1;
  const int stride = 1;

  typedef float  dtype;

  typedef typename Eigen::internal::packet_traits<dtype>::type Packet;
  const int kPacketSize = sizeof(Packet) / sizeof(dtype);
  LOG(INFO) << "kPacketSize: " << kPacketSize;
  // input
  Blob<dtype> input(1,input_height,input_width,input_channels);
  caffe_set(input.count(),dtype(1),input.mutable_cpu_data());

  // weights
  Blob<dtype> weight(kernel_size,kernel_size,1,input_channels);
  caffe_set(weight.count(),dtype(0),weight.mutable_cpu_data());

  Blob<dtype> output(1,input_height,input_width,input_channels);
  caffe_set(output.count(),dtype(0),output.mutable_cpu_data());

  // load
  const dtype* data = input.cpu_data();
  const dtype* w = weight.cpu_data();
  dtype* result = output.mutable_cpu_data();

  const int num = input.shape(0);
  const int channels = input.shape(3);
  const int top_height = input.shape(1);
  const int top_width = input.shape(2);

  Blob<dtype>* buf = new Blob<dtype>(kernel_size,kernel_size,1,input_channels);

  caffe::Timer loader;
  loader.Start();
  for (int i = 0; i < 100; ++i) {
    DepthWiseConv2D<dtype>(w, data, result, num, channels, top_height, top_width, stride, pad, kernel_size, buf);
  }

  float use_time = loader.MicroSeconds();
  LOG(INFO) << "Use Time: " << use_time;
  LOG(INFO) << "Speed: " << use_time / 100 << " us/once.";
  return 0;
}
