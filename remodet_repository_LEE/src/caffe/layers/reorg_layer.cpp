#include <vector>

#include "caffe/layers/reorg_layer.hpp"

namespace caffe {

/**
 * bottom_data: 输入数据指针
 * top_data: 输出数据指针
 * forward: 下采样，从size大的一侧转换到size小的一侧，channels增大
 *     否则，上采样，从size小的一侧转换到size大的一侧，channels减小
 * stride: 采样间隔
 */
template <typename Dtype>
void Reorg(const Dtype* bottom_data, const bool forward,
          const vector<int> input_shape,
          const int stride, Dtype* top_data) {
  CHECK_EQ(input_shape.size(), 4);
  const int num = input_shape[0];
  const int channels = input_shape[1];
  const int height = input_shape[2];
  const int width = input_shape[3];
  int out_width = width / stride;
  int out_height = height / stride;
  int out_channels = channels * stride * stride;

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          int in_idx = ((n * channels + c) * height + i) * width + j;
          int out_w = j / stride;
          int out_h = i / stride;
          // 插入方法1
          // int out_c = ((i % stride) * stride + j % stride) * channels + c;
          // 插入方法2
          int out_c = (i % stride) * stride + j % stride + c * stride * stride;
          int out_idx = ((n * out_channels + out_c) * out_height
                                + out_h) * out_width + out_w;
          if (forward)
              top_data[out_idx] = bottom_data[in_idx];
          else
              top_data[in_idx] = bottom_data[out_idx];
        }
      }
    }
  }
}

template <typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const ReorgParameter& reorg_param = this->layer_param_.reorg_param();
  CHECK_EQ(bottom.size(), 1) << "Reorg Layer only support one input blob.";
  CHECK_EQ(bottom[0]->num_axes(), 4)
      << "Reorg Layer input blob must have 4 axis.";
  // up / down
  up_down_ = reorg_param.up_down();
  // sample gaps
  stride_ = reorg_param.stride();
  CHECK_GE(stride_,1);

  vector<int> top_shape(4,1);
  top_shape[0] = bottom[0]->shape(0);
  if (up_down_ == ReorgParameter_SampleType_DOWN) {
    CHECK_EQ(bottom[0]->shape(2) % stride_, 0)
      << "Error: layer_height does not match the stride.";
    CHECK_EQ(bottom[0]->shape(3) % stride_, 0)
      << "Error: layer_width does not match the stride.";
    top_shape[2] = bottom[0]->shape(2) / stride_;
    top_shape[3] = bottom[0]->shape(3) / stride_;
    top_shape[1] = bottom[0]->shape(1) * stride_ * stride_;
    top[0]->Reshape(top_shape);
  } else if (up_down_ == ReorgParameter_SampleType_UP) {
    CHECK_EQ(bottom[0]->shape(1) % (stride_ * stride_), 0)
      << "Error: layer_channels does not match the stride.";
    top_shape[1] = bottom[0]->shape(1) / stride_ / stride_;
    top_shape[2] = bottom[0]->shape(2) * stride_;
    top_shape[3] = bottom[0]->shape(3) * stride_;
    top[0]->Reshape(top_shape);
  } else {
    LOG(FATAL) << "Unknown Reorg SampleType.";
  }

  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "output count does not match.";
}

template <typename Dtype>
void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->num_axes(), 4);
    vector<int> top_shape(4,1);
    top_shape[0] = bottom[0]->shape(0);
    if (up_down_ == ReorgParameter_SampleType_DOWN) {
      CHECK_EQ(bottom[0]->shape(2) % stride_, 0)
        << "Error: layer_height does not match the stride.";
      CHECK_EQ(bottom[0]->shape(3) % stride_, 0)
        << "Error: layer_width does not match the stride.";
      top_shape[2] = bottom[0]->shape(2) / stride_;
      top_shape[3] = bottom[0]->shape(3) / stride_;
      top_shape[1] = bottom[0]->shape(1) * stride_ * stride_;
      top[0]->Reshape(top_shape);
    } else if (up_down_ == ReorgParameter_SampleType_UP) {
      CHECK_EQ(bottom[0]->shape(1) % (stride_ * stride_), 0)
        << "Error: layer_channels does not match the stride.";
      top_shape[1] = bottom[0]->shape(1) / stride_ / stride_;
      top_shape[2] = bottom[0]->shape(2) * stride_;
      top_shape[3] = bottom[0]->shape(3) * stride_;
      top[0]->Reshape(top_shape);
    } else {
      LOG(FATAL) << "Unknown Reorg SampleType.";
    }
    CHECK_EQ(top[0]->count(), bottom[0]->count())<< "output count must match input count";
}


template <typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  CHECK_EQ(bottom[0]->count(), top[0]->count())
    << "bottom and top blobs should have the same length.";

  if (up_down_ == ReorgParameter_SampleType_DOWN) {
    const vector<int>& shape = bottom[0]->shape();
    Reorg(bottom_data, true, shape, stride_, top_data);
  } else if (up_down_ == ReorgParameter_SampleType_UP) {
    const vector<int>& shape = top[0]->shape();
    Reorg(bottom_data, false, shape, stride_, top_data);
  } else {
    LOG(FATAL) << "Unknown Reorg SampleType.";
  }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    if (up_down_ == ReorgParameter_SampleType_DOWN) {
      const vector<int>& shape = bottom[0]->shape();
      Reorg(top_diff, false, shape, stride_, bottom_diff);
    } else if (up_down_ == ReorgParameter_SampleType_UP) {
      const vector<int>& shape = top[0]->shape();
      Reorg(top_diff, true, shape, stride_, bottom_diff);
    } else {
      LOG(FATAL) << "Unknown Reorg SampleType.";
    }
  }
}

INSTANTIATE_CLASS(ReorgLayer);
REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
