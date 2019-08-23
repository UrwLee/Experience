#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  // for each sample
  buffer_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                  bottom[0]->width());
  // for each channel
  buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
  // for each spatial location
  buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  NormalizeParameter norm_param = this->layer_param().norm_param();
  across_spatial_ = norm_param.across_spatial();
  // across_spatial -> [c,h,w] -> generate only one value
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    // otherwise: just across channels
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  eps_ = norm_param.eps();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->width() * bottom[0]->height();
  // for channels and initialization as 1.
  sum_channel_multiplier_.Reshape(1, channels, 1, 1);
  caffe_set(channels, Dtype(1), sum_channel_multiplier_.mutable_cpu_data());
  // for spatial locations and initial as 1.
  sum_spatial_multiplier_.Reshape(1, 1, bottom[0]->height(),
                                  bottom[0]->width());
  caffe_set(spatial_dim, Dtype(1), sum_spatial_multiplier_.mutable_cpu_data());
  // channel_shared -> all channels share the parameters
  channel_shared_ = norm_param.channel_shared();
  // if blobs_ exists, then skip parameter initial.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // otherwise: initialization
    // blobs_.size -> 1 : only scale parameter
    this->blobs_.resize(1);
    // if channel_shared_ -> scale(blobs_[0]) is just a scalar
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      // otherwise: scale is [1,c] Blob
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    // fill the parameters for initialization: scale
    shared_ptr<Filler<Dtype> > scale_filler;
    // get the defined filler parameter
    if (norm_param.has_scale_filler()) {
      scale_filler.reset(GetFiller<Dtype>(norm_param.scale_filler()));
    } else {
      // otherwise: use default filler: constant(1.)
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      scale_filler.reset(GetFiller<Dtype>(filler_param));
    }
    // fill the scale parameter
    scale_filler->Fill(this->blobs_[0].get());
  }

  // if channel_shared_ -> scale size should be 1. (a scalar)
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    // otherwise: scale size should be channels, (1,c)
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Scale size is inconsistent with prototxt config";
  }
  // allows Backward: scale parameter
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  // top[0] Reshape
  top[0]->ReshapeLike(*bottom[0]);
  // buffer_ Reshape -> [1,c,h,w]
  buffer_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                  bottom[0]->width());
  // define the parameter
  // !across_spatial_ -> differs within each output location, just across
  // channels
  // across_spatial_ -> across channels and [spatial_dim]
  if (!across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  // define the spatial_dim
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  // define the sum_spatial_multiplier_ Reshape
  if (spatial_dim != sum_spatial_multiplier_.count()) {
    sum_spatial_multiplier_.Reshape(1, 1, bottom[0]->height(),
                                    bottom[0]->width());
    caffe_set(spatial_dim, Dtype(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
}

/**
 * 正向传播：
 * 针对每个样本计算通道均方值
 * 根据该值：输入归一化
 * 输出再进行scale增益处理
 * 用于反向传播的Blobs：
 * norm_ -> 存储每个样本的通道均方值
 * scale -> 存储待学习的参数
 */
template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  // get the input data [b,c,h,w]
  const Dtype *bottom_data = bottom[0]->cpu_data();
  // get the output data [b,c,h,w]
  Dtype *top_data = top[0]->mutable_cpu_data();
  // get the scale parameter [1,c]
  const Dtype *scale = this->blobs_[0]->cpu_data();
  // get the buffer data [1,c,h,w]
  Dtype *buffer_data = buffer_.mutable_cpu_data();
  // get the norm data, [b,1,h,w]
  Dtype *norm_data = norm_.mutable_cpu_data();
  // add eps to avoid overflow
  caffe_set<Dtype>(norm_.count(), Dtype(eps_), norm_data);
  // get the sum_channel_multiplier data [1,c,1,1]
  const Dtype *sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
  // get the sum_spatial_multiplier data [1,1,h,w]
  const Dtype *sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
  // num
  int num = bottom[0]->num();
  // dim = c*h*w
  int dim = bottom[0]->count() / num;
  // spatial_dim = h*w
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  // channels
  int channels = bottom[0]->channels();
  // scan all num samples
  for (int n = 0; n < num; ++n) {
    // x^2 -> buffer data
    caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
    if (across_spatial_) {
      // add eps to avoid overflow
      /**
       * if across_spatial_ defined
       * norm_[n] = sqrt{ sum(x^2)+eps }
       */
      norm_data[n] =
          pow(caffe_cpu_asum<Dtype>(dim, buffer_data) + eps_, Dtype(0.5));
      /**
       * norm_data[n] -> the avg root
       * out[..]= in[..]/norm_data[n]
       */
      caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      /**
       * for each output location: sum across the channels
       * norm_data -> [h,w]
       * buffer_data -> [c,h,w]
       * sum_channel_multiplier -> [c]
       */
      caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                            buffer_data, sum_channel_multiplier, Dtype(1),
                            norm_data);
      /**
       * for c in channels: norm_data = sqrt{ sum(in[c,location]) }
       */
      caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      // scale the layer
      /**
       * 将norm的结果乘以每个通道的增益，得到buff
       * sum_channel_multiplier全部为1
       */
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Dtype(1), sum_channel_multiplier, norm_data,
                            Dtype(0), buffer_data);
      /**
       * 输入除以该结果，得到输出
       */
      caffe_div<Dtype>(dim, bottom_data, buffer_data, top_data);
      // point to the next sample
      norm_data += spatial_dim;
    }
    /**
     * 输出结果再次乘以增益scale
     * 如果通道共享，则直接乘以scale[0]即可
     * 否则，将scale[c..]分布到整个输出空间面，然后对输出结果执行element-wise乘法
     * 注意：sum_spatial_multiplier/sum_channel_multiplier -> 1.
     */
    if (channel_shared_) {
      caffe_scal<Dtype>(dim, scale[0], top_data);
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Dtype(1), scale, sum_spatial_multiplier,
                            Dtype(0), buffer_data);
      caffe_mul<Dtype>(dim, top_data, buffer_data, top_data);
    }
    /**
     * 指向下个样本
     */
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  // input err
  const Dtype *top_diff = top[0]->cpu_diff();
  // output data
  const Dtype *top_data = top[0]->cpu_data();
  // input data
  const Dtype *bottom_data = bottom[0]->cpu_data();
  // output err
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  // scale
  const Dtype *scale = this->blobs_[0]->cpu_data();
  // norm
  const Dtype *norm_data = norm_.cpu_data();
  // buffer_
  Dtype *buffer_data = buffer_.mutable_cpu_data();
  // buffer_channel
  Dtype *buffer_channel = buffer_channel_.mutable_cpu_data();
  // buffer_spatial
  Dtype *buffer_spatial = buffer_spatial_.mutable_cpu_data();
  // sum_channel_multiplier
  const Dtype *sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
  // sum_spatial_multiplier
  const Dtype *sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
  // count
  int count = top[0]->count();
  // num
  int num = top[0]->num();
  // c*h*w
  int dim = count / num;
  // h*w
  int spatial_dim = top[0]->height() * top[0]->width();
  // channels
  int channels = top[0]->channels();

  // Propagate to param
  // 计算scale的梯度
  if (this->param_propagate_down_[0]) {
    // scale error
    Dtype *scale_diff = this->blobs_[0]->mutable_cpu_diff();
    /**
     * 通道共享：
     * 矩阵累加求和： SUM(data,diff) / scale
     */
    if (channel_shared_) {
      scale_diff[0] +=
          caffe_cpu_dot<Dtype>(count, top_data, top_diff) / scale[0];
    } else {
      /**
       * 遍历每个样本
       * 1. 梯度传播map buffer[c,h,w] = top_data * top_diff
       * 2. 沿通道累加梯度： buffer_channel[c] = buffer[c,h,w] *
       * sum_spatial_multiplie[h,w] (SUM)
       * 3. 计算梯度结果： buffer_channel[c] /= scale[c]
       * 4. 梯度叠加：scale_diff[c] += buffer_channel[c]
       */
      for (int n = 0; n < num; ++n) {
        caffe_mul<Dtype>(dim, top_data + n * dim, top_diff + n * dim,
                         buffer_data);
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_spatial_multiplier, Dtype(0),
                              buffer_channel);
        // store a / scale[i] in buffer_data temporary
        caffe_div<Dtype>(channels, buffer_channel, scale, buffer_channel);
        caffe_add<Dtype>(channels, buffer_channel, scale_diff, scale_diff);
      }
    }
  }

  // 计算输出到下一层的误差梯度
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
        /**
         * across_spatial_：
         * 1. 计算全局误差累计
         * 2. 计算初始误差：输入×增益
         * 3. 误差矫正：减去输入误差
         * 4. 误差增益：乘以增益因子
         */
        Dtype a = caffe_cpu_dot<Dtype>(dim, bottom_data, top_diff);
        caffe_cpu_scale<Dtype>(dim, a / norm_data[n] / norm_data[n],
                               bottom_data, bottom_diff);
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_scal<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_diff);
      } else {
        // dot product between bottom_data and top_diff
        // 误差缓冲图：[c,h,w]
        caffe_mul<Dtype>(dim, bottom_data, top_diff, buffer_data);
        // 沿通道累加： [h,w]
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(0),
                              buffer_spatial);
        // 重新计算误差图
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier,
                              buffer_spatial, Dtype(0), buffer_data);
        /**
         * 计算初始误差
         */
        caffe_mul<Dtype>(dim, bottom_data, buffer_data, bottom_diff);
        // 计算归一化的传播
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(2), buffer_spatial);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier,
                              buffer_spatial, Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
        // subtract
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier, norm_data,
                              Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_scal<Dtype>(dim, scale[0], bottom_diff);
      } else {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), scale, sum_spatial_multiplier,
                              Dtype(0), buffer_data);
        caffe_mul<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

} // namespace caffe
