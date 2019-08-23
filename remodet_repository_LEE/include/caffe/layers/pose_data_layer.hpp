#ifndef CAFFE_POSE_DATA_LAYER_HPP_
#define CAFFE_POSE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/pose_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为姿态识别任务提供数据输入。
 * 该层内部集成了PoseDataTransformer类用于进行数据载入和数据曾广
 * 建议阅读该层和PoseDataTransformer的实现细节来了解数据如何载入。
 * 图像曾广包括：
 * （１）随机Scale
 * （２）随机Crop
 * （３）随机Flip
 * （４）随机旋转
 * （５）[optional] 随机颜色失真
 */

template <typename Dtype>
class PoseDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PoseDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), pose_data_transform_param_(param.pose_data_transform_param()) {}
  virtual ~PoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  /**
   * top[0]: -> [N,3,H,W]
   * top[1]: -> [N,2*(34+18),RH,RW]
   * 乘以２的原因：
   * (1) for mask
   * (2) for labelMap
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  //  随机数
  shared_ptr<Caffe::RNG> prefetch_rng_;
  // 随机乱序样本队列
  virtual void ShuffleLists();

  // 加载一个minibatch
  virtual void load_batch(Batch<Dtype>* batch);

  // xmls
  vector<std::string> lines_;

  // 编号
  int lines_id_;

  // unused.
  Blob<Dtype> transformed_label_;

  // 数据转换器和参数
  PoseDataTransformationParameter pose_data_transform_param_;
  shared_ptr<PoseDataTransformer<Dtype> > pose_data_transformer_;
};

}  // namespace caffe

#endif  // CAFFE_POSE_DATA_LAYER_HPP_
