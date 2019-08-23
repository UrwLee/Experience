#ifndef CAFFE_FTRACKER_DATA_LAYER_HPP_
#define CAFFE_FTRACKER_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/tracker/ftracker_data_loader.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为FTracker的训练提供输入数据。
 * 该层只需要定义号ftracker_data_param的参数即可。
 * 该层使用了一个FTrackerDataLoader类来加载和转换数据。
 * 请参考ftracker_data_param参数字段和FTrackerDataLoaderParameter参数字段来获取该层的详细性能。
 */

template <typename Dtype>
class FTrackerDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit FTrackerDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), loader_param_(param.ftracker_data_param().load_param()) {}
  virtual ~FTrackerDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FTrackerData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  /**
   * top[0]: (2N,C,FH,FW) -> 样本对特征输入
   * top[1]: (N*4,1,1,1) -> 估计的box-gt-label
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  //  加载一个minibatch的数据
  virtual void load_batch(Batch<Dtype>* batch);

  // 数据加载器参数
  FTrackerDataLoaderParameter loader_param_;

  // 数据加载器
  shared_ptr<FTrackerDataLoader<Dtype> > ftracker_data_loader_;
};

}  // namespace caffe

#endif
