#ifndef CAFFE_TRACKER_DATA_LAYER_HPP_
#define CAFFE_TRACKER_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/tracker/tracker_data_loader.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为Tracker的训练提供数据输入功能。
 * 该层内部集成了一个TrackerDataLoader进行数据处理和转换。
 * 该层只需要指定参数tracker_data_param即可。
 * 该层的所有行为有tracker_data_param/TrackerDataLoaderParameter决定。
 */

template <typename Dtype>
class TrackerDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TrackerDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), loader_param_(param.tracker_data_param().load_param()) {}
  virtual ~TrackerDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TrackerData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  /**
   * top[0]: -> [2N,3,H,W] (样本对)
   * top[1]: -> [N*4,1,1,1] (gt-Label值)
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  // 加载一个minibatch数据
  virtual void load_batch(Batch<Dtype>* batch);

  // 加载器和参数
  TrackerDataLoaderParameter loader_param_;
  shared_ptr<TrackerDataLoader<Dtype> > tracker_data_loader_;
};

}  // namespace caffe

#endif
