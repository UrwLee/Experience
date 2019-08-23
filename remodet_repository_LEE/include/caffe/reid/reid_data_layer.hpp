#ifndef CAFFE_REID_DATA_LAYER_HPP_
#define CAFFE_REID_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/reid/reid_transformer.hpp"

namespace caffe {

/**
 * 该层用于在Re-Identification任务中作为数据输入层使用。
 * 禁止直接使用该层。
 * 请在熟练掌握源码基础上使用。
 */

template <typename Dtype>
class ReidDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ReidDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), reid_transform_param_(param.reid_transform_param()) {}
  virtual ~ReidDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReidData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleLists();
  virtual void load_batch(Batch<Dtype>* batch);
  // xmls
  vector<std::string> lines_;
  int lines_id_;
  ReidTransformationParameter reid_transform_param_;
  shared_ptr<ReidTransformer<Dtype> > reid_transformer_;
};

}  // namespace caffe

#endif
