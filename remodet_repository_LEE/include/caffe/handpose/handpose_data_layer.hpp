#ifndef CAFFE_HANDPOSE_DATA_LAYER_HPP_
#define CAFFE_HANDPOSE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/handpose/handpose_augment.hpp"
#include "caffe/handpose/handpose_util.hpp"
#include "caffe/handpose/handpose_instance.hpp"
#include "caffe/handpose/bbox.hpp"

#include <boost/shared_ptr.hpp>

namespace caffe {

template <typename Dtype>
class HandPoseDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit HandPoseDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~HandPoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HandPoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  // 所有XML文件的集合列表 root<--->path pair
  vector<pair<std::string, std::string> > lines_;
  // vector<std::string> lines_;
  int lines_id_;
  // 增广
  boost::shared_ptr<HandPoseAugmenter> augPtr_;
};

}  // namespace caffe

#endif
