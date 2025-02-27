#ifndef CAFFE_DENSE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_DENSE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
//相关的函数需要修改，增加mask和image同时变换的函数
#include "caffe/mask/seg_data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {


template <typename Dtype>
class DenseImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
  explicit DenseImageDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param),
      seg_data_transform_param_(param.seg_data_transformer_param())
  {
    seg_data_data_transformer_.reset(
      new SegDataTransformer<Dtype>(seg_data_transform_param_, this->phase_));
  }
  virtual ~DenseImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
//  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch);
  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
  SegDataTransformationParameter seg_data_transform_param_;
  shared_ptr<SegDataTransformer<Dtype> > seg_data_data_transformer_;
  Blob<Dtype> transformed_label_;
};


}  // namespace caffe

#endif

