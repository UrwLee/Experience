#ifndef CAFFE_MASK_DENSE_DET_OUT_LAYER_HPP_
#define CAFFE_MASK_DENSE_DET_OUT_LAYER_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

/**
 * DenseBBox.
 */

template <typename Dtype>
class VisualMtdLayer : public Layer<Dtype> {
 public:
  explicit VisualMtdLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VisualMtd"; }
  /**
   * bottom[0]: -> loc predictions
   * bottom[1]: -> conf predictions
   * bottom[2]: -> prior_bbox
   * bottom[3]: -> [optional] image_data
   */
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  /**
   * top[0]: -> loss (1)
   */
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //   NOT_IMPLEMENTED;
  // }

 

  // 可视化参数
  VisualizeParameter visual_param_;
  int num_classes_;
  
};

}

#endif
