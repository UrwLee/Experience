#ifndef CAFFE_MASK_DET_OUT_LAYER_HPP_
#define CAFFE_MASK_DET_OUT_LAYER_HPP_

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

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * 该类提供了检测器输出的方法。
 * 包括：
 * １．将回归器和分类器的结果进行结构化
 * ２．坐标进行解码
 * ３．执行NMS
 * ４．将检测器结果输出
 * ５．[optional]可视化
 * 注意：输出的cid提供了alias_id参数进行修改。该参数用于对不同的类别的分类器结果进行拼接。
 * 默认：
 * 0 -- body
 * 1 -- hand
 * 2 -- head
 * 3 -- face
 */

template <typename Dtype>
class DetOutLayer : public Layer<Dtype> {
public:
  explicit DetOutLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
      sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetOut"; }
  /**
   * bottom[0]: -> loc predictions
   * bottom[1]: -> conf predictions
   * bottom[2]: -> prior_bbox
   * bottom[3]: -> [optional] image_data
   */
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 5; }
  /**
   * top[0]: -> loss (1)
   * top[1]: -> matched ROIs (1,1,Nroi,7)
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
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // minibatch样本数
  int num_;
  // prior_bboxes的数量
  int num_priors_;
  // N+1
  int num_classes_;
  // 背景ID: 0 -> 未使用
  int background_label_id_;
  // boxes的格式：默认为CENTER
  CodeType code_type_;

  // 是否在目标中直接集成boxes-编码功能
  // 默认是FALSE,需要程序额外编解码
  bool variance_encoded_in_target_;
  DetectionOutputParameter_NmsType vote_or_not_;
  DetectionOutputParameter_SoftType soft_type_;
  // priors按照置信度保留的最高数量
  int top_k_;

  // alias id
  int alias_id_;

  vector<int> target_labels_;

  // 参与评估的阈值
  Dtype conf_threshold_;

  // NMS阈值
  Dtype nms_threshold_;

  // 参与评估的size阈值，低于该值不参与评估
  Dtype size_threshold_;

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  float objectness_score_;

  vector<vector<LabeledBBox<Dtype> > > all_arm_loc_preds_;
  vector<vector<vector<Dtype> > > neg_scores_;
};

}

#endif
