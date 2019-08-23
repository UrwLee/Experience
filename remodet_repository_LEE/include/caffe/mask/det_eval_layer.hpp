#ifndef CAFFE_MASK_DET_EVAL_LAYER_HPP_
#define CAFFE_MASK_DET_EVAL_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供了对检测器结果的评估。
 * 评估方法：
 * １．比较检测器检测结果与GT结果
 * ２．统计每一个检测器结果的状态：TP/FP
 * ３．针对每一个检测器结果提供一行输出：７位 <定义参考源文件>
 * ４．针对每个类别，提供GT的结果统计，每个类别输出一行：<定义参考源文件>
 */

template <typename Dtype>
class DetEvalLayer : public Layer<Dtype> {
 public:
  explicit DetEvalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetEval"; }

  /**
   * bottom[0]: -> 检测器结果输出, [1,1,Nd,7]
   * bottom[1]: -> GT结果，　[1,1,Ng,9]
   */
  virtual inline int ExactBottomBlobs() const { return 2; }

  /**
   * top[0]: -> 评估检测输出　[1,1,#,7]
   * # -> 所有LEVEL/DIFF评估级别下的总条目数数量
   * 条目包括：每个类别GT的统计结果，每个检测结果的评估状态
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // 背景ID: 0，未使用
  int background_label_id_;
  // deprecated
  Dtype overlap_threshold_;
  // 是否评估diff的gt，默认为False
  bool evaluate_difficult_gt_;
  // N+1
  int num_classes_;

  vector<int> gt_labels_;

  // 评估不同尺寸的阈值定义
  map<int, Dtype> size_thre_;
  // 评估不同难度级别的与正义定义
  map<int, Dtype> iou_thre_;
  std::vector<int> level_cnt_;
};

}

#endif
