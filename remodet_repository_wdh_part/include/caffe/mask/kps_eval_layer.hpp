#ifndef CAFFE_MASK_KPS_EVAL_LAYER_HPP_
#define CAFFE_MASK_KPS_EVAL_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层提供了对姿态识别任务的评估方法。
 * 该方法包括：
 * １．针对于每个实例，计算其精度
 * ２．对每个实例，输出一行评估信息，包括id,accuracy等
 * ３．对于无需评估的实例(ROI)，使用Flags进行标记　(< 0)
 * 实例的精度统计为：
 * for(i = 0; i < 18; ++i) {
 *  if pred[i] == gt[i]
 *    tp++;
 *  else
 *    fp++;
 * }
 * accuracy = tp / (tp + fp)
 */

template <typename Dtype>
class KpsEvalLayer : public Layer<Dtype> {
 public:
  explicit KpsEvalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KpsEval"; }

  /**
   * bottom[0]: -> Kps估计信息　  (1,N,18,3)
   * bottom[1]: -> Kps GT信息    (1,N,18,3)
   * bottom[2]: -> 每个实例的ROI  (1,1,N,7)
   * bottom[3]: -> 每个实例是否需要评估的Flags (1,1,1,N)
   */
  virtual inline int ExactBottomBlobs() const { return 4; }

  /**
   * top[0]: -> 每个实例的评估返回信息　(1,1,N,3)
   * 每一行的信息为：　<cid, size, accuracy>
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // 评估参数
  // 置信度阈值：超过该值认为是正常的检测，否则被PASS
  Dtype conf_thre_;
  // 距离阈值，当对应关节点之间的距离低于该值时，认为是成功的检测
  Dtype distance_thre_;
};

}

#endif
