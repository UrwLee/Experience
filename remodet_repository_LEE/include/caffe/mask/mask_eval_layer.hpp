#ifndef CAFFE_MASK_MASK_EVAL_LAYER_HPP_
#define CAFFE_MASK_MASK_EVAL_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * 该层对实例的Mask分割精度进行评估。
 * 评估方法：
 * １．对实例的pred/gt进行每个点的对比；
 * ２．相等，则为tp，否则为fp
 * ３．accuracy = tp / (tp + fp)
 */

template <typename Dtype>
class MaskEvalLayer : public Layer<Dtype> {
 public:
  explicit MaskEvalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskEval"; }

  /**
   * bottom[0]: -> Mask preds (Nroi,1,RH,RW)
   * bottom[1]: -> Mask label (Nroi,1,RH,RW)
   * bottom[2]: -> ROI instance (1,1,Nroi,7)
   * bottom[3]: -> Active Flags for each ROI-instance (1,1,1,Nroi)
   */
  virtual inline int ExactBottomBlobs() const { return 4; }

  /**
   * top[0]: -> 针对于每个实例的评估结果　(1,1,Nroi,3)
   * <cid, size, accuracy>
   * 对于roi为unactive的实例，输出全部是－１
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

};

}

#endif
