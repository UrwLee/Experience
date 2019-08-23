#ifndef CAFFE_REID_ALLOCATE_ID_LAYER_HPP_
#define CAFFE_REID_ALLOCATE_ID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/reid/basic_match.hpp"

namespace caffe {

template <typename Dtype>
class AllocateIdLayer : public Layer<Dtype> {
 public:
  explicit AllocateIdLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AllocateId"; }
  // bottom[0] -> Propsals  [1,1,N,61]
  // bottom[1] -> [1,N,11,512] (fMaps)
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // top[0] -> [1,1,N,62]  (the last one is similarity)
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 private:
   // 匹配IOU阈值
   Dtype match_iou_thre_;
   // 计算模板相似度的阈值　(Limbs & Torso)
   Dtype thre_for_cal_similarity_;
   // 判定遮挡的coverage阈值
   Dtype occu_coverage_thre_;
   // 用于结构化模板更新的scale
   Dtype scale_for_update_str_;
   // 用于区域性模板更新的scale
   Dtype scale_for_update_area_;
   // 用于分裂匹配的IOU阈值
   Dtype split_iou_thre_;
   // 用于分裂匹配的模板相似度阈值
   Dtype split_simi_thre_;

   // 提议池
   vector<pProp<Dtype> > props_;
   // 目标池
   vector<pTarget<Dtype> > targets_;
};  // namespace caffe

}
#endif
