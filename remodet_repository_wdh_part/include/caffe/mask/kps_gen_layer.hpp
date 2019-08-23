#ifndef CAFFE_MASK_KPS_GEN_LAYER_HPP_
#define CAFFE_MASK_KPS_GEN_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层将根据ROI自动生成所有关节点的Heatmaps－Label
 * 方法如下：
 * １．计算对应实例的关节点在对应ROI区域内的坐标
 * ２．如果第i个关节点可见，则在该关节点的对应MAP(RH,RW)上，在这个点处放置１，其余位置全部是０
 */

template <typename Dtype>
class KpsGenLayer : public Layer<Dtype> {
 public:
  explicit KpsGenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KpsGen"; }

  /**
   * bottom[0]: -> ROIs (1,1,Nroi,7)
   * <bindex,cid,pid,xmin,ymin,xmax,ymax>
   * bottom[1]: -> Kps Label for input image (1,1,Ng,61)
   * <bindex,cid,pid,is_diff,iscrowd,has_kps,num_kps,18*3>
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0]: -> Heatmaps for each ROI-instance. (Nroi,18,RH,RW)
   *       : -> Results for each ROI-instance. (Nroi,18)
   * top[1]: -> Flags for each channel of every ROI-instance. (1,1,Nroi,18)
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  // 生成的Heatmap-Label的resize尺寸
  int resized_height_, resized_width_;
  // type of outputs, default: false
  bool use_softmax_;
};

}

#endif
