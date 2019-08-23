#ifndef CAFFE_POSE_DET_LAYER_HPP_
#define CAFFE_POSE_DET_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

/**
 * 该层负责接收如下两个提议：
 * （１）来自于检测器的提议：[1,1,Nd,7] <bindex,cid,score,xmin,ymin,xmax,ymax>
 * （２）来自于姿态的提议：[1,Np,18+1,3]
 * （３）peaks
 * （４）Heatmaps
 * 然后合并这两个proposal，输出如下结果：
 * top[0]: -> [1,1,N,61]
 * 61: -> <xmin,ymin,xmax,ymax,18*3,num_kps,score,id=-1>
 * 原理：
 * （１）获取姿态提取的pose-boxes
 * （２）遍历pose-boxes完成匹配：pose-boxes / det-boxes
 * （３）坐标更新：更新匹配成功的det-boxes，并为它添加关键点信息
 * （４）未匹配的pose-boxes，直接生成一个det-box，然后更新它的关节点信息
 * （５）为匹配的det-boxes，根据置信度，如果超过阈值，则保留，关键点信息不存在，或添加一些范围内可见的关键点，但不连线。
 */

namespace caffe {

template <typename Dtype>
class PoseDetLayer : public Layer<Dtype> {
 public:
  explicit PoseDetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseDet"; }

  /**
   * bottom[0] -> pose_proposal  [1,Np,18+1,3]
   * bottom[1] -> det_proposal [1,1,Nd,7]
   * bottom[2] -> peaks [1,18,num_peaks+1,3]
   * bottom[3] -> heatmaps(vecmaps) [1,52,H,W]
   */
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  /**
   * top[0]: -> [1,1,N,61] proposals
   * <xmin,ymin,xmax,ymax,18*3,num_kps,score,id=-1>
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // 18
  int num_parts_;
  // settings
  int max_peaks_;
  // 最大包含度　(按照converage计算，而不是IOU)
  Dtype coverage_min_thre_;
  // pose的proposal直接获得的box的置信度默认值
  Dtype score_pose_ebox_;
  // 未匹配成功的det-box，其保留的最低置信度
  Dtype keep_det_box_thre_;
};

}

#endif
