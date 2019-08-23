#ifndef CAFFE_NMS_LAYER_HPP_
#define CAFFE_NMS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层的目的是在HeatMaps上找寻极值点。
 * 该层负责在每个Map上找到所有的极值点。
 * 我们定义了在每个Map上寻找的极值点数量上限：num_peaks
 * 该层的输入为：　[1,C,H,W]，C为通道数，一个Map的尺寸为[H,W]
 * 极值点的寻找方法：
 * （１）如果一个点的值，比周围8个点的值都要大；
 * （２）该点的值超过阈值(e.g.,0.05)
 * 满足上述两个条件的点被称之为极值点。
 * 该层的输出为：
 * top[0]: -> [1,C,num_peaks+1,3]
 * 3: -> <x,y,v>，归一化的x/y坐标以及该点的置信度
 * num_peaks: 寻找的极值点最大上限数
 * 1: -> 该通道Map实际找到的极值点数量
 * 注意：
 * 在实际实现时，输入的通道Map是52通道，包括了Limbs的Vec估计Map，因此我们处理时只处理了前18个通道；
 * 输入的bottom[0]: -> [1,52,H,W]
 * 输出的top[0]: -> [1,18,num_peaks+1,3]
 */

template <typename Dtype>
class NmsLayer : public Layer<Dtype> {
 public:
  explicit NmsLayer(const LayerParameter& param)
      : Layer<Dtype>(param), num_parts_(15), max_peaks_(36) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Nms"; }

  /**
   * bottom[0]: -> [1,52,H,W] (Heatmaps)
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: -> [1,18,num_peaks+1,3]
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

  /**
   * 获取设定的最大极值点数
   */
  virtual inline int GetMaxPeaks() const { return max_peaks_; }

  /**
   * 返回18
   */
  virtual inline int GetNumParts() const { return num_parts_; }

  /**
   * 返回设定的阈值
   */
  virtual inline Dtype GetThreshold() const { return threshold_; }

  /**
   * 设定阈值
   */
  virtual inline void SetThreshold(Dtype threshold) { threshold_ = threshold; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // GPU实现时使用，临时存储
  Blob<int> workspace;

  // 阈值
  Dtype threshold_;
  // 18
  int num_parts_;

  // 设定的最大极值数
  int max_peaks_;
};

}

#endif
