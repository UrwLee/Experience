#ifndef CAFFE_ROI_POOLING_LAYER_HPP_
#define CAFFE_ROI_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/**
 * ROI-Pooling层：用于提取指定区域的特征，并做尺度归一化
 * 该层对roi的边界进行了量化，并对输出bins的输入映射区域也做了量化。
 * 该层可以被RoiAlign替代。
 */

namespace caffe {

template <typename Dtype>
class ROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

  /**
   * bottom[0]: -> [N,C,H,W]
   * bottom[1]: -> [1,1,Nroi,5] (ROIs)
   * 5: -> <bindex,xmin,ymin,xmax,ymax>
   * 注意：xmin/ymin/xmax/ymax是绝对尺寸，与输入图片的尺度处于相同水平
   */
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }

  /**
   * top[0]: -> [Nroi,C,RH,RW]
   */
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 通道数
  int channels_;
  // 输入尺度
  int height_;
  int width_;
  // 输出尺度
  int pooled_height_;
  int pooled_width_;

  // 该featureMap相比于输入图像的空间尺度比例：e.g., 1/8或1/16
  Dtype spatial_scale_;

  // GPU反向传播用
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_ROI_POOLING_LAYER_HPP_
