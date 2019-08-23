#ifndef CAFFE_ROI_ALIGN_LAYER_HPP_
#define CAFFE_ROI_ALIGN_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层类似于RoiPooling层，但去掉了ROI边界的量化过程。
 * 该层执行如下计算：
 * （１）计算ROI的位置
 * （２）将输出的bin映射到输入的区域；
 * （３）对输入的区域执行maxpool操作
 *    注意：目前的pool点为该区域的四个中间点：
 *     [0.25,0.25] [0.25,0.75] [0.75,0.25] [0.75,0.75]
 *     全部使用归一化坐标进行计算
 * （４）对输出赋值
 * 该层提供了前向和反向计算方法。
 */

template <typename Dtype>
class RoiAlignLayer : public Layer<Dtype> {
 public:
  explicit RoiAlignLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RoiAlign"; }

  /**
   * bottom[0] -> [N,C,H,W]
   * bottom[1] -> [1,1,Nroi,7/9]
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0] -> [Nroi,C,RH,RW]
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * 双线性插值
   * @param map   [插值Map]
   * @param x     [浮点位置坐标]
   * @param y     [浮点位置坐标]
   * @param fw    [Map尺度]
   * @param fh    [Map尺度]
   * @param v     [返回的插值计算结果]
   * @param coeff [返回插值系数]
   * @param idx   [返回插值坐标点]
   */
  void bilinearInterpolation(const Dtype* map, Dtype x, Dtype y, int fw, int fh, Dtype* v,
  										Dtype* coeff, int* idx);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 输出resize尺寸
  int roiResizedWidth_;
  int roiResizedHeight_;
  // 1 -> 中心插值
  // 4 -> 4个位置插值，，默认为4，max-pool
  int inter_times_;

  // unused.
  Dtype spatial_scale_;

  // 索引，用于反向传播
  Blob<int> idx_;
  // 系数，用于反向传播
  Blob<Dtype> coeff_;
};

}

#endif
