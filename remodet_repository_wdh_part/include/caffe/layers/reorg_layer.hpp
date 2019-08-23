#ifndef CAFFE_REORG_LAYER_HPP_
#define CAFFE_REORG_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/**
 * 该层提供了一种简单的下采样或上采样方法。
 * 模式：up_down_ = UP or DOWN
 * 默认stride = 2 (也可以指定其他值，但不建议)
 * 针对于DOWN:
 *  bottom[0]: (N,C,H,W)　　 (X)
 *  top[0]: (N,4*C,H/2,W/2) (Y)
 *  空间变换方式：
 *    X(n,c,i,j): -> Y(n,Oc,Oi,Oj)
 *      Oi = i / 2
 *      Oj = j / 2
 *      Oc = c * 4 + (i%2)*2 + (j%2)
 * 针对于UP:
 *  bottom[0]: (N,C,H,W)    (X)
 *  top[0]: (N,C/4,H*2,W*2) (Y)
 *  空间变换方式：
 *   X(n,c,i,j): -> Y(n,Oc,Oi,Oj)
 *     Oc = c / 4
 *        di = (c % 4) / 2
 *        dj = (c % 4) % 2
 *     Oi = i + di
 *     Oj = j + dj
 * Reorg层提供了上面所述的空间尺度变换。
 */

namespace caffe {

template <typename Dtype>
void Reorg(const Dtype* bottom_data, const bool forward,
          const vector<int> input_shape,
          const int stride, Dtype* top_data);

template <typename Dtype>
class ReorgLayer : public Layer<Dtype> {
 public:
  explicit ReorgLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reorg"; }

  /**
   * bottom[0]: (N,C,H,W)
   */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /**
   * top[0]: (N,OC,OH,OW)
   * for DOWN: OC=4*C, OH=H/2, OW=W/2
   * for UP: OC=C/4, OH=H*2, OW=W*2
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // DOWN or UP
  SampleType up_down_;
  // 默认为2
  int stride_;
};

}

#endif
