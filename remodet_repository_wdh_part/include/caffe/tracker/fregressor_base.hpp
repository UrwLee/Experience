#ifndef CAFFE_TRACKER_FREGRESSOR_BASE_H
#define CAFFE_TRACKER_FREGRESSOR_BASE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * ##和F##的区别：
 * ##是使用ROI-Patch
 * F##是使用ROI-Blobs(特征Blobs)
 * 这是一个使用ROI特征Blob的回归器基类
 * 注意：该回归器接收前后两帧的特征Blobs，然后输出一个Box坐标
 */

// 回归器
template <typename Dtype>
class FRegressorBase {
public:
  FRegressorBase();

  /**
   * 回归方法：对单目标或多目标进行估计
   * @param curr [当前Blobs]
   * @param prev [历史Blobs]
   * @param bbox [回归的结果]
   */
  virtual void Regress(const Blob<Dtype>& curr, const Blob<Dtype>& prev, BoundingBox<Dtype>* bbox) = 0;
  virtual void Regress(const std::vector<boost::shared_ptr<Blob<Dtype> > >& curr,
                       const std::vector<boost::shared_ptr<Blob<Dtype> > >& prev,
                       std::vector<BoundingBox<Dtype> >* bboxes) = 0;

  /**
   * 初始化
   */
  virtual void Init() {}

  // 网络对象
  boost::shared_ptr<caffe::Net<Dtype> > get_net() { return net_; }

protected:
  // 网络对象
  boost::shared_ptr<caffe::Net<Dtype> > net_;
};

}

#endif
