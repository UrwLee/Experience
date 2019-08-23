#ifndef CAFFE_TRACKER_REGRESSOR_BASE_H
#define CAFFE_TRACKER_REGRESSOR_BASE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * 该类是一个基类，请在派生类实现具体的构造过程。
 * 该类提供了一个回归的方法：
 * １．输入当前帧的ROI-Patch，以及历史帧的ROI-Patch
 * ２．输出回归器的坐标回归结果
 */

// 回归器
template <typename Dtype>
class RegressorBase {
public:
  // must be constructed by the subclasses
  RegressorBase();

  /**
   * 对单个目标对象进行回归
   * @param image  [当前帧的ROI-Patch]
   * @param target [历史帧的ROI-Patch]
   * @param bbox   [回归结果]
   * 注意：务必在派生类中实现该方法
   */
  virtual void Regress(const cv::Mat& image, const cv::Mat& target, BoundingBox<Dtype>* bbox) = 0;

  /**
   * 对多个目标对象进行回归
   * @param images  [当前帧的ROI-Patch的集合]
   * @param targets [历史帧的ROI-Patch的集合]
   * @param bboxes  [回归结果的集合]
   */
  virtual void Regress(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& targets, std::vector<BoundingBox<Dtype> >* bboxes) = 0;

  /**
   * 初始化
   */
  virtual void Init() {}

  /**
   * 获取回归网络
   */
  boost::shared_ptr<caffe::Net<Dtype> > get_net() { return net_; }

protected:

  /**
   * 内部集成的回归网络，在派生类中构造实现
   */
  boost::shared_ptr<caffe::Net<Dtype> > net_;
};

}

#endif
