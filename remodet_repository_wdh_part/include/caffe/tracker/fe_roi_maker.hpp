#ifndef CAFFE_TRACKER_FEROIMAKER_H
#define CAFFE_TRACKER_FEROIMAKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * ROI特征提取器：对某一张图片的某个ROI提取CNN上的特征
 * 注意：我们使用指定的CNN网络来提取某个ROI的层次化特征。
 * 在指定的ROI区域和特征图上，使用RoiResizeLayer进行尺度标准化。
 * 也可以使用RoAlignLayer进行尺度标准化操作。
 */

template <typename Dtype>
class FERoiMaker {
public:
  /**
   * network_proto／caffe_model　－> 网络
   * gpu_id -> GPU ID
   * features -> 使用的特征图
   * resized_width/resized_height -> resized尺寸　【使用RoiResizeLayer】
   */
  FERoiMaker(const std::string& network_proto,
             const std::string& caffe_model,
             const int gpu_id,
             const std::string& features,
             const int resized_width,
             const int resized_height);

  /**
   * 对指定图片的ROI区域提取特征。【使用RoiResizeLayer进行尺度标准化了】
   * @param image        [图片]
   * @param bbox         [位置]
   * @param resized_fmap [返回的特征Map]
   */
  void get_features(const cv::Mat& image, const BoundingBox<Dtype>& bbox, Blob<Dtype>* resized_fmap);

  /**
   * 对两幅图片的ROI进行特征提取。
   * @param image_prev   [历史图片]
   * @param image_curr   [当前图片]
   * @param bbox_prev    [历史位置]
   * @param bbox_curr    [当前位置]
   * @param resized_prev [历史特征]
   * @param resized_curr [当前特征]
   */
  void get_features(const cv::Mat& image_prev, const cv::Mat& image_curr,
                    const BoundingBox<Dtype>& bbox_prev, const BoundingBox<Dtype>& bbox_curr,
                    Blob<Dtype>* resized_prev, Blob<Dtype>* resized_curr);

  /**
   * 对两幅图片的ROI进行特征提取。　【暂未使用】
   * @param image_prev          [历史图片]
   * @param image_curr          [当前图片]
   * @param bbox_prev           [历史位置]
   * @param bbox_curr           [当前位置]
   * @param resized_unified_map [合并后的特征]
   */
  void get_features(const cv::Mat& image_prev, const cv::Mat& image_curr,
                    const BoundingBox<Dtype>& bbox_prev, const BoundingBox<Dtype>& bbox_curr,
                    Blob<Dtype>* resized_unified_map);

  /**
   * 获取整幅图的特征图和ROI的resize特征图
   * @param image        [图片]
   * @param bbox         [位置]
   * @param resized_fmap [ROI的resize特征图]
   * @param fmap         [整幅图的特征图]
   */
  void get_fmap(const cv::Mat& image, const BoundingBox<Dtype>& bbox,
                Blob<Dtype>* resized_fmap, Blob<Dtype>* fmap);

protected:

  /**
   * 加载图像数据到网络
   */
  void load(const cv::Mat& image);

  /**
   * 根据特征Blob名称获取特征图
   * @param feature [特征名]
   * @param map     [返回Blob]
   */
  void getFeatures(const std::string& feature, const Blob<Dtype>* map);

  /**
   * ROI的resize过程
   * @param feature     [特征Blob]
   * @param bbox        [位置]
   * @param resize_fmap [ROI的resize特征]
   */
  void roi_resize(const Blob<Dtype>* feature, const BoundingBox<Dtype>& bbox, Blob<Dtype>* resize_fmap);

  // 特征抽取网络
  boost::shared_ptr<caffe::Net<Dtype> > net_;
  // 网络输入尺度
  int input_width_;
  int input_height_;
  // roi_resize_layer层
  boost::shared_ptr<caffe::Layer<Dtype> > roi_resize_layer_;
  // 特征名
  std::string features_;
};

}

#endif
