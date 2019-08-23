#ifndef CAFFE_TRACKER_BOUNDING_BOX_H
#define CAFFE_TRACKER_BOUNDING_BOX_H

#include <vector>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类定义了一个Box对象，并提供了一系列方法。
 */

template<typename Dtype>
class BoundingBox
{
public:
  BoundingBox();

  /**
   * 构造方法：使用四个值即可构造一个Box对象
   * bounding_box[0] -> xmin
   * bounding_box[1] -> ymin
   * bounding_box[2] -> xmax
   * bounding_box[3] -> ymax　
   */
  BoundingBox(const std::vector<Dtype>& bounding_box);

  /**
   * 将Box对象的四个值返回为一个vector对象
   * @param bounding_box [返回的vector对象指针]
   * xmin -> bounding_box->[0]
   * ymin -> bounding_box->[1]
   * xmax -> bounding_box->[2]
   * ymax -> bounding_box->[3]
   */
  void GetVector(std::vector<Dtype>* bounding_box) const;

  /**
   * 打印Box的信息
   */
  void Print() const;

  // 返回一个常量，仅在Tracker中使用，作为ROI的区域定义
  Dtype get_context_factor();

  /**
   * 在指定的图像上绘制该Box
   * @param r     [绘制线条颜色R]
   * @param g     [绘制线条颜色G]
   * @param b     [绘制线条颜色B]
   * @param image [待绘制的cv::Mat指针]
   */
  void Draw(const int r, const int g, const int b, cv::Mat* image) const;
  void DrawNorm(const int r, const int g, const int b, cv::Mat* image) const;
  /**
   * 使用默认参数，在指定的图像上绘制该Box
   * @param figure_ptr [待绘制的cv::Mat指针]
   */
  void DrawBoundingBox(cv::Mat* figure_ptr) const;
  void DrawBoundingBoxNorm(cv::Mat* figure_ptr) const;
  /**
   * 将Box按照指定图片进行归一化，然后乘以默认的scale参数　【该参数直接由static const定义】
   * @param image       [指定图片]
   * @param bbox_scaled [返回的Scale-Box对象]
   */
  void Scale(const cv::Mat& image, BoundingBox<Dtype>* bbox_scaled) const;

  /**
   * Scale的逆操作：先/scale，然后乘以image的尺寸
   * @param image         [指定图片]
   * @param bbox_unscaled [返回的Unscale-Box对象]
   */
  void Unscale(const cv::Mat& image, BoundingBox<Dtype>* bbox_unscaled) const;

  /**
   * 将Box在指定的ROI上进行坐标转换
   * @param search_location [ROI－Box]
   * @param edge_spacing_x  [边界-x]
   * @param edge_spacing_y  [边界-y]
   * @param bbox_recentered [返回的转换Box]
   */
  void Recenter(const BoundingBox<Dtype>& search_location,
                const Dtype edge_spacing_x, const Dtype edge_spacing_y,
                BoundingBox<Dtype>* bbox_recentered) const;

  /**
   * 将Box在指定的ROI上进行反坐标转换
   * @param raw_image       [原图]
   * @param search_location [ROI-Box]
   * @param edge_spacing_x  [边界-x]
   * @param edge_spacing_y  [边界-y]
   * @param bbox_uncentered [返回的反转换Box]
   */
  void Uncenter(const cv::Mat& raw_image, const BoundingBox<Dtype>& search_location,
                const Dtype edge_spacing_x, const Dtype edge_spacing_y,
                BoundingBox<Dtype>* bbox_uncentered) const;

  /**
   * Box的随机增广，返回增广后的Box
   * @param image              [图像]
   * @param lambda_scale_frac  [参数：w/h的随机尺度变化系数]
   * @param lambda_shift_frac  [参数：中心位置的随机偏移系数]
   * @param min_scale          [尺度变化的下限]
   * @param max_scale          [尺度变化的上限]
   * @param shift_motion_model [是否使用运动模型，false -> 新的ROI默认与上一帧的ROI在同一位置]
   * @param bbox_rand          [返回的增广后的Box]
   */
  void Shift(const cv::Mat& image,
             const Dtype lambda_scale_frac, const Dtype lambda_shift_frac,
             const Dtype min_scale, const Dtype max_scale,
             const bool shift_motion_model,
             BoundingBox<Dtype>* bbox_rand) const;

  /**
   * 返回scale系数
   */
  Dtype get_scale_factor() const { return scale_factor_; }

  /**
   * 返回Box的长和宽
   * @return [返回w/h]
   */
  Dtype get_width() const { return x2_ - x1_;  }
  Dtype get_height() const { return y2_ - y1_; }

  /**
   * 返回Box的中心位置
   * @return [中心位置x/y]
   */
  Dtype get_center_x() const;
  Dtype get_center_y() const;

  /**
   * 按照kContextFactor返回含上下文的宽度和高度
   */
  Dtype compute_output_height() const;
  Dtype compute_output_width() const;

  /**
   * 当输出含上下文的Box时，该方法计算这个上下文Box与原始图像边界之间的距离
   * 如果超过原始图像边界，则返回该边界距离
   * 如果没有超过，则返回0
   * @return [边界值]
   */
  Dtype edge_spacing_x() const;
  Dtype edge_spacing_y() const;

  /**
   * 计算该Box的面积：　get_width() * get_height()
   * @return [面积返回值]
   */
  Dtype compute_area() const;

  /**
   * 计算与另一个Box对象之间的交叠面积
   * @param  bbox [对方Box]
   * @return      [交叠面积]
   */
  Dtype compute_intersection(const BoundingBox<Dtype>& bbox) const;

  /**
   * 计算与另一个Box对象之间的IOU
   * @param  bbox [对方Box]
   * @return      [IOU]
   */
  Dtype compute_iou(const BoundingBox<Dtype>& bbox) const;
  vector<Dtype> compute_iou_expdist(const BoundingBox<Dtype>& bbox, Dtype sigma) const;

  /**
   * 计算与另一个Box对象之间的Coverage
   * Coverage = compute_intersection(bbox) / compute_area();
   * @param  bbox [对方Box]
   * @return      [Coverage值]
   */
  Dtype compute_coverage(const BoundingBox<Dtype>& bbox) const;
  vector<Dtype> compute_iou_coverage(const BoundingBox<Dtype>& bbox) const;
  /**
   * 计算与另一个Box对象之间的Coverage
   * 注意：相对于另一方Box的Coverage
   * Coverage = compute_intersection(bbox) / bbox.compute_area();
   * @param  bbox [对方Box]
   * @return      [相对于另一方的Coverage值]
   */
  Dtype compute_obj_coverage(const BoundingBox<Dtype>& bbox) const;

  /**
   * 计算该Box在第三方bbox上面的相对坐标，返回coverage值
   * 当coverage > 0时，返回的proj_bbox有效
   * @param  bbox      [对方Box]
   * @param  proj_bbox [输出的proj_bbox]
   * @return           [coverage值]
   */
  Dtype project_bbox(const BoundingBox<Dtype>& bbox, BoundingBox<Dtype>* proj_bbox) const;

  /**
   * 对Box的值进行裁剪
   * @param min [下限]
   * @param max [上限]
   */
  void clip(const Dtype min, const Dtype max);

  /**
   * 对Box的值进行裁剪，默认上限和下限是1/0
   */
  void clip();

  /**
   * x1_ -> xmin
   * y1_ -> ymin
   * x2_ -> xmax
   * y2_ -> ymax
   */
  Dtype x1_, y1_, x2_, y2_;

  /**
   * scale_factor_ -> 默认＝ 10
   * 该值可以随意修改，确保在进行Scale/Unscale时使用相同的值即可
   */
  Dtype scale_factor_;
};

}

#endif // CAFFE_TRACKER_BOUNDING_BOX_H
