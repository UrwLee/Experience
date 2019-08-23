#include "caffe/tracker/image_proc.hpp"

namespace caffe {

// 获取裁剪上下文的实际区域,不超过原图片边界,上下文尺寸为2倍
template<typename Dtype>
void ComputeCropPadImageLocation(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, BoundingBox<Dtype>* pad_image_location) {
  // 获取中心位置
  const Dtype bbox_center_x = bbox_tight.get_center_x();
  const Dtype bbox_center_y = bbox_tight.get_center_y();
  // 获取图片尺寸
  const Dtype image_width = image.cols;
  const Dtype image_height = image.rows;
  // 获取输出box的尺寸: 加上上下文,2wx2h
  const Dtype output_width = bbox_tight.compute_output_width();
  const Dtype output_height = bbox_tight.compute_output_height();
  // 获取输出左上角坐标,限制(0,0)
  const Dtype roi_left = std::max((Dtype)0.0, bbox_center_x - output_width / 2);
  const Dtype roi_bottom = std::max((Dtype)0.0, bbox_center_y - output_height / 2);
  // 以中心为界,左边的宽度
  const Dtype left_half = std::min(output_width / 2, bbox_center_x);
  // 以中心为界, 右边的宽度
  const Dtype right_half = std::min(output_width / 2, image_width - bbox_center_x);
  // 实际输出的区域宽度: 左边与右边之和
  const Dtype roi_width =  std::max((Dtype)1.0, left_half + right_half);
  // 以中心为界, 上半高度
  const Dtype top_half = std::min(output_height / 2, bbox_center_y);
  // 以中心为界, 下半高度
  const Dtype bottom_half = std::min(output_height / 2, image_height - bbox_center_y);
  // 输出区域实际高度
  const Dtype roi_height = std::max((Dtype)1.0, top_half + bottom_half);
  // 设置输出ROI
  pad_image_location->x1_ = roi_left;
  pad_image_location->y1_ = roi_bottom;
  pad_image_location->x2_ = roi_left + roi_width;
  pad_image_location->y2_ = roi_bottom + roi_height;
}
template void ComputeCropPadImageLocation(const BoundingBox<float>& bbox_tight,
                      const cv::Mat& image, BoundingBox<float>* pad_image_location);
template void ComputeCropPadImageLocation(const BoundingBox<double>& bbox_tight,
                      const cv::Mat& image, BoundingBox<double>* pad_image_location);

template<typename Dtype>
void CropPadImage(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image) {
  BoundingBox<Dtype> pad_image_location;
  Dtype edge_spacing_x, edge_spacing_y;
  CropPadImage(bbox_tight, image, pad_image, &pad_image_location, &edge_spacing_x, &edge_spacing_y);
}
template void CropPadImage(const BoundingBox<float>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image);
template void CropPadImage(const BoundingBox<double>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image);

template<typename Dtype>
void CropPadImage(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
                  BoundingBox<Dtype>* pad_image_location, Dtype* edge_spacing_x, Dtype* edge_spacing_y) {
  // 获取当前box的上下文ROI的box位置
  // 在当前图片中的位置
  ComputeCropPadImageLocation(bbox_tight, image, pad_image_location);

  // Compute the ROI, ensuring that the crop stays within the boundaries of the image.
  const Dtype roi_left = std::min(pad_image_location->x1_, static_cast<Dtype>(image.cols - 1));
  const Dtype roi_bottom = std::min(pad_image_location->y1_, static_cast<Dtype>(image.rows - 1));
  const Dtype roi_width = std::min(static_cast<Dtype>(image.cols), std::max((Dtype)1.0, (Dtype)ceil(pad_image_location->x2_ - pad_image_location->x1_)));
  const Dtype roi_height = std::min(static_cast<Dtype>(image.rows), std::max((Dtype)1.0, (Dtype)ceil(pad_image_location->y2_ - pad_image_location->y1_)));

  // 裁剪ROI区域
  cv::Rect myROI(roi_left, roi_bottom, roi_width, roi_height);
  cv::Mat cropped_image = image(myROI);

  // 产生新图片,以装载裁剪得到的patch
  const Dtype output_width = std::max((Dtype)ceil(bbox_tight.compute_output_width()), roi_width);
  const Dtype output_height = std::max((Dtype)ceil(bbox_tight.compute_output_height()), roi_height);
  // 创建roi = (2w,2h)的图片
  cv::Mat output_image = cv::Mat(output_height, output_width, image.type(), cv::Scalar(0, 0, 0));

  // 上下文区域,左侧与上方可能超过边界的部分
  *edge_spacing_x = std::min(bbox_tight.edge_spacing_x(), static_cast<Dtype>(output_image.cols - 1));
  *edge_spacing_y = std::min(bbox_tight.edge_spacing_y(), static_cast<Dtype>(output_image.rows - 1));
  // 移除边界效应, 保证裁剪的ROI位于输出patch的中心位置
  cv::Rect output_rect(*edge_spacing_x, *edge_spacing_y, roi_width, roi_height);
  cv::Mat output_image_roi = output_image(output_rect);

  cropped_image.copyTo(output_image_roi);

  *pad_image = output_image;
}
template void CropPadImage(const BoundingBox<float>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
                  BoundingBox<float>* pad_image_location, float* edge_spacing_x, float* edge_spacing_y);
template void CropPadImage(const BoundingBox<double>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
                  BoundingBox<double>* pad_image_location, double* edge_spacing_x, double* edge_spacing_y);

}
