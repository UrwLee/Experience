#include "caffe/tracker/tracker_base.hpp"

#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/image_proc.hpp"

namespace caffe {

template <typename Dtype>
void TrackerBase<Dtype>::Init(const cv::Mat& image, const BoundingBox<Dtype>& bbox_gt) {
  image_prev_ = image;
  bbox_prev_ = bbox_gt;

  // TODO - use a motion model?
  bbox_curr_init_ = bbox_gt;
}

template <typename Dtype>
void TrackerBase<Dtype>::Tracking(const cv::Mat& image_curr, RegressorBase<Dtype>* reg,
                           BoundingBox<Dtype>* bbox_estimate_uncentered) {
  cv::Mat target_pad;
  CropPadImage(bbox_prev_, image_prev_, &target_pad);

  // curr patch
  cv::Mat curr_search_region;
  // box in patch
  BoundingBox<Dtype> search_location;
  // relocation
  Dtype edge_spacing_x, edge_spacing_y;
  CropPadImage(bbox_curr_init_, image_curr, &curr_search_region, &search_location, &edge_spacing_x, &edge_spacing_y);

  // 估计box
  BoundingBox<Dtype> bbox_estimate;
  reg->Regress(curr_search_region, target_pad, &bbox_estimate);

  // Unscale: 获取在裁剪patch中的坐标
  BoundingBox<Dtype> bbox_estimate_unscaled;
  bbox_estimate.Unscale(curr_search_region, &bbox_estimate_unscaled);

  // 原图重定位
  bbox_estimate_unscaled.Uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y, bbox_estimate_uncentered);

  // 是否需要显示结果
  if (show_tracking_) {
    ShowTracking(target_pad, curr_search_region, bbox_estimate);
  }

  image_prev_ = image_curr;
  bbox_prev_ = *bbox_estimate_uncentered;

  // TODO - replace with a motion model prediction?
  bbox_curr_init_ = *bbox_estimate_uncentered;
}

template <typename Dtype>
void TrackerBase<Dtype>::ShowTracking(const cv::Mat& target_pad, const cv::Mat& curr_search_region, const BoundingBox<Dtype>& bbox_estimate) const {
  cv::Mat target_resize;
  cv::resize(target_pad, target_resize, cv::Size(227, 227));

  cv::namedWindow("Target", cv::WINDOW_AUTOSIZE);
  cv::imshow("Target", target_resize);

  cv::Mat image_resize;
  cv::resize(curr_search_region, image_resize, cv::Size(227, 227));

  BoundingBox<Dtype> bbox_estimate_unscaled;
  bbox_estimate.Unscale(image_resize, &bbox_estimate_unscaled);

  // 显示结果: box
  cv::Mat image_with_box;
  image_resize.copyTo(image_with_box);
  bbox_estimate_unscaled.DrawBoundingBox(&image_with_box);

  cv::namedWindow("Estimate", cv::WINDOW_AUTOSIZE);
  cv::imshow("Estimate", image_with_box);
  cv::waitKey(0);
}

INSTANTIATE_CLASS(TrackerBase);

}
