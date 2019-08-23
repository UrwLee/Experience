#ifndef CAFFE_TRACKER_IMAGE_PROC_H
#define CAFFE_TRACKER_IMAGE_PROC_H

#include "caffe/tracker/bounding_box.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

/**
 * 该头文件提供了对ROI进行裁剪以及生成对应Patch的方法。
 * 包含：随机位移后的Patch (模拟Curr)
 */

namespace caffe {

// Functions to process images for tracking.
// Crop the image at the bounding box location, plus some additional padding.
// To account for edge effects, we use a black background for space beyond the border
// of the image.
template<typename Dtype>
void CropPadImage(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image);

template<typename Dtype>
void CropPadImage(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
                  BoundingBox<Dtype>* pad_image_location, Dtype* edge_spacing_x, Dtype* edge_spacing_y);

// Compute the location of the cropped image, which is centered on the bounding box center
// but has a size given by (output_width, output_height) to account for additional padding.
// The cropped image location is also limited by the edge of the image.
template<typename Dtype>
void ComputeCropPadImageLocation(const BoundingBox<Dtype>& bbox_tight, const cv::Mat& image, BoundingBox<Dtype>* pad_image_location);

}

#endif
