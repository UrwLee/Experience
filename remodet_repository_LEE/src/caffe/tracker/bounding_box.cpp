#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/image_proc.hpp"
#include "caffe/tracker/basic.hpp"

#include <cstdio>

namespace caffe {

using namespace std;

// How much context to pad the image and target with (relative to the
// bounding box size).
const float kContextFactor = 2;

// Factor by which to scale the bounding box coordinates, based on the
// neural network default output range.
const float kScaleFactor = 10;

// If true, the neural network will estimate the bounding box corners: (x1, y1, x2, y2)
// If false, the neural network will estimate the bounding box center location and size: (center_x, center_y, width, height)
const bool use_coordinates_output = true;

template<typename Dtype>
BoundingBox<Dtype>::BoundingBox(): scale_factor_(kScaleFactor) {
}

template<typename Dtype>
BoundingBox<Dtype>::BoundingBox(const std::vector<Dtype>& bounding_box): scale_factor_(kScaleFactor) {
  if (bounding_box.size() != 4) {
    LOG(FATAL) << "Error - bounding box vector has "
               << bounding_box.size() << " elements.";
  }
  if (use_coordinates_output) {
    x1_ = bounding_box[0];
    y1_ = bounding_box[1];
    x2_ = bounding_box[2];
    y2_ = bounding_box[3];
  } else {
    const Dtype center_x = bounding_box[0];
    const Dtype center_y = bounding_box[1];
    const Dtype width = bounding_box[2];
    const Dtype height = bounding_box[3];
    x1_ = center_x - width / 2;
    y1_ = center_y - height / 2;
    x2_ = center_x + width / 2;
    y2_ = center_y + height / 2;
  }
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::get_context_factor() {
  return (Dtype)kContextFactor;
}

template<typename Dtype>
void BoundingBox<Dtype>::GetVector(std::vector<Dtype>* bounding_box) const {
  if (use_coordinates_output) {
    bounding_box->push_back(x1_);
    bounding_box->push_back(y1_);
    bounding_box->push_back(x2_);
    bounding_box->push_back(y2_);
  } else {
    bounding_box->push_back(get_center_x());
    bounding_box->push_back(get_center_y());
    bounding_box->push_back(get_width());
    bounding_box->push_back(get_height());
  }
}

template<typename Dtype>
void BoundingBox<Dtype>::Print() const {
   LOG(INFO) << "Bounding box: x,y: "
             << x1_ << ", " << y1_ << ", "
             << x2_ << ", " << y2_ << ", "
             << "w,h: "
             << get_width() << ", " << get_height() << " .";
}

template<typename Dtype>
void BoundingBox<Dtype>::Scale(const cv::Mat& image, BoundingBox<Dtype>* bbox_scaled) const {
  *bbox_scaled = *this;

  const int width = image.cols;
  const int height = image.rows;

  // Scale the bounding box so that the coordinates range from 0 to 1.
  bbox_scaled->x1_ /= width;
  bbox_scaled->y1_ /= height;
  bbox_scaled->x2_ /= width;
  bbox_scaled->y2_ /= height;

  // Scale the bounding box so that the coordinates range from 0 to scale_factor_.
  bbox_scaled->x1_ *= (Dtype)scale_factor_;
  bbox_scaled->x2_ *= (Dtype)scale_factor_;
  bbox_scaled->y1_ *= (Dtype)scale_factor_;
  bbox_scaled->y2_ *= (Dtype)scale_factor_;
}

template<typename Dtype>
void BoundingBox<Dtype>::Unscale(const cv::Mat& image, BoundingBox<Dtype>* bbox_unscaled) const {
  *bbox_unscaled = *this;

  const int image_width = image.cols;
  const int image_height = image.rows;

  // Unscale the bounding box so that the coordinates range from 0 to 1.
  bbox_unscaled->x1_ /= (Dtype)scale_factor_;
  bbox_unscaled->x2_ /= (Dtype)scale_factor_;
  bbox_unscaled->y1_ /= (Dtype)scale_factor_;
  bbox_unscaled->y2_ /= (Dtype)scale_factor_;

  // Unscale the bounding box so that the coordinates match the original image coordinates
  // (undoing the effect from the Scale method).
  bbox_unscaled->x1_ *= image_width;
  bbox_unscaled->y1_ *= image_height;
  bbox_unscaled->x2_ *= image_width;
  bbox_unscaled->y2_ *= image_height;
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_output_width() const {
  // Get the bounding box width.
  const Dtype bbox_width = (x2_ - x1_);
  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const Dtype output_width = kContextFactor * bbox_width;
  // Ensure that the output width is at least 1 pixel.
  return std::max((Dtype)1.0, output_width);
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_output_height() const {
  // Get the bounding box height.
  const Dtype bbox_height = (y2_ - y1_);
  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const Dtype output_height = kContextFactor * bbox_height;
  // Ensure that the output height is at least 1 pixel.
  return std::max((Dtype)1.0, output_height);
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::get_center_x() const {
  // Compute the bounding box center x-coordinate.
  return (x1_ + x2_) / 2;
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::get_center_y() const {
  // Compute the bounding box center y-coordinate.
  return (y1_ + y2_) / 2;
}

template<typename Dtype>
void BoundingBox<Dtype>::Recenter(const BoundingBox<Dtype>& search_location,
              const Dtype edge_spacing_x, const Dtype edge_spacing_y,
              BoundingBox<Dtype>* bbox_gt_recentered) const {
  // Location of bounding box relative to the focused image and edge_spacing.
  bbox_gt_recentered->x1_ = x1_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y1_ = y1_ - search_location.y1_ + edge_spacing_y;
  bbox_gt_recentered->x2_ = x2_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y2_ = y2_ - search_location.y1_ + edge_spacing_y;
}

template<typename Dtype>
void BoundingBox<Dtype>::Uncenter(const cv::Mat& raw_image,
                           const BoundingBox<Dtype>& search_location,
                           const Dtype edge_spacing_x, const Dtype edge_spacing_y,
                           BoundingBox<Dtype>* bbox_uncentered) const {
  // Undo the effect of Recenter.
  bbox_uncentered->x1_ = std::max((Dtype)0.0, x1_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y1_ = std::max((Dtype)0.0, y1_ + search_location.y1_ - edge_spacing_y);
  bbox_uncentered->x2_ = std::min(static_cast<Dtype>(raw_image.cols), x2_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y2_ = std::min(static_cast<Dtype>(raw_image.rows), y2_ + search_location.y1_ - edge_spacing_y);
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::edge_spacing_x() const {
  const Dtype output_width = compute_output_width();
  const Dtype bbox_center_x = get_center_x();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_width / 2 < bbox_center_x, but if the crop is near the left
  // edge of the image then we would have output_width / 2 > bbox_center_x, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max((Dtype)0.0, output_width / 2 - bbox_center_x);
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::edge_spacing_y() const {
  const Dtype output_height = compute_output_height();
  const Dtype bbox_center_y = get_center_y();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_height / 2 < bbox_center_y, but if the crop is near the bottom
  // edge of the image then we would have output_height / 2 > bbox_center_y, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max((Dtype)0.0, output_height / 2 - bbox_center_y);
}

template<typename Dtype>
void BoundingBox<Dtype>::Draw(const int r, const int g, const int b,
                       cv::Mat* image) const {
  // Get the top-left point.
  const cv::Point point1(x1_, y1_);

  // Get the bottom-rigth point.
  const cv::Point point2(x2_, y2_);

  // Get the selected color.
  const cv::Scalar box_color(b, g, r);

  // Draw a rectangle corresponding to this bbox with the given color.
  const int thickness = 2;
  cv::rectangle(*image, point1, point2, box_color, thickness);
}

template<typename Dtype>
void BoundingBox<Dtype>::DrawNorm(const int r, const int g, const int b,
                       cv::Mat* image) const {
  // Get the top-left point.
  const cv::Point point1(x1_ * image->cols, y1_ * image->rows);

  // Get the bottom-rigth point.
  const cv::Point point2(x2_ * image->cols, y2_ * image->rows);

  // Get the selected color.
  const cv::Scalar box_color(b, g, r);

  // Draw a rectangle corresponding to this bbox with the given color.
  const int thickness = 1;
  cv::rectangle(*image, point1, point2, box_color, thickness);
}
template<typename Dtype>
void BoundingBox<Dtype>::DrawBoundingBox(cv::Mat* image) const {
  // Draw a white bounding box on the image.
  Draw(0, 255, 0, image);
}
template<typename Dtype>
void BoundingBox<Dtype>::DrawBoundingBoxNorm(cv::Mat* image) const {
  // Draw a white bounding box on the image.
  DrawNorm(0, 255, 0, image);
}
template<typename Dtype>
void BoundingBox<Dtype>::Shift(const cv::Mat& image,
                        const Dtype lambda_scale_frac,
                        const Dtype lambda_shift_frac,
                        const Dtype min_scale, const Dtype max_scale,
                        const bool shift_motion_model,
                        BoundingBox<Dtype>* bbox_rand) const {
  // 获取box的长和宽
  const Dtype width = get_width();
  const Dtype height = get_height();
  // 获取box的中心位置
  Dtype center_x = get_center_x();
  Dtype center_y = get_center_y();

  // Number of times to try shifting the bounding box.
  const int kMaxNumTries = 10;

  // Sample a width scaling factor for the new crop window, thresholding the scale to stay within a reasonable window.
  Dtype new_width = -1;
  int num_tries_width = 0;
  // 获取随机宽度
  while ((new_width < 0 || new_width > image.cols - 1) && num_tries_width < kMaxNumTries) {
    // Sample.
    // 宽度变化因子
    Dtype width_scale_factor;
    if (shift_motion_model) {
      width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)));
    } else {
      const Dtype rand_num = sample_rand_uniform();
      width_scale_factor = rand_num * (max_scale - min_scale) + min_scale;
    }
    // Expand width by scaling factor.
    new_width = width * (1 + width_scale_factor);
    // Ensure that width stays within valid limits.
    new_width = max((Dtype)1.0, min(static_cast<Dtype>(image.cols - 1), new_width));
    num_tries_width++;
  }

  // Find a height scaling factor for the new crop window, thresholding the scale to stay within a reasonable window.
  Dtype new_height = -1;
  int num_tries_height = 0;
  // 获取随机高度
  while ((new_height < 0 || new_height > image.rows - 1) && num_tries_height < kMaxNumTries) {
    // Sample.
    Dtype height_scale_factor;
    if (shift_motion_model) {
      height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)));
    } else {
      const Dtype rand_num = sample_rand_uniform();
      height_scale_factor = rand_num * (max_scale - min_scale) + min_scale;
    }
    // Expand height by scaling factor.
    new_height = height * (1 + height_scale_factor);
    // Ensure that height stays within valid limits.
    new_height = max((Dtype)1.0, min(static_cast<Dtype>(image.rows - 1), new_height));
    num_tries_height++;
  }

  // Find a random x translation for the new crop window.
  // 获取随机偏移dx
  bool first_time_x = true;
  Dtype new_center_x = -1;
  int num_tries_x = 0;
  while ((first_time_x ||
         // 确保新的中心位于原来上下文窗口之内
         new_center_x < center_x - width * kContextFactor / 2 ||
         new_center_x > center_x + width * kContextFactor / 2 ||
         // 确保新的目标窗口要位于原来的图像之中
         new_center_x - new_width / 2 < 0 ||
         new_center_x + new_width / 2 > image.cols)
         && num_tries_x < kMaxNumTries) {
    // Sample.
    Dtype new_x_temp;
    if (shift_motion_model) {
      new_x_temp = center_x + width * sample_exp_two_sided(lambda_shift_frac);
    } else {
      const Dtype rand_num = sample_rand_uniform();
      new_x_temp = center_x + rand_num * (2 * new_width) - new_width;
    }
    // 确保新窗口位于图像区域内
    // x - new_width/2 > 0
    // w - x > new_width/2
    // nw/2 < x < w-nw/2
    new_center_x = min(image.cols - new_width / 2, max(new_width / 2, new_x_temp));
    first_time_x = false;
    num_tries_x++;
  }

  // Find a random y translation for the new crop window.
  bool first_time_y = true;
  Dtype new_center_y = -1;
  int num_tries_y = 0;
  while ((first_time_y ||
         new_center_y < center_y - height * kContextFactor / 2 ||
         new_center_y > center_y + height * kContextFactor / 2  ||
         new_center_y - new_height / 2 < 0 ||
         new_center_y + new_height / 2 > image.rows)
         && num_tries_y < kMaxNumTries) {
    // Sample.
    Dtype new_y_temp;
    if (shift_motion_model) {
      new_y_temp = center_y + height * sample_exp_two_sided(lambda_shift_frac);
    } else {
      const Dtype rand_num = sample_rand_uniform();
      new_y_temp = center_y + rand_num * (2 * new_height) - new_height;
    }
    // Make sure that the window stays within the image.
    new_center_y = min(image.rows - new_height / 2, max(new_height / 2, new_y_temp));
    first_time_y = false;
    num_tries_y++;
  }

  // 产生的新的随机窗口
  bbox_rand->x1_ = new_center_x - new_width / 2;
  bbox_rand->x2_ = new_center_x + new_width / 2;
  bbox_rand->y1_ = new_center_y - new_height / 2;
  bbox_rand->y2_ = new_center_y + new_height / 2;
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_intersection(const BoundingBox<Dtype>& bbox) const {
  const Dtype area = std::max((Dtype)0.0, std::min(x2_, bbox.x2_) - std::max(x1_, bbox.x1_)) * std::max((Dtype)0.0, std::min(y2_, bbox.y2_) - std::max(y1_, bbox.y1_));
  return area;
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_iou(const BoundingBox<Dtype>& bbox) const {
  Dtype inter = compute_intersection(bbox);
  Dtype area_this = compute_area();
  Dtype area_that = bbox.compute_area();
  // LOG(INFO)<<"hzw area_this"<<area_this;
  // LOG(INFO)<<"hzw area_that"<<area_that;  
  Dtype iou = inter / (area_this + area_that - inter);
  iou = std::max(iou, (Dtype)0);
  return iou;
}

template<typename Dtype>
vector<Dtype> BoundingBox<Dtype>::compute_iou_expdist(const BoundingBox<Dtype>& bbox, Dtype sigma) const {
  vector<Dtype> v;
  Dtype inter = compute_intersection(bbox);
  Dtype area_this = compute_area();
  Dtype area_that = bbox.compute_area();
  // LOG(INFO)<<"hzw area_this"<<area_this;
  // LOG(INFO)<<"hzw area_that"<<area_that;  
  Dtype iou = inter / (area_this + area_that - inter);
  iou = std::max(iou, (Dtype)0);

  Dtype center_x_1 = get_center_x();
  Dtype center_y_1 = get_center_y();
  Dtype center_x_2 = bbox.get_center_x();
  Dtype center_y_2 = bbox.get_center_y();
  Dtype diffx = center_x_1 - center_x_2;
  Dtype diffy = center_y_1 - center_y_2;
  Dtype dist = diffx*diffx + diffy*diffy;
  Dtype expdst = std::exp(-dist/2/sigma/sigma);
  v.push_back(iou);
  v.push_back(expdst);
  return v;
}


template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_coverage(const BoundingBox<Dtype>& bbox) const {
  Dtype inter = compute_intersection(bbox);
  Dtype area_this = compute_area();
  Dtype coverage = inter / area_this;
  coverage = std::max(coverage, (Dtype)0);
  return coverage; 
}

template<typename Dtype>
vector<Dtype>  BoundingBox<Dtype>::compute_iou_coverage(const BoundingBox<Dtype>& bbox) const {
  vector<Dtype> v;
  Dtype inter = compute_intersection(bbox);
  Dtype area_that = bbox.compute_area();
  Dtype area_this = compute_area();
  Dtype coverage = inter / area_this;
  coverage = std::max(coverage, (Dtype)0);
  Dtype iou = inter / (area_this + area_that - inter);
  iou = std::max(iou, (Dtype)0);
  v.push_back(iou);
  v.push_back(coverage);
  return v; 
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_obj_coverage(const BoundingBox<Dtype>& bbox) const {
  Dtype inter = compute_intersection(bbox);
  Dtype area_that = bbox.compute_area();
  Dtype coverage = inter / area_that;
  coverage = std::max(coverage, (Dtype)0);
  return coverage;
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::project_bbox(const BoundingBox<Dtype>& bbox, BoundingBox<Dtype>* proj_bbox) const {
  if (x1_ >= bbox.x2_ || x2_ <= bbox.x1_ || y1_ >= bbox.y2_ || y2_ <= bbox.y1_) {return 0;}
  proj_bbox->x1_ = (x1_ - bbox.x1_) / bbox.get_width();
  proj_bbox->y1_ = (y1_ - bbox.y1_) / bbox.get_height();
  proj_bbox->x2_ = (x2_ - bbox.x1_) / bbox.get_width();
  proj_bbox->y2_ = (y2_ - bbox.y1_) / bbox.get_height();
  proj_bbox->x1_ = std::min(std::max(proj_bbox->x1_, (Dtype)0), (Dtype)1);
  proj_bbox->y1_ = std::min(std::max(proj_bbox->y1_, (Dtype)0), (Dtype)1);
  proj_bbox->x2_ = std::min(std::max(proj_bbox->x2_, (Dtype)0), (Dtype)1);
  proj_bbox->y2_ = std::min(std::max(proj_bbox->y2_, (Dtype)0), (Dtype)1);
  if (proj_bbox->get_width() * proj_bbox->get_height() <= 0) {return 0;}
  return compute_coverage(bbox);
}

template<typename Dtype>
Dtype BoundingBox<Dtype>::compute_area() const {
  return get_width() * get_height();
}

template <typename Dtype>
void BoundingBox<Dtype>::clip(const Dtype min, const Dtype max) {
  x1_ = std::min(std::max(x1_, min), max);
  y1_ = std::min(std::max(y1_, min), max);
  x2_ = std::min(std::max(x2_, min), max);
  y2_ = std::min(std::max(y2_, min), max);
}

template <typename Dtype>
void BoundingBox<Dtype>::clip() {
  clip(Dtype(0),Dtype(1));
}

INSTANTIATE_CLASS(BoundingBox);
}
