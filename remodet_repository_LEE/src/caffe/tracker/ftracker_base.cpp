#include "caffe/tracker/ftracker_base.hpp"

#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/image_proc.hpp"

namespace caffe {

template <typename Dtype>
FTrackerBase<Dtype>::FTrackerBase(const std::string& network_proto,
                                  const std::string& caffe_model,
                                  const int gpu_id,
                                  const std::string& features,
                                  const int resized_width,
                                  const int resized_height) {
  // FeRoiMaker
  roi_maker_.reset(new FERoiMaker<Dtype>(network_proto,caffe_model,gpu_id,features,resized_width,resized_height));
  // resize layer
  LayerParameter layer_param;
  layer_param.set_name("roi_resize_layer");
  layer_param.set_type("RoiResize");
  RoiResizeParameter* roi_resize_param = layer_param.mutable_roi_resize_param();
  roi_resize_param->set_target_spatial_width(resized_width);
  roi_resize_param->set_target_spatial_height(resized_height);
  roi_resize_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  vector<Blob<Dtype>*> resize_bottom_vec;
  vector<Blob<Dtype>*> resize_top_vec;
  Blob<Dtype> bot_0(1,1,12,12);
  Blob<Dtype> bot_1(1,4,1,1);
  Blob<Dtype> top_0(1,1,resized_height,resized_width);
  resize_bottom_vec.push_back(&bot_0);
  resize_bottom_vec.push_back(&bot_1);
  resize_top_vec.push_back(&top_0);
  roi_resize_layer_->SetUp(resize_bottom_vec, resize_top_vec);
  // use basenet
  use_basenet_ = true;
}

template <typename Dtype>
FTrackerBase<Dtype>::FTrackerBase(const int resized_width, const int resized_height) {
  // caffe
  if (caffe::Caffe::mode() != caffe::Caffe::GPU) {
    LOG(FATAL) << "Error - caffe must run under GPU mode.";
  }
  LayerParameter layer_param;
  layer_param.set_name("roi_resize_layer");
  layer_param.set_type("RoiResize");
  RoiResizeParameter* roi_resize_param = layer_param.mutable_roi_resize_param();
  roi_resize_param->set_target_spatial_width(resized_width);
  roi_resize_param->set_target_spatial_height(resized_height);
  roi_resize_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  vector<Blob<Dtype>*> resize_bottom_vec;
  vector<Blob<Dtype>*> resize_top_vec;
  Blob<Dtype> bot_0(1,1,12,12);
  Blob<Dtype> bot_1(1,4,1,1);
  Blob<Dtype> top_0(1,1,resized_height,resized_width);
  resize_bottom_vec.push_back(&bot_0);
  resize_bottom_vec.push_back(&bot_1);
  resize_top_vec.push_back(&top_0);
  roi_resize_layer_->SetUp(resize_bottom_vec, resize_top_vec);
  // without basenet
  use_basenet_ = false;
}

/**
 * NOTE: bbox_gt: 实际坐标
 * bbox_prev_ -> 归一化坐标
 * bbox_curr_init_ -> 归一化坐标
 */
template <typename Dtype>
void FTrackerBase<Dtype>::Init(const cv::Mat& image, const BoundingBox<Dtype>& bbox_gt) {
  if (!use_basenet_) {
    LOG(FATAL) << "Error - Tracker is initialized without basenet.";
  }
  // get the first prev_fmap
  Blob<Dtype> resized_fmap;
  int image_width = image.cols;
  int image_height = image.rows;
  Dtype x_center = bbox_gt.get_center_x() / (Dtype)image_width;
  Dtype y_center = bbox_gt.get_center_y() / (Dtype)image_height;
  Dtype w = bbox_gt.compute_output_width() / (Dtype)image_width;
  Dtype h = bbox_gt.compute_output_height() / (Dtype)image_height;
  BoundingBox<Dtype> roi_bbox;
  roi_bbox.x1_ = x_center - w/2;
  roi_bbox.y1_ = y_center - h/2;
  roi_bbox.x2_ = x_center + w/2;
  roi_bbox.y2_ = y_center + h/2;
  roi_maker_->get_fmap(image, roi_bbox, &resized_fmap, &prev_fmap_);
  bbox_prev_.x1_ = (Dtype)bbox_gt.x1_ / (Dtype)image_width;
  bbox_prev_.y1_ = (Dtype)bbox_gt.y1_ / (Dtype)image_height;
  bbox_prev_.x2_ = (Dtype)bbox_gt.x2_ / (Dtype)image_width;
  bbox_prev_.y2_ = (Dtype)bbox_gt.y2_ / (Dtype)image_height;
  bbox_curr_init_ = bbox_prev_;
}

/**
 * NOTE: bbox_gt　-> 归一化坐标
 */
template <typename Dtype>
void FTrackerBase<Dtype>::Init(const Blob<Dtype>& fmap, const BoundingBox<Dtype>& bbox_gt) {
  if(use_basenet_) {
    LOG(FATAL) << "Error - Tracker is initialized with basenet, please use cv::Mat& to initialize.";
  }
  prev_fmap_.ReshapeLike(fmap);
  caffe_copy(prev_fmap_.count(),fmap.cpu_data(),prev_fmap_.mutable_cpu_data());
  bbox_prev_ = bbox_gt;
  // TODO - use a motion model?
  bbox_curr_init_ = bbox_gt;
}

/**
 * NOTE: bbox_estimate_uncentered -> 实际坐标
 */
template <typename Dtype>
void FTrackerBase<Dtype>::Tracking(const cv::Mat& image_curr, FRegressorBase<Dtype>* freg,
                           BoundingBox<Dtype>* bbox_estimate_uncentered) {
  if (!use_basenet_) {
    LOG(FATAL) << "Error - Tracker is initialized without basenet.";
  }
  // prev resized_map
  Blob<Dtype> bbox_blob(1,1,1,4);
  Blob<Dtype> prev_resized;
  Dtype* bbox_data = bbox_blob.mutable_cpu_data();
  // get prev roi
  bbox_data[0] = bbox_prev_.get_center_x() - bbox_prev_.get_width() * bbox_prev_.get_context_factor() / 2;
  bbox_data[1] = bbox_prev_.get_center_y() - bbox_prev_.get_height() * bbox_prev_.get_context_factor() / 2;
  bbox_data[2] = bbox_prev_.get_center_x() + bbox_prev_.get_width() * bbox_prev_.get_context_factor() / 2;
  bbox_data[3] = bbox_prev_.get_center_y() + bbox_prev_.get_height() * bbox_prev_.get_context_factor() / 2;
  vector<Blob<Dtype>* > resize_bottom_vec;
  vector<Blob<Dtype>* > resize_top_vec;
  resize_bottom_vec.push_back(&prev_fmap_);
  resize_bottom_vec.push_back(&bbox_blob);
  resize_top_vec.push_back(&prev_resized);
  roi_resize_layer_->Reshape(resize_bottom_vec, resize_top_vec);
  roi_resize_layer_->Forward(resize_bottom_vec, resize_top_vec);
  // curr resized_map
  // 当前帧的fmap直接保存到prev_fmap_
  Blob<Dtype> curr_resized;
  BoundingBox<Dtype> bbox_curr_roi;
  bbox_curr_roi.x1_ = bbox_data[0];
  bbox_curr_roi.y1_ = bbox_data[1];
  bbox_curr_roi.x2_ = bbox_data[2];
  bbox_curr_roi.y2_ = bbox_data[3];
  roi_maker_->get_fmap(image_curr, bbox_curr_roi, &curr_resized, &prev_fmap_);
  // 计算
  BoundingBox<Dtype> bbox_estimate;
  freg->Regress(curr_resized, prev_resized, &bbox_estimate);
  // 坐标转换
  Dtype xmin, ymin, xmax, ymax;
  Dtype x1 = bbox_curr_roi.x1_;
  Dtype y1 = bbox_curr_roi.y1_;
  Dtype w = bbox_curr_roi.x2_ - bbox_curr_roi.x1_;
  Dtype h = bbox_curr_roi.y2_ - bbox_curr_roi.y1_;
  // Uncenter
  xmin = x1 + bbox_estimate.x1_ * w;
  ymin = y1 + bbox_estimate.y1_ * h;
  xmax = x1 + bbox_estimate.x2_ * w;
  ymax = y1 + bbox_estimate.y2_ * h;
  xmin = std::max(std::min(xmin,Dtype(1)),Dtype(0));
  ymin = std::max(std::min(ymin,Dtype(1)),Dtype(0));
  xmax = std::max(std::min(xmax,Dtype(1)),Dtype(0));
  ymax = std::max(std::min(ymax,Dtype(1)),Dtype(0));
  // 保存到prev
  bbox_prev_.x1_ = xmin;
  bbox_prev_.y1_ = ymin;
  bbox_prev_.x2_ = xmax;
  bbox_prev_.y2_ = ymax;
  // 输出
  bbox_estimate_uncentered->x1_ = xmin * image_curr.cols;
  bbox_estimate_uncentered->y1_ = ymin * image_curr.rows;
  bbox_estimate_uncentered->x2_ = xmax * image_curr.cols;
  bbox_estimate_uncentered->y2_ = ymax * image_curr.rows;
  // TODO - replace with a motion model prediction?
  bbox_curr_init_ = bbox_prev_;
}

/**
 * NOTE: bbox_estimate_uncentered -> 归一化坐标
 */
template <typename Dtype>
void FTrackerBase<Dtype>::Tracking(const Blob<Dtype>& fmap, FRegressorBase<Dtype>* freg,
                                   BoundingBox<Dtype>* bbox_estimate_uncentered) {
  if (use_basenet_) {
    LOG(FATAL) << "Error - Tracker is initialized using basenet, please use cv::Mat& to track.";
  }
  // prev_fmap_
  Blob<Dtype> bbox_blob(1,1,1,4);
  Blob<Dtype> prev_resized;
  Dtype* bbox_data = bbox_blob.mutable_cpu_data();
  bbox_data[0] = bbox_prev_.get_center_x() - bbox_prev_.get_width() * bbox_prev_.get_context_factor() / 2;
  bbox_data[1] = bbox_prev_.get_center_y() - bbox_prev_.get_height() * bbox_prev_.get_context_factor() / 2;
  bbox_data[2] = bbox_prev_.get_center_x() + bbox_prev_.get_width() * bbox_prev_.get_context_factor() / 2;
  bbox_data[3] = bbox_prev_.get_center_y() + bbox_prev_.get_height() * bbox_prev_.get_context_factor() / 2;
  vector<Blob<Dtype>* > resize_bottom_vec;
  vector<Blob<Dtype>* > resize_top_vec;
  resize_bottom_vec.push_back(&prev_fmap_);
  resize_bottom_vec.push_back(&bbox_blob);
  resize_top_vec.push_back(&prev_resized);
  roi_resize_layer_->Reshape(resize_bottom_vec, resize_top_vec);
  roi_resize_layer_->Forward(resize_bottom_vec, resize_top_vec);
  // curr_fmap
  Blob<Dtype> curr_resized;
  Blob<Dtype> curr_fmap;
  curr_fmap.ReshapeLike(fmap);
  curr_fmap.ShareData(fmap);
  resize_bottom_vec.clear();
  resize_top_vec.clear();
  resize_bottom_vec.push_back(&curr_fmap);
  resize_bottom_vec.push_back(&bbox_blob);
  resize_top_vec.push_back(&curr_resized);
  roi_resize_layer_->Reshape(resize_bottom_vec, resize_top_vec);
  roi_resize_layer_->Forward(resize_bottom_vec, resize_top_vec);
  // 计算
  BoundingBox<Dtype> bbox_estimate;
  freg->Regress(curr_resized, prev_resized, &bbox_estimate);
  // 坐标转换
  Dtype xmin, ymin, xmax, ymax;
  Dtype x1 = bbox_data[0];
  Dtype y1 = bbox_data[1];
  Dtype w = bbox_data[2] - bbox_data[0];
  Dtype h = bbox_data[3] - bbox_data[1];
  xmin = x1 + bbox_estimate.x1_ * w;
  ymin = y1 + bbox_estimate.y1_ * h;
  xmax = x1 + bbox_estimate.x2_ * w;
  ymax = y1 + bbox_estimate.y2_ * h;
  xmin = std::max(std::min(xmin,Dtype(1)),Dtype(0));
  ymin = std::max(std::min(ymin,Dtype(1)),Dtype(0));
  xmax = std::max(std::min(xmax,Dtype(1)),Dtype(0));
  ymax = std::max(std::min(ymax,Dtype(1)),Dtype(0));
  // 保存到prev　／　curr_init
  bbox_prev_.x1_ = xmin;
  bbox_prev_.y1_ = ymin;
  bbox_prev_.x2_ = xmax;
  bbox_prev_.y2_ = ymax;
  bbox_curr_init_ = bbox_prev_;
  // 保存fmap
  CHECK_EQ(prev_fmap_.count(), fmap.count());
  caffe_copy(prev_fmap_.count(), fmap.cpu_data(), prev_fmap_.mutable_cpu_data());
  // 输出
  bbox_estimate_uncentered->x1_ = xmin;
  bbox_estimate_uncentered->y1_ = ymin;
  bbox_estimate_uncentered->x2_ = xmax;
  bbox_estimate_uncentered->y2_ = ymax;
}

INSTANTIATE_CLASS(FTrackerBase);

}
