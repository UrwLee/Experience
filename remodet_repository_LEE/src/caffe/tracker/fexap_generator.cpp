#include "caffe/tracker/fexap_generator.hpp"
#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/image_proc.hpp"
#include "caffe/tracker/bounding_box.hpp"

#include <string>

namespace caffe {

using std::string;

const bool shift_motion_model = true;
const float ScaleFactor = 1;

template <typename Dtype>
FExampleGenerator<Dtype>::FExampleGenerator(const Dtype lambda_shift,
                                            const Dtype lambda_scale,
                                            const Dtype min_scale,
                                            const Dtype max_scale,
                                            const std::string& network_proto,
                                            const std::string& caffe_model,
                                            const int gpu_id,
                                            const std::string& features,
                                            const int resized_width,
                                            const int resized_height)
  : lambda_shift_(lambda_shift), lambda_scale_(lambda_scale), min_scale_(min_scale), max_scale_(max_scale) {
  roi_maker_.reset(new FERoiMaker<Dtype>(network_proto,caffe_model,gpu_id,features,resized_width,resized_height));
}

template <typename Dtype>
void FExampleGenerator<Dtype>::Init(const Dtype lambda_shift, const Dtype lambda_scale,
                                    const Dtype min_scale, const Dtype max_scale) {
  lambda_shift_ = lambda_shift;
  lambda_scale_ = lambda_scale;
  min_scale_ = min_scale;
  max_scale_ = max_scale;
}

/**
 * bbox_prev/bbox_curr -> 实际坐标
 */
template <typename Dtype>
void FExampleGenerator<Dtype>::Reset(const BoundingBox<Dtype>& bbox_prev,
                                     const BoundingBox<Dtype>& bbox_curr,
                                     const cv::Mat& image_prev,
                                     const cv::Mat& image_curr) {
  // cal the roi box (normalized)
  int image_width = image_prev.cols;
  int image_height = image_prev.rows;
  Dtype x_center = bbox_prev.get_center_x() / (Dtype)image_width;
  Dtype y_center = bbox_prev.get_center_y() / (Dtype)image_height;
  Dtype w = bbox_prev.compute_output_width() / (Dtype)image_width;
  Dtype h = bbox_prev.compute_output_height() / (Dtype)image_height;
  BoundingBox<Dtype> roi_bbox;
  roi_bbox.x1_ = x_center - w/2;
  roi_bbox.y1_ = y_center - h/2;
  roi_bbox.x2_ = x_center + w/2;
  roi_bbox.y2_ = y_center + h/2;
  roi_maker_->get_features(image_prev, roi_bbox, &prev_f_);
  image_curr_ = image_curr;
  bbox_curr_gt_ = bbox_curr;
  bbox_prev_gt_ = bbox_prev;
}

template <typename Dtype>
void FExampleGenerator<Dtype>::MakeTrainingExamples(
                          const int num_examples,
                          std::vector<boost::shared_ptr<Blob<Dtype> > >* curr,
                          std::vector<boost::shared_ptr<Blob<Dtype> > >* prev,
                          std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled) {
  if (num_examples == 0) return;
  for (int i = 0; i < num_examples; ++i) {
    boost::shared_ptr<Blob<Dtype> > curr_f(new Blob<Dtype>());
    boost::shared_ptr<Blob<Dtype> > prev_f(new Blob<Dtype>());
    BoundingBox<Dtype> bbox_gt_scaled_f;
    MakeTrainingExampleBBShift(curr_f.get(), prev_f.get(), &bbox_gt_scaled_f);
    curr->push_back(curr_f);
    prev->push_back(prev_f);
    bboxes_gt_scaled->push_back(bbox_gt_scaled_f);
  }
}

/**
 * bbox_gt_scaled -> 归一化坐标
 */
template <typename Dtype>
void FExampleGenerator<Dtype>::MakeTrueExample(Blob<Dtype>* curr,Blob<Dtype>* prev,
                                               BoundingBox<Dtype>* bbox_gt_scaled) {
  prev->ReshapeLike(prev_f_);
  prev->ShareData(prev_f_);
  // TODO - use a motion model?
  const BoundingBox<Dtype>& curr_prior_tight = bbox_prev_gt_;
  int image_width = image_curr_.cols;
  int image_height = image_curr_.rows;
  Dtype x_center = curr_prior_tight.get_center_x() / (Dtype)image_width;
  Dtype y_center = curr_prior_tight.get_center_y() / (Dtype)image_height;
  Dtype w = curr_prior_tight.compute_output_width() / (Dtype)image_width;
  Dtype h = curr_prior_tight.compute_output_height() / (Dtype)image_height;
  BoundingBox<Dtype> roi_bbox;
  roi_bbox.x1_ = x_center - w/2;
  roi_bbox.y1_ = y_center - h/2;
  roi_bbox.x2_ = x_center + w/2;
  roi_bbox.y2_ = y_center + h/2;
  roi_maker_->get_features(image_curr_, roi_bbox, curr);
  // get gt {x1_,y1_,x2_,y2_}
  Dtype gt_x1 = (Dtype)bbox_curr_gt_.x1_ / (Dtype)image_width;
  Dtype gt_y1 = (Dtype)bbox_curr_gt_.y1_ / (Dtype)image_height;
  Dtype gt_x2 = (Dtype)bbox_curr_gt_.x2_ / (Dtype)image_width;
  Dtype gt_y2 = (Dtype)bbox_curr_gt_.y2_ / (Dtype)image_height;
  // Recenter
  gt_x1 -= roi_bbox.x1_;
  gt_y1 -= roi_bbox.y1_;
  gt_x2 -= roi_bbox.x1_;
  gt_y2 -= roi_bbox.y1_;
  // Normal
  gt_x1 = gt_x1 / (roi_bbox.x2_ - roi_bbox.x1_);
  gt_y1 = gt_y1 / (roi_bbox.y2_ - roi_bbox.y1_);
  gt_x2 = gt_x2 / (roi_bbox.x2_ - roi_bbox.x1_);
  gt_y2 = gt_y2 / (roi_bbox.y2_ - roi_bbox.y1_);
  // Scale
  gt_x1 *= ScaleFactor;
  gt_y1 *= ScaleFactor;
  gt_x2 *= ScaleFactor;
  gt_y2 *= ScaleFactor;
  // output
  bbox_gt_scaled->x1_ = gt_x1;
  bbox_gt_scaled->y1_ = gt_y1;
  bbox_gt_scaled->x2_ = gt_x2;
  bbox_gt_scaled->y2_ = gt_y2;
}

template <typename Dtype>
void FExampleGenerator<Dtype>::get_default_bb_params(BBParams<Dtype>* default_params) {
  default_params->lambda_scale = lambda_scale_;
  default_params->lambda_shift = lambda_shift_;
  default_params->min_scale = min_scale_;
  default_params->max_scale = max_scale_;
}

template <typename Dtype>
void FExampleGenerator<Dtype>::MakeTrainingExampleBBShift(Blob<Dtype>* curr,
                                                  Blob<Dtype>* prev,
                                                  BoundingBox<Dtype>* bbox_gt_scaled){

  BBParams<Dtype> default_bb_params;
  get_default_bb_params(&default_bb_params);
  MakeTrainingExampleBBShift(default_bb_params, curr, prev, bbox_gt_scaled);
}

template <typename Dtype>
void FExampleGenerator<Dtype>::MakeTrainingExampleBBShift(const BBParams<Dtype>& bbparams,
                                                  Blob<Dtype>* curr,
                                                  Blob<Dtype>* prev,
                                                  BoundingBox<Dtype>* bbox_gt_scaled){
  // 将prev裁剪后的pad图像定义为target_pad
  prev->ReshapeLike(prev_f_);
  prev->ShareData(prev_f_);
  BoundingBox<Dtype> bbox_curr_shift;
  // 将当前帧的box进行随机增强,获得新的box
  bbox_curr_gt_.Shift(image_curr_, bbparams.lambda_scale, bbparams.lambda_shift,
                      bbparams.min_scale, bbparams.max_scale,
                      shift_motion_model,
                      &bbox_curr_shift);
  // fe
  int image_width = image_curr_.cols;
  int image_height = image_curr_.rows;
  Dtype x_center = bbox_curr_shift.get_center_x() / (Dtype)image_width;
  Dtype y_center = bbox_curr_shift.get_center_y() / (Dtype)image_height;
  Dtype w = bbox_curr_shift.compute_output_width() / (Dtype)image_width;
  Dtype h = bbox_curr_shift.compute_output_height() / (Dtype)image_height;
  BoundingBox<Dtype> roi_bbox;
  roi_bbox.x1_ = x_center - w/2;
  roi_bbox.y1_ = y_center - h/2;
  roi_bbox.x2_ = x_center + w/2;
  roi_bbox.y2_ = y_center + h/2;
  roi_maker_->get_features(image_curr_, roi_bbox, curr);
  // get gt
  Dtype gt_x1 = (Dtype)bbox_curr_gt_.x1_ / (Dtype)image_width;
  Dtype gt_y1 = (Dtype)bbox_curr_gt_.y1_ / (Dtype)image_height;
  Dtype gt_x2 = (Dtype)bbox_curr_gt_.x2_ / (Dtype)image_width;
  Dtype gt_y2 = (Dtype)bbox_curr_gt_.y2_ / (Dtype)image_height;
  // Recenter
  gt_x1 -= roi_bbox.x1_;
  gt_y1 -= roi_bbox.y1_;
  gt_x2 -= roi_bbox.x1_;
  gt_y2 -= roi_bbox.y1_;
  // Normal
  gt_x1 = gt_x1 / (roi_bbox.x2_ - roi_bbox.x1_);
  gt_y1 = gt_y1 / (roi_bbox.y2_ - roi_bbox.y1_);
  gt_x2 = gt_x2 / (roi_bbox.x2_ - roi_bbox.x1_);
  gt_y2 = gt_y2 / (roi_bbox.y2_ - roi_bbox.y1_);
  // Scale
  gt_x1 *= ScaleFactor;
  gt_y1 *= ScaleFactor;
  gt_x2 *= ScaleFactor;
  gt_y2 *= ScaleFactor;
  // output
  bbox_gt_scaled->x1_ = gt_x1;
  bbox_gt_scaled->y1_ = gt_y1;
  bbox_gt_scaled->x2_ = gt_x2;
  bbox_gt_scaled->y2_ = gt_y2;
}

INSTANTIATE_CLASS(FExampleGenerator);
}
