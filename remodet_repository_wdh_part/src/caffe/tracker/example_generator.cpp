#include "caffe/tracker/example_generator.hpp"
#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/image_proc.hpp"
#include "caffe/tracker/bounding_box.hpp"

#include <string>

namespace caffe {

using std::string;

// Choose whether to shift boxes using the motion model or using a uniform distribution.
const bool shift_motion_model = true;

template <typename Dtype>
ExampleGenerator<Dtype>::ExampleGenerator(const Dtype lambda_shift,
                                         const Dtype lambda_scale,
                                         const Dtype min_scale,
                                         const Dtype max_scale)
  : lambda_shift_(lambda_shift), lambda_scale_(lambda_scale), min_scale_(min_scale), max_scale_(max_scale) {
}

template <typename Dtype>
void ExampleGenerator<Dtype>::Init(const Dtype lambda_shift, const Dtype lambda_scale,
                                    const Dtype min_scale, const Dtype max_scale) {
  lambda_shift_ = lambda_shift;
  lambda_scale_ = lambda_scale;
  min_scale_ = min_scale;
  max_scale_ = max_scale;
}

template <typename Dtype>
void ExampleGenerator<Dtype>::Reset(const BoundingBox<Dtype>& bbox_prev,
                             const BoundingBox<Dtype>& bbox_curr,
                             const cv::Mat& image_prev,
                             const cv::Mat& image_curr) {
  CropPadImage(bbox_prev, image_prev, &target_pad_);
  image_curr_ = image_curr;
  bbox_curr_gt_ = bbox_curr;
  bbox_prev_gt_ = bbox_prev;
}

template <typename Dtype>
void ExampleGenerator<Dtype>::MakeTrainingExamples(const int num_examples,
                                            std::vector<cv::Mat>* images,
                                            std::vector<cv::Mat>* targets,
                                            std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled) {
  // 创建N个样本对
  for (int i = 0; i < num_examples; ++i) {
    cv::Mat image_rand_focus;
    cv::Mat target_pad;
    BoundingBox<Dtype> bbox_gt_scaled;
    // 生成训练样本
    MakeTrainingExampleBBShift(&image_rand_focus, &target_pad, &bbox_gt_scaled);
    images->push_back(image_rand_focus);
    targets->push_back(target_pad);
    bboxes_gt_scaled->push_back(bbox_gt_scaled);
  }
}

template <typename Dtype>
void ExampleGenerator<Dtype>::MakeTrueExample(cv::Mat* curr_search_region,
                                       cv::Mat* target_pad,
                                       BoundingBox<Dtype>* bbox_gt_scaled) const {
  *target_pad = target_pad_;

  // TODO - use a motion model?
  const BoundingBox<Dtype>& curr_prior_tight = bbox_prev_gt_;

  BoundingBox<Dtype> curr_search_location;
  Dtype edge_spacing_x, edge_spacing_y;
  // 在当前帧的历史位置处进行裁剪
  CropPadImage(curr_prior_tight, image_curr_, curr_search_region, &curr_search_location, &edge_spacing_x, &edge_spacing_y);
  BoundingBox<Dtype> bbox_gt_recentered;
  // 将gt在裁剪后的patch中进行定位
  bbox_curr_gt_.Recenter(curr_search_location, edge_spacing_x, edge_spacing_y, &bbox_gt_recentered);
  // 获取输出结果
  bbox_gt_recentered.Scale(*curr_search_region, bbox_gt_scaled);
}

template <typename Dtype>
void ExampleGenerator<Dtype>::get_default_bb_params(BBParams<Dtype>* default_params) const {
  default_params->lambda_scale = lambda_scale_;
  default_params->lambda_shift = lambda_shift_;
  default_params->min_scale = min_scale_;
  default_params->max_scale = max_scale_;
}

template <typename Dtype>
void ExampleGenerator<Dtype>::MakeTrainingExampleBBShift(cv::Mat* image_rand_focus,
                                                  cv::Mat* target_pad,
                                                  BoundingBox<Dtype>* bbox_gt_scaled) const {

  BBParams<Dtype> default_bb_params;
  get_default_bb_params(&default_bb_params);

  const bool visualize_example = false;
  MakeTrainingExampleBBShift(visualize_example, default_bb_params,
                             image_rand_focus, target_pad, bbox_gt_scaled);

}

template <typename Dtype>
void ExampleGenerator<Dtype>::MakeTrainingExampleBBShift(
    const bool visualize_example, cv::Mat* image_rand_focus,
    cv::Mat* target_pad, BoundingBox<Dtype>* bbox_gt_scaled) const {
  BBParams<Dtype> default_bb_params;
  get_default_bb_params(&default_bb_params);
  MakeTrainingExampleBBShift(visualize_example, default_bb_params,
                             image_rand_focus, target_pad, bbox_gt_scaled);
}

template <typename Dtype>
void ExampleGenerator<Dtype>::MakeTrainingExampleBBShift(const bool visualize_example,
                                                  const BBParams<Dtype>& bbparams,
                                                  cv::Mat* rand_search_region,
                                                  cv::Mat* target_pad,
                                                  BoundingBox<Dtype>* bbox_gt_scaled) const {
  // 将prev裁剪后的pad图像定义为target_pad
  *target_pad = target_pad_;
  BoundingBox<Dtype> bbox_curr_shift;
  // 将当前帧的box进行随机增强,获得新的box
  bbox_curr_gt_.Shift(image_curr_, bbparams.lambda_scale, bbparams.lambda_shift,
                      bbparams.min_scale, bbparams.max_scale,
                      shift_motion_model,
                      &bbox_curr_shift);
  // Crop the image based at the new location (after applying translation and scale changes).
  Dtype edge_spacing_x, edge_spacing_y;
  BoundingBox<Dtype> rand_search_location;
  // 在当前帧中,以新的随机box:bbox_curr_shift进行ROI裁剪,输出图像: rand_search_region
  // 有效的图片区域: rand_search_location, 左上角定位:edge_spacing_x/edge_spacing_y
  CropPadImage(bbox_curr_shift, image_curr_, rand_search_region, &rand_search_location,
               &edge_spacing_x, &edge_spacing_y);
  // 重新定位gt在裁剪后的patch中的坐标
  BoundingBox<Dtype> bbox_gt_recentered;
  bbox_curr_gt_.Recenter(rand_search_location, edge_spacing_x, edge_spacing_y, &bbox_gt_recentered);

  // 计算归一化坐标: 首先使用裁剪patch的w/h进行归一化,再使用scale进行统一放大: 为了与Net的输出值范围匹配
  bbox_gt_recentered.Scale(*rand_search_region, bbox_gt_scaled);
  // 可视化增广结果
  if (visualize_example) {
    VisualizeExample(*target_pad, *rand_search_region, *bbox_gt_scaled);
  }
}

template <typename Dtype>
void ExampleGenerator<Dtype>::VisualizeExample(const cv::Mat& target_pad,
                                        const cv::Mat& image_rand_focus,
                                        const BoundingBox<Dtype>& bbox_gt_scaled) const {
  cv::Mat target_resize;
  cv::resize(target_pad, target_resize, cv::Size(227,227));
  cv::namedWindow("Target object", cv::WINDOW_AUTOSIZE );
  cv::imshow("Target object", target_resize);

  cv::Mat image_resize;
  cv::resize(image_rand_focus, image_resize, cv::Size(227, 227));

  // Draw gt bbox.
  BoundingBox<Dtype> bbox_gt_unscaled;
  bbox_gt_scaled.Unscale(image_resize, &bbox_gt_unscaled);
  bbox_gt_unscaled.Draw(0, 255, 0, &image_resize);
  // Show image with bbox.
  cv::namedWindow("Search_region_gt", cv::WINDOW_AUTOSIZE );
  cv::imshow("Search_region_gt", image_resize);
  cv::waitKey(0);
}

INSTANTIATE_CLASS(ExampleGenerator);
}
