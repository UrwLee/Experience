#ifndef CAFFE_TRACKER_EXAMPLE_GENERATOR_H
#define CAFFE_TRACKER_EXAMPLE_GENERATOR_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/video.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类为一个样本发生类，主要作用：随机生成一个新的样本
 * 随机方法：长宽随机Scale, 中心位置随机偏移
 */

/**
 * 随机增广参数
 */
template <typename Dtype>
struct BBParams {
  Dtype lambda_shift;
  Dtype lambda_scale;
  Dtype min_scale;
  Dtype max_scale;
};

template <typename Dtype>
class ExampleGenerator {
public:

  /**
   * 构造方法：　使用随机过程的参数进行构造即可
   */
  ExampleGenerator(const Dtype lambda_shift, const Dtype lambda_scale,
                   const Dtype min_scale, const Dtype max_scale);

  /**
   * 初始化方法
   * @param lambda_shift [参数]
   * @param lambda_scale [参数]
   * @param min_scale    [参数]
   * @param max_scale    [参数]
   */
  void Init(const Dtype lambda_shift, const Dtype lambda_scale,
                   const Dtype min_scale, const Dtype max_scale);

  /**
   * 复位：使用前后两帧以及对应的Box即可将样本发生器复位
   * 注意：在使用样本发生器生产样本前，务必用该方法进行复位
   * @param bbox_prev  [前一帧的Box]
   * @param bbox_curr  [当前帧的Box]
   * @param image_prev [前一帧的Image]
   * @param image_curr [当前帧的Image]
   */
  void Reset(const BoundingBox<Dtype>& bbox_prev, const BoundingBox<Dtype>& bbox_curr,
             const cv::Mat& image_prev, const cv::Mat& image_curr);

  /**
   * 使用构造器的默认参数／复位样本进行增广样本生产
   * @param visualize_example [是否可视化生成的样本]
   * @param image_rand_focus  [增广后的ROI-Patch]
   * @param target_pad        [历史的ROI-Patch]
   * @param bbox_gt_scaled    [box-GT值]
   */
  void MakeTrainingExampleBBShift(const bool visualize_example,
                                  cv::Mat* image_rand_focus,
                                  cv::Mat* target_pad,
                                  BoundingBox<Dtype>* bbox_gt_scaled) const;

  /**
   * 与上述方法一致，可视化默认是False
   */
  void MakeTrainingExampleBBShift(cv::Mat* image_rand_focus,
                                  cv::Mat* target_pad,
                                  BoundingBox<Dtype>* bbox_gt_scaled) const;

  /**
   * 真实样本构造: 此时样本发生器仅仅使用复位的样本和boxes进行样本构造
   * @param image_focus    [当前的ROI-Patch]
   * @param target_pad     [历史的ROI-Patch]
   * @param bbox_gt_scaled [box-GT值]
   */
  void MakeTrueExample(cv::Mat* image_focus, cv::Mat* target_pad,
                       BoundingBox<Dtype>* bbox_gt_scaled) const;

  /**
   * 批量生成样本
   * 注意：默认生成一个True-Example
   * 剩余的全部是随机生成的增广样本
   * @param num_examples     [样本对数]
   * @param images           [当前的ROI-Patches]
   * @param targets          [历史的ROI-Patches]
   * @param bboxes_gt_scaled [boxes-GT值]
   */
  void MakeTrainingExamples(const int num_examples, std::vector<cv::Mat>* images,
                            std::vector<cv::Mat>* targets,
                            std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled);

  // 该方法无明显作用，跳过即可
  void set_indices(const int video_index, const int frame_index) {
    video_index_ = video_index; frame_index_ = frame_index;
  }

private:
  /**
   * 创建一个随机增广样本
   * @param visualize_example [是否可视化创建的样本]
   * @param bbparams          [随机参数]
   * @param image_rand_focus  [增广后的ROI-Patch]
   * @param target_pad        [原先的ROI-Patch]
   * @param bbox_gt_scaled    [Box-GT值]
   */
  void MakeTrainingExampleBBShift(const bool visualize_example,
                                  const BBParams<Dtype>& bbparams,
                                  cv::Mat* image_rand_focus,
                                  cv::Mat* target_pad,
                                  BoundingBox<Dtype>* bbox_gt_scaled) const;

  /**
   * 可视化生成的样本
   * @param target_pad       [历史ROI-Patch]
   * @param image_rand_focus [当前ROI-Patch]
   * @param bbox_gt_scaled   [box-GT值]
   */
  void VisualizeExample(const cv::Mat& target_pad,
                        const cv::Mat& image_rand_focus,
                        const BoundingBox<Dtype>& bbox_gt_scaled) const;

  // 获取随机增广参数：使用构造器的默认参数
  void get_default_bb_params(BBParams<Dtype>* default_params) const;

  // 构造器的默认参数
  Dtype lambda_shift_;
  Dtype lambda_scale_;
  Dtype min_scale_;
  Dtype max_scale_;

  // 当前帧
  cv::Mat image_curr_;
  // 历史ROI-Patch
  cv::Mat target_pad_;

  // 当前box-gt
  BoundingBox<Dtype> bbox_curr_gt_;
  // 历史box-gt
  BoundingBox<Dtype> bbox_prev_gt_;

  // 视频和帧编号
  int video_index_;
  int frame_index_;
};

}

#endif
