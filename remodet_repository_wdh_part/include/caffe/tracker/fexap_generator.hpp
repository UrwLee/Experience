#ifndef CAFFE_TRACKER_FEXAP_GENERATOR_H
#define CAFFE_TRACKER_FEXAP_GENERATOR_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/fe_roi_maker.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
struct BBParams {
  Dtype lambda_shift;
  Dtype lambda_scale;
  Dtype min_scale;
  Dtype max_scale;
};

template <typename Dtype>
class FExampleGenerator {
public:

  /**
   * 样本发生器（Ｆ）
   * 不同于普通的样本发生器（ExampleGenerator），该样本发生器对图片及ROI进行特征提取
   * 输出的不是ROI-Patch，而是一个resized-Blob
   */

  /**
   * lambda_shift/lambda_scale/min_scale/max_scale: -> 随机参数
   * network_proto/caffe_model: -> 特征提取网络
   * gpu_id: -> GPU ID
   * features: -> 特征名
   * resized_width/resized_height: -> ROI特征resized尺寸
   */
  FExampleGenerator(const Dtype lambda_shift, const Dtype lambda_scale,
                    const Dtype min_scale, const Dtype max_scale,
                    const std::string& network_proto,
                    const std::string& caffe_model,
                    const int gpu_id,
                    const std::string& features,
                    const int resized_width,
                    const int resized_height);

  /**
   * 初始化：使用随机参数
   */
  void Init(const Dtype lambda_shift, const Dtype lambda_scale,
            const Dtype min_scale, const Dtype max_scale);

  /**
   * 复位过程
   * @param bbox_prev  [历史位置]
   * @param bbox_curr  [当前位置]
   * @param image_prev [历史图片]
   * @param image_curr [当前图片]
   */
  void Reset(const BoundingBox<Dtype>& bbox_prev, const BoundingBox<Dtype>& bbox_curr,
             const cv::Mat& image_prev, const cv::Mat& image_curr);

  /**
   * 创建一个随机增广的样本对
   * @param curr           [当前特征Blob]
   * @param prev           [历史特征Blob]
   * @param bbox_gt_scaled [box-GT值]
   */
  void MakeTrainingExampleBBShift(Blob<Dtype>* curr,
                                  Blob<Dtype>* prev,
                                  BoundingBox<Dtype>* bbox_gt_scaled);

  /**
   * 创建一个真实样本对：不使用增广
   * @param curr           [当前特征Blob]
   * @param prev           [历史特征Blob]
   * @param bbox_gt_scaled [box-GT值]
   */
  void MakeTrueExample(Blob<Dtype>* curr, Blob<Dtype>* prev,
                       BoundingBox<Dtype>* bbox_gt_scaled);

  /**
   * 批量创建样本对
   * 注意：默认创建一个真实样本对，其余全部是增广的样本对
   * @param num_examples     [样本对数]
   * @param curr             [当前特征Blobs]
   * @param prev             [历史特征Blobs]
   * @param bboxes_gt_scaled [box-GTs值]
   */
  void MakeTrainingExamples(const int num_examples,
                            std::vector<boost::shared_ptr<Blob<Dtype> > >* curr,
                            std::vector<boost::shared_ptr<Blob<Dtype> > >* prev,
                            std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled);

  // ignored.
  void set_indices(const int video_index, const int frame_index) {
    video_index_ = video_index; frame_index_ = frame_index;
  }

private:
  /**
   * 构造随机增广样本
   * @param bbparams       [随机参数]
   * @param curr           [当前特征Blob]
   * @param prev           [历史特征Blob]
   * @param bbox_gt_scaled [box-GT值]
   */
  void MakeTrainingExampleBBShift(const BBParams<Dtype>& bbparams,
                                  Blob<Dtype>* curr,
                                  Blob<Dtype>* prev,
                                  BoundingBox<Dtype>* bbox_gt_scaled);

  // 构造的默认参数
  void get_default_bb_params(BBParams<Dtype>* default_params);

  // 随机参数
  Dtype lambda_shift_;
  Dtype lambda_scale_;
  Dtype min_scale_;
  Dtype max_scale_;

  // 内部的ROI特征抽取器
  boost::shared_ptr<FERoiMaker<Dtype> > roi_maker_;
  // 当前图像
  cv::Mat image_curr_;
  // 历史的resized-Blob
  Blob<Dtype> prev_f_;

  // 当前和历史的box
  BoundingBox<Dtype> bbox_curr_gt_;
  BoundingBox<Dtype> bbox_prev_gt_;

  // ignored.
  int video_index_;
  int frame_index_;
};

}

#endif
