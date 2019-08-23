#ifndef CAFFE_TRACKER_TRACKER_DATA_LOADER_H
#define CAFFE_TRACKER_TRACKER_DATA_LOADER_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/video.hpp"
#include "caffe/tracker/video_loader.hpp"
#include "caffe/tracker/image_loader.hpp"
#include "caffe/tracker/example_generator.hpp"
#include "caffe/tracker/vot_loader.hpp"
#include "caffe/tracker/alov_loader.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供了对Tracker训练的数据加载器
 */

template <typename Dtype>
class TrackerDataLoader {
public:

  /**
   * 数据加载构造：由proto参数定义
   */
  TrackerDataLoader(const TrackerDataLoaderParameter& param);

  /**
   * 加载一个batch
   * @param transformed_data  [数据指针]
   * @param transformed_label [box数据指针]
   */
  void Load(Dtype* transformed_data, Dtype* transformed_label);

protected:
  /**
   * 样本生成
   * @param num_generated_examples [样本对的数量]
   * @param images                 [当前ROI-Patches]
   * @param targets                [历史ROI-Patches]
   * @param bboxes_gt_scaled       [Boxes的GT值列表]
   */
  void MakeExamples(const int num_generated_examples,
                    std::vector<cv::Mat>* images,
                    std::vector<cv::Mat>* targets,
                    std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled);

  // 样本生成器
  shared_ptr<ExampleGenerator<Dtype> > example_generator_;
  // 图片加载器
  shared_ptr<ImageLoader<Dtype> > image_loader_;
  // 视频序列加载器
  shared_ptr<VideoLoader<Dtype> > video_loader_;

  // 数据加载参数
  TrackerDataLoaderParameter param_;
};

}

#endif
