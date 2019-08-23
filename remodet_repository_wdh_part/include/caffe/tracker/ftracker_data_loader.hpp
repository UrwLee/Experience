#ifndef CAFFE_TRACKER_FTRACKER_DATA_LOADER_H
#define CAFFE_TRACKER_FTRACKER_DATA_LOADER_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/video.hpp"
#include "caffe/tracker/video_loader.hpp"
#include "caffe/tracker/image_loader.hpp"
#include "caffe/tracker/fexap_generator.hpp"
#include "caffe/tracker/vot_loader.hpp"
#include "caffe/tracker/alov_loader.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供了Ftracker训练的数据输入
 */

template <typename Dtype>
class FTrackerDataLoader {
public:
  /**
   * 构造：使用proto定义的参数输入即可
   */
  FTrackerDataLoader(const FTrackerDataLoaderParameter& param);
  // 每次载入一个batch
  // transformed_data：　数据
  // transformed_label: box-gt
  void Load(Dtype* transformed_data, Dtype* transformed_label);

protected:
  /**
   * 生成样本
   * @param num_generated_examples [样本对数]
   * @param curr                   [当前的Roi-Blobs]
   * @param prev                   [历史的Roi-Blobs]
   * @param bboxes_gt_scaled       [box-gt值]
   */
  void MakeExamples(const int num_generated_examples,
                    std::vector<boost::shared_ptr<Blob<Dtype> > >* curr,
                    std::vector<boost::shared_ptr<Blob<Dtype> > >* prev,
                    std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled);

  // 样本生成器
  shared_ptr<FExampleGenerator<Dtype> > fexap_generator_;
  // 图片加载器
  shared_ptr<ImageLoader<Dtype> > image_loader_;
  // 视频加载器
  shared_ptr<VideoLoader<Dtype> > video_loader_;
  // 数据加载参数
  FTrackerDataLoaderParameter param_;
};

}

#endif
