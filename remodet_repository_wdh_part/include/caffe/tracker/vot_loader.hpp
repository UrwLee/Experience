#ifndef CAFFE_TRACKER_VOT_LOADER_H
#define CAFFE_TRACKER_VOT_LOADER_H

#include "caffe/tracker/video_loader.hpp"
#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * VOT视频序列数据加载器
 */

template <typename Dtype>
class VOTLoader : public VideoLoader<Dtype> {
public:
  /**
   * 构造方法：提供VOT数据集的路径即可。
   */
  VOTLoader(const std::string& vot_folder);
};

}

#endif
