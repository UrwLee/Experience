#ifndef CAFFE_TRACKER_VIDEO_LOADER_H
#define CAFFE_TRACKER_VIDEO_LOADER_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/video.hpp"
#include "caffe/tracker/image_loader.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供了对视频类数据的加载方法。
 * 该类是一个基类。
 */

template <typename Dtype>
class VideoLoader {
public:
  VideoLoader();

  /**
   *　显示数据中的图片和位置标记
   */
  void ShowVideos() const;

  /**
   * 显示随机增广后的图片和位置
   */
  void ShowVideosShift() const;

  /**
   * 获取所有的视频集合
   */
  std::vector<Video<Dtype> > get_videos() const { return videos_; }

  /**
   * 获取第三方视频数据加载器的数据
   * @param dst [第三方视频数据加载器]
   */
  void merge_from(const VideoLoader<Dtype>* dst);

  /**
   * 获取视频的数量
   * @return [数量]
   */
  int get_size() const { return videos_.size(); }
protected:
  // 视频集合
  std::vector<Video<Dtype> > videos_;
};

}

#endif
