#ifndef CAFFE_TRACKER_ALOV_LOADER_H
#define CAFFE_TRACKER_ALOV_LOADER_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/video.hpp"
#include "caffe/tracker/video_loader.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * ALOV数据集的加载器。
 * 继承自VideoLoader类，该类的派生类有两种：
 * １．VOTLoader: 用于加载VOT数据集
 * ２．ALOVLoader: 用于加载ALOV数据集
 */

template <typename Dtype>
class ALOVLoader : public VideoLoader<Dtype> {
public:
  /**
   * 构造方法：
   * images -> 图片所在目录
   * annotations－> 标注文件所在目录
   */
  ALOVLoader(const std::string& images, const std::string& annotations);

  /**
   * 按照训练／测试：获取视频序列
   * @param get_train [训练或测试标记]
   * @param videos    [返回的视频序列]
   */
  void get_videos(const bool get_train, std::vector<Video<Dtype> >* videos) const;

private:

  /**
   * 获取所有的视频序列
   */
  const std::vector<Video<Dtype> >& get_videos() const { return this->videos_; }

  /**
   * 所有类别的视频序列集合
   * 将所有视频按照其类比进行组织
   */
  std::vector<Category<Dtype> > categories_;
};

}

#endif
