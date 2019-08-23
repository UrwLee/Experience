#ifndef CAFFE_VIDEO_DATA_LAYER_HPP_
#define CAFFE_VIDEO_DATA_LAYER_HPP_

#ifdef USE_OPENCV
#if OPENCV_VERSION == 3
#include <opencv2/videoio.hpp>
#else
#include <opencv2/opencv.hpp>
#endif  // OPENCV_VERSION == 3
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * 该层提供了使用网络摄像头和视频作为数据帧输入的方法。
 * 其行为由参数VideoDataParameter确定。
 * 该层将提供两种输出：
 * （１）data -> [N,3,H,W] (作为网络的计算输入)
 * （２）image_data -> [N,3,IH,IW] (作为显示的图像数据)
 */

template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoDataLayer(const LayerParameter& param);
  virtual ~VideoDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:

  /**
   * 加载一个batch数据
   */
  virtual void load_batch(Batch<Dtype>* batch);

  // 数据类型：摄像头或视频
  VideoDataParameter_VideoType video_type_;

  // 视频流
  cv::VideoCapture cap_;

  // 针对视频：总帧数
  int total_frames_;
  // 已处理帧数
  int processed_frames_;

  // Unused.
  vector<int> top_shape_;
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
