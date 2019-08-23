#ifndef CAFFE_VIDEO_FRAME_LAYER_HPP_
#define CAFFE_VIDEO_FRAME_LAYER_HPP_

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
#include "caffe/layers/base_data_layer_2.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * 使用摄像头和视频文件提供数据。
 * 同时提供网络数据输入的data和终端显示的图片数据image
 * 注意：该层使用了Batch_Orig类型的batch数据
 * Batch_Orig数据包含了data,image_data,label三个字段
 */

template <typename Dtype>
class VideoframeLayer : public BasePrefetchingData2Layer<Dtype> {
 public:
  explicit VideoframeLayer(const LayerParameter& param);
  virtual ~VideoframeLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Videoframe"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  /**
   * 加载数据batch
   * <data,image_data,label>
   * label could be ignored.
   */
  virtual void load_batch_orig(Batch_Orig<Dtype>* batch_orig);

  // 视频类型
  VideoframeParameter_VideoType video_type_;
  // 视频流
  cv::VideoCapture cap_;

  // 总帧数
  int total_frames_;
  // 已处理帧数
  int processed_frames_;
  // 初始跳过帧数
  int initial_frame_;
  // unused.
  vector<int> top0_shape_;
  vector<int> top1_shape_;

  // 定义摄像头的输入尺寸
  int webcam_width_;
  int webcam_height_;

  // normalize:数据加载进入网络是否进行归一化：(-0.5,+0.5)
  bool normalize_;
  // 通道的均值列表：[104,117,123]
  vector<Dtype> mean_values_;
};

}  

#endif
