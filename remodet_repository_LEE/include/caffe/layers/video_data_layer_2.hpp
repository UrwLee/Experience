#ifndef CAFFE_VIDEO_DATA_LAYER_2_HPP_
#define CAFFE_VIDEO_DATA_LAYER_2_HPP_

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
 * 注意：该层已停止使用。
 */

template <typename Dtype>
class VideoData2Layer : public BasePrefetchingData2Layer<Dtype> {
 public:
  explicit VideoData2Layer(const LayerParameter& param);
  virtual ~VideoData2Layer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoData2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  virtual void load_batch_orig(Batch_Orig<Dtype>* batch_orig);

  VideoDataParameter_VideoType video_type_;
  cv::VideoCapture cap_;

  int total_frames_;
  int processed_frames_;
  vector<int> top0_shape_;
  vector<int> top1_shape_;

  // 定义摄像头的输入尺寸
  int webcam_width_;
  int webcam_height_;
  // 定义裁剪的尺寸
  int crop_width_;
  int crop_height_;

  float contrast_scale_;
  int exposure_;
  int medianblur_ksize_;
  // greyworld_awb gains
  float kb_ = 0;
  float kg_ = 0;
  float kr_ = 0;
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
