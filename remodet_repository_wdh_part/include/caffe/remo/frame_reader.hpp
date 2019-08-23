#ifndef CAFFE_REMO_FRAME_READER_H
#define CAFFE_REMO_FRAME_READER_H

#include <boost/shared_ptr.hpp>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/remo/data_frame.hpp"
#include "caffe/remo/basic.hpp"

namespace caffe {

template <typename Dtype>
class FrameReader {
public:

  /**
   * 该类用于构建一个实时视频流，并获取数据帧，数据帧提供：
   * １．　原始图像帧
   * ２．　resized图像帧，用于网络计算
   */

  /**
   * 构造方法：摄像头输入
   * cam_id -> 摄像头编号
   * width/height -> 摄像头读入帧的宽度和高度设置
   * resized_width/resized_height -> resized后的尺寸，用于网络输入
   */
  FrameReader(int cam_id, int width, int height, int resized_width, int resized_height);

  /**
   * 构造方法：视频输入
   * video_file -> 视频文件
   * start_frame -> 起始帧编号
   * resized_width/resized_height -> resized后的尺寸，用于网络输入
   */
  FrameReader(const std::string& video_file, int start_frame, int resized_width, int resized_height);

  /**
   * 构造方法：RTSP流
   * ip_addr -> IP 地址
   * resized_width/resized_height -> resized后的尺寸，用于网络输入
   */
  FrameReader(const std::string& ip_addr, int resized_width, int resized_height);

  /**
   * 析构方法：释放数据流
   */
  virtual ~FrameReader() {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

  /**
   * 从实时数据流中获取新帧
   * @param  frame [获取的帧:DataFrame]
   * @return       [获取状态，０－正常，１－视频流读取结束]
   */
  int pop(DataFrame<Dtype>* frame);

  /**
   * 实时显示数据流
   */
  void show();

protected:
  // 数据流捕获
  cv::VideoCapture cap_;

  // 视频总帧数
  int total_frames_;
  // 已处理帧数
  int processed_frames_;
  // 初始帧数
  int initial_frame_;

  // resized尺寸
  int resized_width_;
  int resized_height_;

  // 数据流类型
  // 0 -> video
  // 1 -> cam
  // 2 -> web-server
  int type_video_;
};

}

#endif
