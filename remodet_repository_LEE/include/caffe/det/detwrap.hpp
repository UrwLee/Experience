#ifndef CAFFE_DET_WRAPPER_HPP_
#define CAFFE_DET_WRAPPER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/mask/bbox_func.hpp"

#include "caffe/remo/data_frame.hpp"
#include "caffe/remo/basic.hpp"

namespace caffe {

template <typename Dtype>
class DetWrapper {
public:
  // network_proto -> 网络文件
  // caffe_model -> 权值文件
  // gpu_id -> GPU的ID
  // proposals -> "proposals" <blob name>
  // max_dis_size -> 可视化的最大尺寸：max(width, height)
  DetWrapper(const std::string& network_proto,
             const std::string& caffe_model,
             const bool mode,
             const int gpu_id,
             const std::string& proposals,
             const int max_dis_size);

  // 使用一个在线的网络对象初始化
  DetWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
             const int gpu_id,
             const std::string& proposals,
             const int max_dis_size);

  /**
   * 获取输出的ROIs
   * @param frame  [输入帧]
   * @param rois   [返回Rois]
   * @param scores [返回置信度]
   */
  void get_rois(DataFrame<Dtype>& frame, std::vector<LabeledBBox<Dtype> >* rois);

  /**
   * 绘制检测结果
   * @param  frame  [输入帧]
   * @param  labels [需要绘制的类型]
   * @param  rois   [返回Rois]
   * @param  scores [返回置信度]
   * @return        [返回绘制好的图片]
   */
  cv::Mat get_drawn_bboxes(DataFrame<Dtype>& frame, const std::vector<int>& labels, std::vector<LabeledBBox<Dtype> >* rois);

  /**
   * 返回加载数据时间
   */
  Dtype get_preload_us() { return pre_load_time_; }

  /**
   * 返回网络计算时间
   */
  Dtype get_forward_us() { return process_time_; }

  /**
   * 返回绘制时间
   */
  Dtype get_drawn_us() { return drawn_time_; }

  /**
   * 返回帧号
   */
  int get_frames() { return frames_; }

  boost::shared_ptr<caffe::Net<Dtype> > get_net() { return net_; }

  /**
   * 使用Blob返回某个Blob
   * @param feature_name [Blob名称]
   * @param data         [返回的Blob]
   */
  void getFeatures(const std::string& feature_name, Blob<Dtype>* data);

  void getMonitorBlob(DataFrame<Dtype>& frame, const std::string& monitor_blob_name, Blob<Dtype>* res_blob);

protected:
  /**
   * 加载数据
   */
  void load(const cv::Mat& image);
  /**
   * 前向计算
   */
  void step(DataFrame<Dtype>& frame);

private:
  // 神经网络
  boost::shared_ptr<caffe::Net<Dtype> > net_;

  // 网络要求的输入尺寸
  int input_width_;
  int input_height_;

  // 网络已经处理的帧数
  int frames_;

  // 图像预处理时间：30帧平均一次
  Dtype pre_load_time_;
  // 网络前向计算时间：30帧平均一次
  Dtype process_time_;
  // 图形绘制时间：30帧平均一次
  Dtype drawn_time_;

  // 30帧累计时间：　分别对应三个过程：预处理／前向计算／绘制
  Dtype pre_load_time_sum_;
  Dtype process_time_sum_;
  Dtype drawn_time_sum_;

  // 输出图像可视化的最大尺寸： max(width,height)
  int max_dis_size_;

  std::string proposals_;

  bool hisi_data_;
  std::map<int,int> maps_;
};

}

#endif
