#ifndef CAFFE_COM_COM_WRAPPER_HPP_
#define CAFFE_COM_COM_WRAPPER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

template <typename Dtype>
class ComWrapper {
public:
  // network_proto -> 网络文件
  // caffe_model -> 权值文件
  // gpu_id -> GPU的ID
  // scoremap -> "scoremap" <blob name>
  // display_size (48x27 -> 768x432, x16)
  ComWrapper(const std::string& network_proto,
             const std::string& caffe_model,
             const bool mode,
             const int gpu_id,
             const std::string& scoremap,
             const int display_size);

  // 使用一个在线的网络对象初始化
  ComWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
             const int gpu_id,
             const std::string& scoremap,
             const int display_size);

  /**
   * 获取输出scoremap
   * @param input_blobs [输入特征列表]
   * @param output      [输出scoremap的数据]
   */
  void get_result(vector<Blob<Dtype>* >& input_blobs, Blob<Dtype>* output);

  /**
   * 获取输出scoremap
   * @param  input_blobs [输入特征列表]
   * @param  output      [输出scoremap的数据]
   * @return             [返回cv::Mat图像]
   */
  cv::Mat get_scoremap(vector<Blob<Dtype>* >& input_blobs, Blob<Dtype>* output);

  Dtype get_preload_us() { return pre_load_time_; }

  Dtype get_forward_us() { return process_time_; }

  Dtype get_drawn_us() { return drawn_time_; }

  int get_frames() { return frames_; }

  boost::shared_ptr<caffe::Net<Dtype> > get_net() { return net_; }

  // 获取输出结果
  void getFeatures(const std::string& feature_name, Blob<Dtype>* data);

protected:
  // 加载数据
  void load(vector<Blob<Dtype>* >& input_blobs);
  // 单词运行
  void step(vector<Blob<Dtype>* >& input_blobs);

private:
  // 神经网络
  boost::shared_ptr<caffe::Net<Dtype> > net_;

  // 网络已经处理的帧数
  int frames_;

  // 数据加载时间
  Dtype pre_load_time_;
  // 网络计算时间
  Dtype process_time_;
  // 图形输出时间
  Dtype drawn_time_;

  Dtype pre_load_time_sum_;
  Dtype process_time_sum_;
  Dtype drawn_time_sum_;

  int display_size_;

  std::string scoremap_;
};

}

#endif
