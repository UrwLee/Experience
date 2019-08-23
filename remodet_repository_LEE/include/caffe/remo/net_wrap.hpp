#ifndef CAFFE_REMO_NET_WRAP_H
#define CAFFE_REMO_NET_WRAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/tracker/bounding_box.hpp"

#include "caffe/remo/data_frame.hpp"
#include "caffe/remo/basic.hpp"
#include "caffe/remo/res_frame.hpp"
#include "caffe/remo/visualizer.hpp"

namespace caffe {

template <typename Dtype>
class NetWrapper {
public:

  /**
   * 通用构造函数
   */
  NetWrapper(const std::string& network_proto,
             const std::string& caffe_model,
             const bool mode,
             const int gpu_id,
             const std::string& proposals,
             const std::string& heatmaps,
             const int max_dis_size);
  /**
   * 该类用于构建一个深度神经网络，并提供将输出可视化的方法。
   * 输入：　一个数据帧
   * 输出：　输出可视化以及结构化的目标对象数据
   */

  // network_proto -> 网络文件
  // caffe_model -> 权值文件
  // gpu_id -> GPU的ID
  // proposals -> "proposals" <blob name>
  // heatmaps -> "heatmaps" <blob name>
  // max_dis_size -> 可视化的最大尺寸：max(width, height)
  NetWrapper(const std::string& network_proto,
            const std::string& caffe_model,
            const int gpu_id,
            const std::string& proposals,
            const std::string& heatmaps,
            const int max_dis_size);

  // 使用一个在线的网络对象初始化
  NetWrapper(const boost::shared_ptr<caffe::Net<Dtype> >& net,
            const int gpu_id,
            const std::string& proposals,
            const std::string& heatmaps,
            const int max_dis_size);

  /**
   * 获取目标对象的数据结构表示
   * @param frame [输入的数据帧]
   * @param meta  [输出的目标对象数据结构]
   * 注意：输出类型为数据结构
   */
  void get_meta(DataFrame<Dtype>& frame, std::vector<PMeta<Dtype> >* meta);

  /**
   * 获取目标对象的数据结构表示
   * @param frame [输入的数据帧]
   * @param meta  [输出的目标对象数据结构]
   * 注意：输出类型为vector<vector<Dtype> >
   */
  void get_meta(DataFrame<Dtype>& frame, std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的Limbs可视化结果
   * @param  frame [输入帧]
   * @param  meta  [输出的目标对象数据结构]
   * @return       [可视化图像cv::Mat]
   */
  cv::Mat get_vecmap(DataFrame<Dtype>& frame, std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的Limbs可视化结果
   * @param  frame     [输入帧]
   * @param  show_bbox [是否绘制BOX]
   * @param  show_id   [是否绘制ID]
   * @param  meta      [输出的目标对象数据结构]
   * @return           [可视化图像cv::Mat]
   */
  cv::Mat get_vecmap(DataFrame<Dtype>& frame, const bool show_bbox, const bool show_id,
                      std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的Points可视化结果
   * @param  frame [输入帧]
   * @param  meta  [输出的目标对象数据结构]
   * @return       [可视化图像cv::Mat]
   */
  cv::Mat get_heatmap(DataFrame<Dtype>& frame, std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的Points可视化结果
   * @param  frame     [输入帧]
   * @param  show_bbox [是否绘制BOX]
   * @param  show_id   [是否绘制ID]
   * @param  meta      [输出的目标对象数据结构]
   * @return           [可视化图像cv::Mat]
   */
  cv::Mat get_heatmap(DataFrame<Dtype>& frame, const bool show_bbox, const bool show_id,
                       std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的BOX结果
   * @param  frame   [输入帧]
   * @param  show_id [是否绘制ID]
   * @param  meta    [输出的目标对象数据结构]
   * @return         [可视化图像cv::Mat]
   */
  cv::Mat get_bbox(DataFrame<Dtype>& frame, const bool show_id, std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的骨架结果
   * @param  frame [输入帧]
   * @param  meta  [输出的目标对象数据结构]
   * @return       [可视化图像cv::Mat]
   */
  cv::Mat get_skeleton(DataFrame<Dtype>& frame, std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的骨架结果
   * @param  frame     [输入帧]
   * @param  show_bbox [是否绘制BOX]
   * @param  show_id   [是否绘制ID]
   * @param  meta      [输出的目标对象数据结构]
   * @return           [可视化图像cv::Mat]
   */
  cv::Mat get_skeleton(DataFrame<Dtype>& frame, const bool show_bbox, const bool show_id,
                        std::vector<std::vector<Dtype> >* meta);

  /**
   * 可视化：输入帧的骨架结果
   * @param  frame [输入帧]
   * @param  meta  [输出的目标对象数据结构]
   * @return       [可视化图像cv::Mat]
   */
  cv::Mat get(DataFrame<Dtype>& frame, std::vector<std::vector<Dtype> >* meta);

  /**
   * 获取数据帧的预处理时间：单位us
   * @return [预处理时间]
   */
  Dtype get_preload_us() { return pre_load_time_; }

  /**
   * 获取网络的前向计算时间：单位us
   * @return [前向计算时间]
   */
  Dtype get_forward_us() { return process_time_; }

  /**
   * 获取图形绘制的时间：单位us
   * @return [图形绘制时间]
   */
  Dtype get_drawn_us() { return drawn_time_; }

  /**
   * 获取网络已经处理的帧数
   * @return [帧数]
   */
  int get_frames() { return frames_; }

  /**
   * 返回内部神经网络的指针
   */
  boost::shared_ptr<caffe::Net<Dtype> > get_net() { return net_; }

  /**
   * 获取网络中某个名称的特征Blob
   * @param feature_name [待获取的特征Blob名]
   * @param data         [输出的Blob指针]
   */
  void getFeatures(const std::string& feature_name, Blob<Dtype>* data);

protected:
  /**
   * 加载图像到网络的输入
   * @param image [待计算的图像cv::Mat]
   */
  void load(const cv::Mat& image);

  /**
   * 网络单次前向计算过程
   * @param frame [输入的数据帧]
   * @param meta  [输出目标对象数据结构]
   * @param map   [输出heat/vec-map特征图Blob指针]
   */
  void step(DataFrame<Dtype>& frame, std::vector<PMeta<Dtype> >* meta, Blob<Dtype>* map);

private:
  // 神经网络
  boost::shared_ptr<caffe::Net<Dtype> > net_;

  // 网络要求的输入尺寸
  int input_width_;
  int input_height_;

  // 输出proposals的特征Blob名
  std::string proposals_;
  // 输出heat/vec-map特征Blob名
  std::string heatmaps_;

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
};

}

#endif
