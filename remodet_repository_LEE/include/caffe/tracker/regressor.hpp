#ifndef CAFFE_TRACKER_REGRESSOR_H
#define CAFFE_TRACKER_REGRESSOR_H

#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/regressor_base.hpp"

namespace caffe {

/**
 * 该类为一个坐标回归器，使用一个回归网络来回归坐标。
 */

template <typename Dtype>
class Regressor : public RegressorBase<Dtype> {
 public:

  /**
   *  network_proto: -> 网络描述文件
   *  caffe_model: -> 网络权值文件
   *  gpu_id: -> GPU计算ID
   *  num_inputs: -> 输入Blobs数量，一般为１
   */
  Regressor(const std::string& network_proto,
            const std::string& caffe_model,
            const int gpu_id,
            const int num_inputs);

  /**
   * 默认构造方法：num_inputs为1
   */
  Regressor(const std::string& network_proto,
            const std::string& caffe_model,
            const int gpu_id);

  /**
   * 使用在线的网络来构造。
   */
  Regressor(const boost::shared_ptr<caffe::Net<Dtype> >& net, const int gpu_id);

  /**
   * 回归方法：单个目标或多个目标
   * @param image  [当前帧的ROI-Patch]
   * @param target [历史帧的ROI-Patch]
   * @param bbox   [返回的回归Box]
   */
  virtual void Regress(const cv::Mat& image, const cv::Mat& target, BoundingBox<Dtype>* bbox);
  virtual void Regress(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& targets,
                       std::vector<BoundingBox<Dtype> >* bboxes);

protected:
  /**
   * 获取网络的输出数据，将输出Blob按照vector<>的形式输出
   * @param output [输出数据]
   */
  void GetOutput(std::vector<Dtype>* output);

  /**
   * 对输入的目标数量（样本数）进行Reshape
   * @param num_images [目标数]
   */
  void ReshapeImageInputs(const int num_images);

  /**
   * 获取回归网络中某个特征名的Blobs，并将其转换为vector<>
   * @param feature_name [特征Blob名称]
   * @param output       [输出vector<>数据]
   */
  void GetFeatures(const std::string& feature_name, std::vector<Dtype>* output) const;

  /**
   * 回归网络的估计方法
   * @param image  [当前帧的ROI-Patch]
   * @param target [历史帧的ROI-Patch]
   * @param output [输出数据]
   * 注意：该方法集合了数据载入，网络前向计算，与输出数据获取
   */
  void Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<Dtype>* output);

  /**
   * 该方法提供了对多个目标的估计方法
   */
  void Estimate(const std::vector<cv::Mat>& images,const std::vector<cv::Mat>& targets,
                std::vector<Dtype>* output);

  /**
   * 初始化方法
   */
  virtual void Init();

 private:
  /**
   * 构建网络对象
   * @param network_proto [网络描述文件]
   * @param caffe_model   [网络权值文件]
   * @param gpu_id        [计算设备ID]
   */
  void SetupNetwork(const std::string& network_proto,
                    const std::string& caffe_model,
                    const int gpu_id);
 private:
  //  目标输入对象，默认为１，也可以设置为N，同时对N个对象记性Tracking
  int num_inputs_;
  // 输入图像（ROI-Patch）的尺寸
  cv::Size input_geometry_;
  // 通道数，原始图片为3
  int num_channels_;
  // 网络权值文件
  std::string caffe_model_;
  // 默认为False,除非在线修改权值
  bool modified_params_;
};

}

#endif
