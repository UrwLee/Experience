#ifndef CAFFE_PIC_VISUALIZER_HPP
#define CAFFE_PIC_VISUALIZER_HPP

#include "caffe/pic/pic_visual_saver.hpp"
#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

/**
 * 构图结构可视化类
 * 该类提供了将构图信息进行可视化以及保存的方法
 */

template <typename Dtype>
class PicVisualizer {
public:

  /**
   * 构造方法
   * meta_pic -> 构图数据结构　【单条目】
   * image_dir -> 图像根目录
   * output_dir -> 输出目录
   */
  PicVisualizer(const PicData<Dtype>& meta_pic,
               const std::string& image_dir,
               const std::string& output_dir);

  /**
   * 将构图数据保存
   */
  void Save();

private:
  // 构图数据
  PicData<Dtype> meta_pic_;
  // 输出目录
  std::string output_dir_;
  // 图像根目录
  std::string image_dir_;
};

}

#endif
