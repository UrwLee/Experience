#ifndef CAFFE_PIC_VISUAL_SAVER_HPP
#define CAFFE_PIC_VISUAL_SAVER_HPP

#include <vector>
#include <string>

#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * 本文件用于保存构图结果输出
 * 作用：可视化构图结果
 */

template <typename Dtype>
struct PicData {
  // 图像路径
  std::string image_path;
  // 背景的宽和高
  int bgw;
  int bgh;
  // 图像块的ROI
  BoundingBox<Dtype> bbox;
  // 构图结果
  vector<Dtype> pic;
};

template <typename Dtype>
class PicVisualSaver {
public:

  /**
   * 构造方法
   * pic_file -> 构图文件　(.txt)
   * image_dir -> 图像根目录
   * output_dir -> 保存地址
   */
  PicVisualSaver(const std::string& pic_file,
                 const std::string& image_dir,
                 const std::string& output_dir);

  /**
   * 保存
   */
  void Save();

protected:

  /**
   * 加载构图信息
   * @param pic_file [构图文件]
   * @param metas    [构图数据结构]
   */
  void LoadMetas(const std::string& pic_file, std::vector<PicData<Dtype> >* metas);

private:
  // 构图信息集合
  std::vector<PicData<Dtype> > metas_;
  // 输出目录
  std::string output_dir_;
  // 图像根目录
  std::string image_dir_;
};

}

#endif
