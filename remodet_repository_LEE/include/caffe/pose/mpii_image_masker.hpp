#ifndef CAFFE_POSE_MPII_MASKER_H
#define CAFFE_POSE_MPII_MASKER_H

#include "caffe/pose/pose_image_loader.hpp"
#include "caffe/pose/mpii_image_mask_generator.hpp"

namespace caffe {

/**
 * 该类定义了Mpii的Mask_Miss图像生成过程。
 */

template <typename Dtype>
class MpiiMasker {
public:

  /**
   * box_xml_dir　－　box-XML文件目录
   * kps_xml_dir　－　kps-XML文件目录
   * image_folder　－　图像目录
   * output_folder　－　输出目录
   */
  MpiiMasker(const std::string& box_xml_dir,
            const std::string& kps_xml_dir,
            const std::string& image_folder,
            const std::string& output_folder);

  /**
   * 处理过程
   * @param save_image [是否保存图像]
   * @param show_box   [是否show-box]
   */
  void Process(const bool save_image, const bool show_box);

private:
  /**
   *  box-xml文件目录
   *  kps_xml文件目录
   *  图像根目录
   *  输出保存路径
   */
  std::string box_xml_dir_;
  std::string kps_xml_dir_;
  std::string image_folder_;
  std::string output_folder_;
};

}

#endif
