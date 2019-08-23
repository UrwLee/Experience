#ifndef CAFFE_POSE_MPII_IMAGE_LOADER_H
#define CAFFE_POSE_MPII_IMAGE_LOADER_H

#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

/**
 * 该类用于加载MPII数据集
 * 该类作为PoseImageLoader的派生类，在加载数据标注后
 * 即可以使用基类的方法进行可视化。
 */


template <typename Dtype>
class MpiiImageLoader : public PoseImageLoader<Dtype> {
public:
  /**
   * 构造：
   * １．image_folder：图像根目录
   * ２．xml_folder：标注文件目录
   * 读取XML文件目录下的所有XML文件
   */
  MpiiImageLoader(const std::string& image_folder, const std::string& xml_folder);

private:

  /**
   * 从XML文件中加载标注信息
   * @param  annotation_file [标注XML文件]
   * @param  image_path      [图像根目录]
   * @param  meta            [返回的标注信息]
   * @return                 [读取成功或失败]
   */
  bool LoadAnnotationFromXmlFile(const std::string& annotation_file,
                                 const std::string& image_path,
                                 MetaData<Dtype>* meta);
};

}

#endif
