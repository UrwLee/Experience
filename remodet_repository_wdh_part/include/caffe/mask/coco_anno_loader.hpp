#ifndef CAFFE_MASK_COCO_ANNO_LOADER_H
#define CAFFE_MASK_COCO_ANNO_LOADER_H

#include "caffe/mask/anno_image_loader.hpp"

namespace caffe {

template <typename Dtype>
class CocoAnnoLoader : public AnnoImageLoader<Dtype> {

/**
 * 基于AnnoImageLoader，提供COCO数据标注加载器
 */

public:
  CocoAnnoLoader(const std::string& image_folder, const std::string& xml_folder);
private:

  /**
   * 从XML文件中加载数据
   * @param  annotation_file [XML文件]
   * @param  image_path      [图像根目录]
   * @param  anno            [返回标注数据结构]
   * @return                 [读取是否失败false/true]
   */
  bool LoadAnnotationFromXmlFile(const std::string& annotation_file,
                                 const std::string& image_path,
                                 AnnoData<Dtype>* anno);
};

}

#endif
