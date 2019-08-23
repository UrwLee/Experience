#ifndef CAFFE_HAND_MPII_HAND_LOADER_H
#define CAFFE_HAND_MPII_HAND_LOADER_H

#include "caffe/hand/hand_loader.hpp"

namespace caffe {

/**
 * 该类提供了加载MPII数据集标注信息的方法。
 */

template <typename Dtype>
class MpiiHandLoader : public HandLoader<Dtype> {
public:
  MpiiHandLoader(const std::string& image_folder, const std::string& xml_folder);
private:
  bool LoadAnnotationFromXmlFile(const std::string& annotation_file,
                                 const std::string& image_folder,
                                 MData<Dtype>* meta);
};

}

#endif
