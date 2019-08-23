#ifndef CAFFE_POSE_XML_MODIFICATION_H
#define CAFFE_POSE_XML_MODIFICATION_H

#include <vector>
#include <string>

#include "caffe/tracker/bounding_box.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"

#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

/**
 * 该类的作用是修改XML文件中的信息，并进行适当修改后保存。
 */

template <typename Dtype>
class XmlModifier {
public:

  /**
   * xml_folder: -> 读取的xml文件目录
   * output_folder: ->　输出保存目录
   */
  XmlModifier(const std::string& xml_folder,
              const std::string& output_folder);

  /**
   * 修改方法
   */
  void Modify();

private:
  /**
   * 修改方法
   * @param meta [标注信息]
   */
  void ModifyXML(MetaData<Dtype>* meta);

  /**
   * 保存XML方法
   * @param meta     [标注信息]
   * @param xml_name [XML文件名称]
   */
  void SaveXml(const MetaData<Dtype>& meta, const std::string& xml_name);

  /**
   * 加载标注信息
   * @param  annotation_file [输入XML文件]
   * @param  meta            [标注信息]
   * @return                 [读取成功或失败]
   */
  bool LoadAnnotationFromXmlFile(const string& annotation_file, MetaData<Dtype>* meta);

  // XML文件目录
  std::string xml_folder_;
  // 输出路径
  std::string output_folder_;
};

}

#endif
