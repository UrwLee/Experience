#ifndef CAFFE_POSE_COCO_SAVE_TXT_H
#define CAFFE_POSE_COCO_SAVE_TXT_H

#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

/**
 * 该类将COCO的标注信息写到一个txt文件里
 * 每一行代表了一个实例的标注信息
 * 每行的数据定义参考cpp文件
 */

template <typename Dtype>
class CocoTxtSaver {
public:

  /**
   * xml_dir: -> xml文件目录
   * output_file: -> 输出文件名
   */
  CocoTxtSaver(const std::string& xml_dir,
               const std::string& output_file);

  /**
   * 保存
   */
  void Save();

protected:
  /**
   * 从XML文件中加载标注数据
   * @param  xml_file [xml文件]
   * @param  meta     [标注数据]
   * @return          [读取成功或失败]
   */
  bool LoadAnnotationFromXmlFile(const std::string& xml_file,
                                 MetaData<Dtype>* meta);

private:
  // 输出文件指针
  FILE* output_file_ptr_;
  // XML文件目录
  std::string xml_dir_;
  // 输出文件名
  std::string output_file_;
};

}

#endif
