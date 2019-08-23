#ifndef CAFFE_POSE_MPII_SAVE_TXT_H
#define CAFFE_POSE_MPII_SAVE_TXT_H

#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

template <typename Dtype>
class MpiiTxtSaver {
public:
  /**
   * xml_dir: -> XML文件目录
   * output_file: ->　输出文件
   */
  MpiiTxtSaver(const std::string& xml_dir,
               const std::string& output_file);

  /**
   * 保存方法：保存到输出txt文件
   */
  void Save();

protected:
  /**
   * 加载标注文件
   * @param  xml_file [XML文件]
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
