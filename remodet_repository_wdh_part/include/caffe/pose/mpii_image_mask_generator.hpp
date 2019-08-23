#ifndef CAFFE_POSE_MPII_MASK_GEN_H
#define CAFFE_POSE_MPII_MASK_GEN_H

#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

/**
 * 该类提供了为MPII数据集生成Mask_miss图像的方法
 * 为了使用MPII数据集进行姿态识别，需要将图片中没有包含关节点的person全部作为Mask_miss屏蔽
 * 该类提供了自动为MPII的图片生成Mask_miss的方法。
 * 注意：该方法仅仅提供了单张图片生成的方法。
 */

/**
 * 单张图片的数据标注结构
 */
template <typename Dtype>
struct MetaDataAll {
  // 图像路径
  string img_path;
  // mask_all -> NONE
  string mask_all_path;
  // mask_miss路径
  string mask_miss_path;
  // 数据集类型：MPII
  string dataset;
  // 图像尺寸
  Size img_size;
  // unused, FALSE
  bool isValidation;
  // 图片中的总人数
  int numPeople;
  // 图像ID
  int image_idx;
  // 所有实例的中心位置
  vector<Point2f> objpos;
  // 所有实例的scale
  vector<Dtype> scale;
  // 所有实例的面积
  vector<Dtype> area;
  // 所有实例的关节点信息
  vector<Joints> joint;
  // 所有实例的boxes
  vector<BoundingBox<Dtype> > bbox;
};

template <typename Dtype>
class MpiiMaskGenerator {
public:
  /**
   * 构造方法：
   * box_xml_path：包含box信息的XML文件　
   * kps_xml_path：包含kps信息的XML文件
   * image_folder：图片根目录
   * output_folder：输出目录
   * xml_idx：XML编号
   */
  MpiiMaskGenerator(const std::string& box_xml_path,
                    const std::string& kps_xml_path,
                    const std::string& image_folder,
                    const std::string& output_folder,
                    const int xml_idx);

  /**
   * 生成方法
   * @param  save_image [是否保存图片]
   * @param  show_box   [是否要显示box]
   * @return            [返回下一个Mask的编号]
   */
  int Generate(const bool save_image, const bool show_box);

private:
  // 获取boxes列表
  bool ReadBoxes(const std::string& box_xml_path, vector<BoundingBox<Dtype> >* bboxes);
  // 获取kps信息
  void ReadKps(const std::string& kps_xml_path, MetaDataAll<Dtype>* kps);
  // 加载图像数据
  void LoadImage(const std::string& image_path, cv::Mat* image);
  // 获取点数
  int Get_points(const Joints& joint, const BoundingBox<Dtype>& box);
  void ResizeBbox(const Joints& joint, const BoundingBox<Dtype>& box, BoundingBox<Dtype>* resized_box);
  int GetMaskPoints(const Joints& joint, const cv::Mat& mask);
  // 完成boxes和kps之间的匹配，更新kps的box和area信息，匹配后的boxes全部删掉
  bool Match(const vector<BoundingBox<Dtype> >& bboxes, MetaDataAll<Dtype>* kps,
             vector<bool>* matched);
  // 使用kps的长宽信息，以及剩余的boxes信息，生成Mask
  void GenMask(const MetaDataAll<Dtype>& kps, const vector<BoundingBox<Dtype> >& bboxes,
               const vector<bool>& matched, cv::Mat* mask);
  //　分成多人
  void Split(const MetaDataAll<Dtype>& kps, const cv::Mat& mask, vector<MetaData<Dtype> >* metas);
  // output_mask_folder -> 输出目录
  void SaveMask(const MetaDataAll<Dtype>& kps, const cv::Mat& mask);
  // 保存xml
  void SaveXml(const MetaData<Dtype>& meta, const int idx);
  // 保存所有的xml
  int SaveXml(const vector<MetaData<Dtype> >& metas);
  // 保存ｉｍａｇｅ
  void SaveImage(const MetaDataAll<Dtype>& kps, const cv::Mat& mask, const bool show_box);

  // xml path
  std::string box_xml_path_;
  std::string kps_xml_path_;
  // image folder
  std::string image_folder_;
  // output folder
  std::string output_folder_;
  // used for saving xml
  int xml_idx_;
  // used for saving
  const std::string mask_folder_ = "mask/";
  const std::string img_folder_ = "visualize/";
  const std::string xml_folder_ = "xml/";
};

}

#endif
