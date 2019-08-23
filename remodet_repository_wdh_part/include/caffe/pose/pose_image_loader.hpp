#ifndef CAFFE_POSE_POSE_IMAGE_LOADER_H
#define CAFFE_POSE_POSE_IMAGE_LOADER_H

#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tracker/bounding_box.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"

using namespace cv;

namespace caffe {

/**
 * 该类提供了将姿态标注数据进行可视化的一系列方法。
 * 该类还提供了实例的数据结构
 */

/**
 * 关节点数据集合，18点
 */
struct Joints {
  vector<Point2f> joints;
  vector<int> isVisible;
};

/**
 * 单张图片的标注数据集合
 */
template <typename Dtype>
struct MetaData {
  // 图片路径
  string img_path;
  // Mask路径　　　　　　　【所有前景】
  string mask_all_path;
  // 需要打马的Mask路径　　【没有标记kps的(person)前景】
  string mask_miss_path;
  // 数据集类型：COCO/MPII
  string dataset;
  // 图像尺寸
  Size img_size;
  // unused.
  bool isValidation;
  // 除去主要对象外的人数
  int numOtherPeople;
  // 该张图片中的person index
  int people_index;
  // 标注文件的编号，unused
  int annolist_index;
  // 主要对象的中心位置
  Point2f objpos;
  // 主要对象的面积
  Dtype area;
  // 主要对象的scale
  Dtype scale_self;
  // 主要对象的关键点信息
  Joints joint_self;
  // 主要对象的box信息
  BoundingBox<Dtype> bbox;

  // 其他对象的中心点信息
  vector<Point2f> objpos_other;
  // 其他对象的scale信息
  vector<Dtype> scale_other;
  // 其他对象的面积信息
  vector<Dtype> area_other;
  // 其他对象的关节点信息
  vector<Joints> joint_others;
  // 其他对象的box信息
  vector<BoundingBox<Dtype> > bbox_others;
};

template <typename Dtype>
class PoseImageLoader {
public:
  // 构造：在派生类中完成具体构造
  // 注意：该类作为其他类的基类
  PoseImageLoader() {}

  /**
   * 加载图片
   * @param image_num [图片id]
   * @param image     [图片cv::Mat]
   */
  void LoadImage(const int image_num, cv::Mat* image);

  /**
   * 加载图片和标注信息
   * @param image_num [图片id]
   * @param image     [图片]
   * @param meta      [标注信息]
   */
  void LoadAnnotation(const int image_num,
                      cv::Mat* image,
                      MetaData<Dtype>* meta);
  // 显示图片
  void ShowImages();

  /**
   * 显示图片和标注信息
   * @param show_bbox [是否显示box]
   */
  void ShowAnnotations(const bool show_bbox);

  /**
   * 随机显示图片和对应标注
   * @param show_bbox [是否显示box]
   */
  void ShowAnnotationsRand(const bool show_bbox);

  /**
   * 保存图片和标注信息
   * @param output_folder [保存的目录]
   * @param show_bbox     [是否显示box]
   */
  void Saving(const std::string& output_folder, const bool show_bbox);

  /**
   * 获取标注信息集合
   */
  std::vector<MetaData<Dtype> > get_annotations() const { return annotations_; }

  /**
   * 获取第三方的标注数据
   * @param dst [第三方Loader]
   */
  void merge_from(const PoseImageLoader<Dtype>* dst);

  /**
   * 标注信息条目的总数
   * @return [数量]
   */
  int size() const { return annotations_.size(); }

protected:
  /**
   * 绘制标注信息
   * @param image_num [图片id]
   * @param show_bbox [是否绘制box]
   * @param dst_image [图像cv::Mat]
   */
  void drawAnnotations(const int image_num, const bool show_bbox, cv::Mat* dst_image);

  /**
   * 绘制box
   * @param meta      [标注信息]
   * @param image_out [图像cv::Mat]
   */
  void drawbox(const MetaData<Dtype>& meta, cv::Mat* image_out);

  /**
   * 绘制关节点
   * @param meta      [标注信息]
   * @param image_out [图像cv::Mat]
   */
  void drawkps(const MetaData<Dtype>& meta, cv::Mat* image_out);

  // 标注信息条目集合
  std::vector<MetaData<Dtype> > annotations_;
};

}

#endif
