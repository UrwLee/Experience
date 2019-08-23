#ifndef CAFFE_MASK_ANNO_IMAGE_LOADER_H
#define CAFFE_MASK_ANNO_IMAGE_LOADER_H

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

#include "caffe/pose/pose_image_loader.hpp"

using namespace cv;

namespace caffe {

/**
 * 实例（person）的数据结构
 */
template <typename Dtype>
struct Instance {
  // minibatch内部的编号
  int bindex;
  // 类别号：0
  int cid;
  // 类下的实例号
  int pid;
  // 该对象是否是diff？
  bool is_diff;
  // 该对象是否是crowd?
  bool iscrowd;
  // 该对象的Box
  BoundingBox<Dtype> bbox;
  // 该对象是否有mask?
  bool mask_included;
  // 该对象的Mask图片的位置
  string mask_path;
  // 该对象是否有关节点?
  bool kps_included;
  // 该对象的可见关节点数量
  int num_kps;
  // 该对象的关节点数据
  Joints joint;
  // TorsoAndHead BBox
  BoundingBox<Dtype> THbbox;
};

/**
 * 一张图片的标注数据
 * 包含：
 * １．图片路径
 * ２．图片数据集
 * ３．图片尺寸
 * ４．有效的实例数
 * ５．实例的数据结构集合
 */
template <typename Dtype>
struct AnnoData {
  // 图片路径
  string img_path;
  // 数据集
  string dataset;
  // 图片尺寸
  int img_width;
  int img_height;
  // 实例数
  int num_person;
  // 实例数据结构集合
  vector<Instance<Dtype> > instances;
};

/**
 * 图片标注集合
 */
template <typename Dtype>
class AnnoImageLoader {
public:
  AnnoImageLoader() {}

  /**
   * 加载图片
   * @param image_num [图片id]
   * @param image     [图片cv::Mat]
   */
  void LoadImage(const int image_num, cv::Mat* image);

  /**
   * 加载标注信息
   * @param image_num [图片id]
   * @param image     [返回图片cv::Mat]
   * @param anno      [标注]
   */
  void LoadAnnotation(const int image_num,
                      cv::Mat* image,
                      AnnoData<Dtype>* anno);
  // 显示图片
  void ShowImages();
  // 显示图片和标记
  void ShowAnnotations();
  // 随机显示图片和标记
  void ShowAnnotationsRand();
  // 保存至输出目录
  void Saving(const std::string& output_folder);
  // 获取标注
  std::vector<AnnoData<Dtype> > get_annotations() const { return annotations_; }
  // 获取第三方的样本数据
  void merge_from(const AnnoImageLoader<Dtype>* dst);
  // 图片size
  int size() const { return annotations_.size(); }
protected:
  /**
   * 绘制标注
   * @param image_num [id]
   * @param dst_image [返回的图片cv::Mat]
   */
  void drawAnnotations(const int image_num, cv::Mat* dst_image);

  /**
   * 绘制box
   * @param anno      [标注信息]
   * @param image_out [绘制的cv::Mat]
   */
  void drawbox(const AnnoData<Dtype>& anno, cv::Mat* image_out);

  /**
   * 绘制kps
   * @param anno      [标注信息]
   * @param image_out [绘制的cv::Mat]
   */
  void drawkps(const AnnoData<Dtype>& anno, cv::Mat* image_out);

  /**
   * 绘制mask
   * @param anno      [标注信息]
   * @param image_out [绘制的cv::Mat]
   */
  void drawmask(const AnnoData<Dtype>& anno, cv::Mat* image_out);

  /**
   * 绘制mask
   * @param mask  [mask图像]
   * @param image [绘制的cv::Mat]
   * @param r     [绘制颜色R]
   * @param g     [绘制颜色G]
   * @param b     [绘制颜色B]
   */
  void drawmask(const cv::Mat& mask, cv::Mat* image, int r, int g, int b);

  /**
   * 所有标注信息集合
   */
  std::vector<AnnoData<Dtype> > annotations_;
};

}

#endif
