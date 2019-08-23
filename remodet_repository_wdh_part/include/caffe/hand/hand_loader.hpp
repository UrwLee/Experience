#ifndef CAFFE_HAND_HAND_LOADER_HPP_
#define CAFFE_HAND_HAND_LOADER_HPP_

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
 * 该类提供了标记手部box的一些方法
 */

/**
 * 每个实例的数据结构
 */
template <typename Dtype>
struct InsData {
  BoundingBox<Dtype> bbox;
  bool kps_included;
  int num_kps;
  Joints joint;
};

/**
 * 图片的标记数据结构，包含多个实例
 */
template <typename Dtype>
struct MData {
  string img_path;
  string dataset;
  int img_width;
  int img_height;
  int num_person;
  vector<InsData<Dtype> > ins;
};

/**
 * 加载Hand的类
 */
template <typename Dtype>
class HandLoader {
public:
  HandLoader() {}

  /**
   * 加载图片
   * @param image_num [图片编号]
   * @param image     [图片cv::Mat]
   */
  void LoadImage(const int image_num, cv::Mat* image);

  /**
   * 加载图片和标注
   * @param image_num [图片编号]
   * @param image     [cv::Mat]
   * @param meta      [标记数据]
   */
  void LoadAnnotation(const int image_num,
                      cv::Mat* image,
                      MData<Dtype>* meta);
  /**
   * 显示手部的box
   */
  void ShowHand();

  /**
   * 保存手部的ROI图片到指定目录
   * 注意：请指定图片文件名的前缀
   * @param output_folder [输出目录]
   * @param prefix        [名称前缀]
   */
  void Saving(const std::string& output_folder, const std::string& prefix);

  /**
   * 获取该Loader的所有标注信息集合
   */
  std::vector<MData<Dtype> > get_annotations() const { return annotations_; }

  /**
   * 获取第三方的标注数据集合
   * @param dst [第三方Loader]
   */
  void merge_from(const HandLoader<Dtype>* dst);

  /**
   * 获取标注条目集合的数量
   * @return [数量]
   */
  int size() const { return annotations_.size(); }

protected:
  /**
   * 根据实例标注信息获取手部box
   * @param ins       [实例]
   * @param width     [图像宽度]
   * @param height    [图像高度]
   * @param box       [返回的box列表]
   * @param num_hands [返回的有效box数量]
   */
  void get_hand_bbox(const InsData<Dtype>& ins, const int width, const int height, vector<BoundingBox<Dtype> >* box, int* num_hands);

  /**
   * 标注信息集合
   */
  std::vector<MData<Dtype> > annotations_;
  /**
   * 数据集类型：COCO & MPII
   */
  std::string type_;
};

}

#endif
