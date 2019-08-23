#ifndef CAFFE_MASK_UNIFIED_DATA_TRANSFORMER_HPP_
#define CAFFE_MASK_UNIFIED_DATA_TRANSFORMER_HPP_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracker/bounding_box.hpp"

#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"
#include "caffe/mask/sample.hpp"

#include "caffe/mask/bbox_func.hpp"

namespace caffe {

/**
 * 该类为数据转换器，提供了从图片和标注数据到网络输入的所有方法。
 */

/**
 * Mask标注数据结构
 */
template <typename Dtype>
struct MaskData {
  int bindex = 0;
  int cid = 0;
  int pid = 0;
  bool is_diff = false;
  bool iscrowd = false;
  bool has_mask = false;
  cv::Mat mask;
};

/**
 * BBox标注数据结构
 */
template <typename Dtype>
struct BBoxData {
  int bindex = 0;
  int cid = 0;
  int pid = 0;
  bool is_diff = false;
  bool iscrowd = false;
  bool ignore_gt = false;
  BoundingBox<Dtype> bbox;
};

/**
 * Kps标注数据结构
 */
template <typename Dtype>
struct KpsData {
  int bindex = 0;
  int cid = 0;
  int pid = 0;
  bool is_diff = false;
  bool iscrowd = false;
  bool has_kps = false;
  int num_kps = 0;
  Joints joint;
};

template <typename Dtype>
class UnifiedDataTransformer {
 public:
  // 构造函数
  explicit UnifiedDataTransformer(const UnifiedTransformationParameter& param, Phase phase)
      : param_(param), phase_(phase) { InitRand(); }
  virtual ~UnifiedDataTransformer() {}

  /**
   * 转换API
   * @param anno   [标注列表]
   * @param image  [返回图像]
   * @param bboxes [GT boxes列表]
   * @param kpses  [Kps 列表]
   * @param masks  [Mask　列表]
   */
  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, vector<BBoxData<Dtype> >* bboxes,
                 vector<KpsData<Dtype> >* kpses, vector<MaskData<Dtype> >* masks,
                 BoundingBox<Dtype>* crop_bbox, bool* doflip);

  /**
   * 转换标注中的关节点
   * @param anno [标注]
   */
  void TransformAnnoJoints(AnnoData<Dtype>& anno);

  /**
   * 转换关节点
   * @param j       [关节点]
   * @param dataset [数据集类型，COCO/MPII]
   */
  void TransformJoints(Joints& j, const string& dataset);

  /**
   * 转换关节点的左右顺序　【在FLIP操作中使用】
   * @param j [关节点]
   */
  void swapLeftRight(Joints& j);

  /**
   * 随机裁剪过程
   * @param anno      [标注]
   * @param image     [图像]
   * @param mask_data [Mask]
   */
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, vector<MaskData<Dtype> >* mask_data, BoundingBox<Dtype>* crop_bbox);

  /**
   * 获取裁剪的box
   * @param anno      [标注]
   * @param crop_bbox [裁剪box]
   */
  void getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox);

  /**
   * 转化裁剪后的关节点坐标
   * @param j         [关节点]
   * @param crop_bbox [擦肩box]
   */
  void kps_crop(Joints& j, const BoundingBox<Dtype>& crop_bbox);

  /**
   * 裁剪过程
   * @param anno      [标注]
   * @param crop_bbox [裁剪box]
   * @param image     [图像]
   * @param mask_data [Mask]
   */
  void TransCrop(AnnoData<Dtype>& anno,const BoundingBox<Dtype>& crop_bbox,cv::Mat* image,vector<MaskData<Dtype> >* mask_data);

  /**
   * bboxes裁剪过程
   * @param crop_bbox    [裁剪的ROI]
   * @param boxes        [输入boxes]
   * @param trans_bboxes [输出boxes]
   */
  void ApplyCrop(const BoundingBox<Dtype>& crop_bbox, vector<LabeledBBox<Dtype> >& boxes, vector<BBoxData<Dtype> >* trans_bboxes);

  /**
   * 颜色失真处理
   * @param image [图像]
   * @param anno  [标注]
   */
  void randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno);

  /**
   * 随机Flip操作
   * @param anno      [标注]
   * @param image     [图像]
   * @param mask_data [mask]
   */
  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, vector<MaskData<Dtype> >& mask_data, bool* doflip);

  void FlipBoxes(vector<BBoxData<Dtype> >* boxes) {
    if (boxes->size() == 0) return;
    for (int i = 0; i < boxes->size(); ++i) {
      BoundingBox<Dtype>& box = (*boxes)[i].bbox;
      Dtype x1 = box.x1_;
      box.x1_ = Dtype(1.) - box.x2_;
      box.x2_ = Dtype(1.)- x1;
    }
  }
  void FlipBoxes(vector<LabeledBBox<Dtype> >* boxes) {
    if (boxes->size() == 0) return;
    for (int i = 0; i < boxes->size(); ++i) {
      BoundingBox<Dtype>& box = (*boxes)[i].bbox;
      Dtype x1 = box.x1_;
      box.x1_ = Dtype(1.) - box.x2_;
      box.x2_ = Dtype(1.) - x1;
    }
  }

  /**
   * 坐标归一化
   * @param anno [标注]
   */
  void Normalize(AnnoData<Dtype>& anno);

  /**
   * resize过程
   * @param anno      [标注]
   * @param image     [图像]
   * @param mask_data [mask]
   */
  void fixedResize(AnnoData<Dtype>& anno, cv::Mat& image, vector<MaskData<Dtype> >& mask_data);

  /**
   * label复制：使用标注信息生成Boxes/Kps标注数据结构
   * @param anno  [标注信息]
   * @param boxes [boxes数据结构]
   * @param kpses [kpses数据结构]
   */
  void copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes, vector<KpsData<Dtype> >* kpses);

  /**
   * 可视化过程
   * @param anno      [标注]
   * @param image     [图像]
   * @param mask_data [mask]
   */
  void visualize(AnnoData<Dtype>& anno, cv::Mat& image, vector<MaskData<Dtype> >& mask_data);

  void visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes,
                 vector<KpsData<Dtype> >& kpses, vector<MaskData<Dtype> >& masks);

  /**
   * 随机种子初始化
   */
  void InitRand();

  /**
   * 生成随机数：在0-n-1之间随机生成一个整数
   * @param  n [N]
   * @return   [随机数]
   */
  int Rand(int n);

  // 转换参数
  UnifiedTransformationParameter param_;
  // 随机数
  shared_ptr<Caffe::RNG> rng_;
  // train / test
  Phase phase_;
};

}

#endif
