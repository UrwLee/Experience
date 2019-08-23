#ifndef CAFFE_MASK_BBOX_DATA_TRANSFORMER_HANDPOSE_HPP_
#define CAFFE_MASK_BBOX_DATA_TRANSFORMER_HANDPOSE_HPP_

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
#include "caffe/mask/unified_data_transformer.hpp"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

template <typename Dtype>
class BBoxDataHandKeypointTransformer {
 public:
  // 构造函数
  // 我们仍然使用UnifiedTransformationParameter进行构造
  explicit BBoxDataHandKeypointTransformer(const UnifiedTransformationParameter& param, Phase phase)
      : param_(param), phase_(phase) { InitRand(); }
  virtual ~BBoxDataHandKeypointTransformer() {}

  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, vector<BBoxData<Dtype> >* bboxes,
                 BoundingBox<Dtype>* crop_bbox, bool* doflip,Dtype* transformed_heatmap);

  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image);
  void RotatePoint(Point2f& p, Mat& R);
  void getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox);

  void TransCrop(AnnoData<Dtype>& anno,const BoundingBox<Dtype>& crop_bbox,cv::Mat* image);

  void randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno);

  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, bool* doflip);

  void FlipBoxes(vector<BBoxData<Dtype> >* boxes) {
    if (boxes->size() == 0) return;
    for (int i = 0; i < boxes->size(); ++i) {
      BoundingBox<Dtype>& box = (*boxes)[i].bbox;
      Dtype x1 = box.x1_;
      box.x1_ = Dtype(1.) - box.x2_;
      box.x2_ = Dtype(1.)- x1;
    }
  }

  void Normalize(AnnoData<Dtype>& anno);

  void fixedResize(AnnoData<Dtype>& anno, cv::Mat& image);
  void augmentation_rotate(AnnoData<Dtype>& anno, cv::Mat& image);
  void copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes);

  void visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes);
  void visualize(AnnoData<Dtype>& anno, cv::Mat& image);
  void visualize(cv::Mat& image, AnnoData<Dtype>& anno);

  void test(AnnoData<Dtype>& anno, cv::Mat& image);
  void get_mask(AnnoData<Dtype>& anno, cv::Mat& image,Dtype* transformed_heatmap);
  void generateLabelMap(Dtype* transformed_heatmap, cv::Mat& img_aug, AnnoData<Dtype>& anno);
  void putGaussianMaps(Dtype* map, Point2f center, int stride, int grid_x, int grid_y, float sigma);




  void InitRand();

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
