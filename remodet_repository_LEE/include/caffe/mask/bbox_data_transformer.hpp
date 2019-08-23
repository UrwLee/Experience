#ifndef CAFFE_MASK_BBOX_DATA_TRANSFORMER_HPP_
#define CAFFE_MASK_BBOX_DATA_TRANSFORMER_HPP_

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
class BBoxDataTransformer {
public:
  // 构造函数
  // 我们仍然使用UnifiedTransformationParameter进行构造
  explicit BBoxDataTransformer(const UnifiedTransformationParameter& param, Phase phase)
    : param_(param), phase_(phase) { InitRand(); }
  virtual ~BBoxDataTransformer() {}

  void Pic916To11(AnnoData<Dtype>& anno, cv::Mat& image);
  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, vector<BBoxData<Dtype> >* bboxes,
                 BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox);
  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, vector<BBoxData<Dtype> >* bboxes,
                 BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes);
  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image);
  void randomExpand(AnnoData<Dtype>& anno, cv::Mat* image);

  void getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox);

  void TransCrop(AnnoData<Dtype>& anno, const BoundingBox<Dtype>& crop_bbox, cv::Mat* image, BoundingBox<Dtype>* image_bbox);
  void TransCrop(AnnoData<Dtype>& anno, const BoundingBox<Dtype>& crop_bbox, cv::Mat* image, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes);

  void randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno);
  void randomDistortion_neg(cv::Mat* image, AnnoData<Dtype>& anno);

  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, bool* doflip, BoundingBox<Dtype>* image_bbox);
  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, bool* doflip);
  void rotate90(cv::Mat& image, AnnoData<Dtype>& anno);
  void rotate90_standPerson(cv::Mat &image, AnnoData<Dtype>& anno);
  void blur(AnnoData<Dtype>& anno, cv::Mat& image);
  void partBlur(AnnoData<Dtype>& anno, cv::Mat& image);
  void randomXFlip(cv::Mat& image, AnnoData<Dtype>& anno, BoundingBox<Dtype>* image_bbox);
  void randomPerspective(AnnoData<Dtype>& anno, cv::Mat& image) ;

  void FlipBoxes(vector<BBoxData<Dtype> >* boxes) {
    if (boxes->size() == 0) return;
    for (int i = 0; i < boxes->size(); ++i) {
      BoundingBox<Dtype>& box = (*boxes)[i].bbox;
      Dtype x1 = box.x1_;
      box.x1_ = Dtype(1.) - box.x2_;
      box.x2_ = Dtype(1.) - x1;
    }
  }

  void Normalize(AnnoData<Dtype>& anno);

  void fixedResize(AnnoData<Dtype>& anno, cv::Mat& image);

  void copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes);

  int StorePreson(cv::Mat* image, AnnoData<Dtype>& anno);
  int StoreHair(cv::Mat* image, AnnoData<Dtype>& anno);
  void AddAdditionPreson(AnnoData<Dtype>& anno, cv::Mat* image);
  void AddAdditionHair(AnnoData<Dtype>& anno, cv::Mat* image);


  void arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y);

  void adjust_gama(Dtype gama, cv::Mat &image);

  void gama_com_neg(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image);
  void gama_com(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image);

  void randomBlock(AnnoData<Dtype>& anno, cv::Mat* image);
  void RandomBacklight(cv::Mat& image, AnnoData<Dtype> anno);

  void visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes);
  void visualize(AnnoData<Dtype>& anno, cv::Mat& image);

  void InitRand();

  int Rand(int n);

  // 转换参数
  UnifiedTransformationParameter param_;
  // 随机数
  shared_ptr<Caffe::RNG> rng_;
  // train / test
  Phase phase_;
  //储存的单人图片和part信息
  std::vector<pair<cv::Mat, std::vector<Instance<Dtype> > > > StoreSingle_;
  std::vector<cv::Mat > StoreSingleHair_;
};

}

#endif
