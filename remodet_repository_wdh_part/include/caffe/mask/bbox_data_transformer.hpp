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
      : param_(param), phase_(phase)   {
       InitRand();   
        } // func: 初始化统一增广参数, 设定随机种子
  virtual ~BBoxDataTransformer() {}
 

  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, vector<BBoxData<Dtype> >* bboxes,
                 BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox, cv::Mat& bg_img);
  void Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox);
  void randomCrop(AnnoData<Dtype>& anno, cv::Mat* image);
  void randomExpand(AnnoData<Dtype>& anno, cv::Mat* image);

  void getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox);

  void TransCrop(AnnoData<Dtype>& anno,const BoundingBox<Dtype>& crop_bbox,cv::Mat* image, BoundingBox<Dtype>* image_bbox);

  void randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno);

  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, bool* doflip,BoundingBox<Dtype>* image_bbox);
  void randomFlip(AnnoData<Dtype>& anno, cv::Mat& image, bool* doflip);
  void randomXFlip(cv::Mat &image, AnnoData<Dtype>& anno, BoundingBox<Dtype>* image_bbox);
  void FlipBoxes(vector<BBoxData<Dtype> >* boxes) {
    if (boxes->size() == 0) return;
    for (int i = 0; i < boxes->size(); ++i) {
      BoundingBox<Dtype>& box = (*boxes)[i].bbox;
      Dtype x1 = box.x1_;
      box.x1_ = Dtype(1.) - box.x2_;
      box.x2_ = Dtype(1.)- x1;
    }
  } 

  // =====================================
  // 抠取gt 
  void borderImage(cv::Mat& image);
  void unNormalize(AnnoData<Dtype>& anno);
  void RelatGtBbox(AnnoData<Dtype> anno, BoundingBox<Dtype>& relat_gt_bbox);
  void changeInstance(AnnoData<Dtype> anno, AnnoData<Dtype>& temp_anno, 
    Dtype x_relat_bbox, Dtype y_relat_bbox, BoundingBox<Dtype> relat_gt_bbox, Dtype scale);
  void cutGtImgChangeBg(cv::Mat raw_image, cv::Mat& bg_img, AnnoData<Dtype> anno, AnnoData<Dtype> temp_anno);
  void normRelatGtBox(BoundingBox<Dtype>& temp_box, BoundingBox<Dtype> box);
  void instancesFitBg(cv::Size shape_bg_img, BoundingBox<Dtype>& relat_gt_bbox, 
  AnnoData<Dtype>& temp_anno , AnnoData<Dtype> anno);
  void cutGtChangeBg(AnnoData<Dtype>& anno , cv::Mat& raw_image, cv::Mat& bg_img);
  void ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                                                  AnnoData<Dtype>* anno);

  // 背光增广 
  void RandomBacklight(cv::Mat& image, AnnoData<Dtype> anno, int skip_cid);
  // =====================================
  void Normalize(AnnoData<Dtype>& anno);

  void fixedResize(AnnoData<Dtype>& anno, cv::Mat& image);

  void copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes);

  void visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes);
  void visualize(AnnoData<Dtype>& anno, cv::Mat& image);

  void InitRand();  

  int Rand(int n);


  // 旋转增广
  void ninetyAngle(AnnoData<Dtype>& anno, cv::Mat& image);
  void rotate90(cv::Mat& image, AnnoData<Dtype>& anno);


  // 转换参数
  UnifiedTransformationParameter param_;
  // 随机数
  shared_ptr<Caffe::RNG> rng_;
  // train / test
  Phase phase_;   

};

}

#endif
