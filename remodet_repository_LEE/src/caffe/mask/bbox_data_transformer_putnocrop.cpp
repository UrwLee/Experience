#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

#include "caffe/mask/bbox_data_transformer_putnocrop.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                    vector<BBoxData<Dtype> >* bboxes,
                                    BoundingBox<Dtype>* crop_bbox, bool* doflip) {
  Normalize(anno);
  putImagetoFixedResize(anno, image);
  // randomly crpp
  CHECK(image->data);
  // perform distortion
  randomDistortion(image, anno);
  // // flip
  randomFlip(anno, *image, doflip);
  // // save to bboxes
  copy_label(anno, bboxes);
  // // visualize
  if (param_.visualize()) {
    visualize(*image, *bboxes);
  }
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip) {
  putImagetoFixedResize(anno, image);

  // randomly cropexpand_ratio
  CHECK(image->data);
  // perform distortion
  randomDistortion(image, anno);
  // // flip
  randomFlip(anno, *image, doflip);
  // // resized
}

//###########################################################
template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::putImagetoFixedResize(AnnoData<Dtype>& anno, cv::Mat* image) {
    // 读入图片
*image = cv::imread(anno.img_path.c_str());
   // 读入图片
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  //LOG(FATAL) << "Error when open image file: " << anno.img_path;
//
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  float scale_h, scale_w;
  scale_w = (float)param_.resized_width()/(float)image->cols;
  scale_h = (float)param_.resized_height()/(float)image->rows;
  float scale = std::min(scale_w,scale_h);
   cv::Mat image_rsz;
  resize(*image, image_rsz, Size(),scale,scale, INTER_LINEAR);
  BoundingBox<Dtype> expand_bbox;
  expand_bbox.x1_=0.0;
  expand_bbox.y1_=0.0;
  expand_bbox.x2_=(float)image_rsz.cols/(float)param_.resized_width();
  expand_bbox.y2_=(float)image_rsz.rows/(float)param_.resized_height();
  //图像转换
  cv::Mat expand_img;
  expand_img.create(param_.resized_height(), param_.resized_width(), image->type());
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
  expand_img.setTo(cv::Scalar(104,117,123,0.0));
  cv::Rect bbox_roi(0, 0, image_rsz.cols, image_rsz.rows);
  image_rsz.copyTo(expand_img(bbox_roi));
  *image = expand_img;

  ///改变anno
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {

    BoundingBox<Dtype>& gt_bbox = it->bbox;
    BoundingBox<Dtype> proj_bbox;
    // keep instance/ emitting true ground truth
    Dtype emit_coverage_thre = 1;
    if (gt_bbox.project_bbox(expand_bbox, &proj_bbox) >=emit_coverage_thre) {
      //判断的同时，project_bbox记录了gt相对于expand_bbox的坐标
      it->bbox = proj_bbox;
        ++it;
      }

  }
  //for test
  bool test=false;
  if(test){
  int i=fabs(caffe_rng_rand()%1000);
  std::stringstream ss;
  std::string str;
  ss<<i;
  ss>>str;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype> proj_bbox = it->bbox;
   cv::rectangle(*image,cvPoint(int(proj_bbox.x1_*param_.resized_width()),int(proj_bbox.y1_*param_.resized_height())),cvPoint(int(proj_bbox.x2_*param_.resized_width()),int(proj_bbox.y2_*param_.resized_height())),Scalar(255,0,0),1,1,0);
   ++it;
  }
  cv::imwrite("/home/xjx/xjx/AIC/"+str+".jpg",*image);
}
}
//#############################################################


template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
                            bool* doflip) {
  //生成０－１随机数
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  *doflip = (dice <= param_.flip_prob());
  if (*doflip) {
    cv::flip(image, image, 1);//水平翻转
    for (int i = 0; i < anno.instances.size(); ++i) {
      BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
      // bbox
      Dtype temp = bbox.x2_;
      bbox.x2_ = 1.0 - bbox.x1_;
      bbox.x1_ = 1.0 - temp;
    }
  }
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
 // bboxÒÔ¼°kpsÈ«²¿¹éÒ»»¯
  const int image_width = anno.img_width;
  const int image_height = anno.img_height;
  // ËùÓÐÊµÀý
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    bbox.x1_ /= (Dtype)image_width;
    bbox.x2_ /= (Dtype)image_width;
    bbox.y1_ /= (Dtype)image_height;
    bbox.y2_ /= (Dtype)image_height;
  }
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes) {
  boxes->clear();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    BBoxData<Dtype> bbox_element;
    Instance<Dtype>& ins = anno.instances[i];
    // bbox
    bbox_element.bindex = ins.bindex;
    bbox_element.cid = ins.cid;
    bbox_element.pid = ins.pid;
    bbox_element.is_diff = ins.is_diff;
    bbox_element.iscrowd = ins.iscrowd;
    bbox_element.bbox = ins.bbox;
    boxes->push_back(bbox_element);
  }
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
  // modify header
  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // resize image
  cv::Mat image_rsz;
  // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
  // CHECK(image.data);
  // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;
}

template<typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255,0,0,0,255,0,0,0,255,255,0,128,0,128,128,128,128,255};
  for (int i = 0; i < boxes.size(); ++i) {
    BBoxData<Dtype>& box = boxes[i];
    BoundingBox<Dtype>& tbox = box.bbox;
    BoundingBox<Dtype> bbox_real;
    bbox_real.x1_ = tbox.x1_ * img_vis.cols;
    bbox_real.y1_ = tbox.y1_ * img_vis.rows;
    bbox_real.x2_ = tbox.x2_ * img_vis.cols;
    bbox_real.y2_ = tbox.y2_ * img_vis.rows;
    if (box.iscrowd && (box.cid == 0)) continue;
    const int cid = box.cid;
    int r = color_maps[3*(cid % 6)];
    int g = color_maps[3*(cid % 6) + 1];
    int b = color_maps[3*(cid % 6) + 2];
    bbox_real.Draw(r,g,b,&img_vis);
  }
  char imagename [256];
  sprintf(imagename, "%s/augment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

// show function
template<typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255,0,0,0,255,0,0,0,255,255,0,128,0,128,128,128,128,255};
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    int r = color_maps[3*(i % 6)];
    int g = color_maps[3*(i % 6) + 1];
    int b = color_maps[3*(i % 6) + 2];
    // draw box
    BoundingBox<Dtype> bbox_real;
    bbox_real.x1_ = bbox.x1_ * img_vis.cols;
    bbox_real.y1_ = bbox.y1_ * img_vis.rows;
    bbox_real.x2_ = bbox.x2_ * img_vis.cols;
    bbox_real.y2_ = bbox.y2_ * img_vis.rows;
    if (anno.instances[i].iscrowd) {
      bbox_real.Draw(0,0,0,&img_vis);
    } else {
      bbox_real.Draw(r,g,b,&img_vis);
    }
  }
  char imagename [256];
  sprintf(imagename, "%s/augment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

template <typename Dtype>
void BBoxDataTransformerPutNoCropLayer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int BBoxDataTransformerPutNoCropLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

//INSTANTIATE_CLASS(BBoxDataTransformerPutNoCropLayerNoAug);
INSTANTIATE_CLASS(BBoxDataTransformerPutNoCropLayer);
// REGISTER_LAYER_CLASS(BBoxDataTransformerNoAug);
}  // namespace caffe
