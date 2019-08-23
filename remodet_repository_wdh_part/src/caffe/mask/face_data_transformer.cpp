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

#include "caffe/mask/face_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {

template <typename Dtype>
void FaceDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                              vector<BBoxData<Dtype> >* bboxes,
                                              BoundingBox<Dtype>* crop_bbox,
                                              bool* doflip) {
  // 坐标归一化
  Normalize(anno);
  // 裁剪过程
  randomCrop(anno, image,crop_bbox);
  CHECK(image->data);
  // 失真
  randomDistortion(image, anno);
  // flip
  randomFlip(anno, *image, doflip);
  // resized
  fixedResize(anno, *image);
  // save to bboxes & kpses
  copy_label(anno, bboxes);
  // LOG(INFO)<<"cv rows"<<image->rows;
  // visualize
  if (param_.visualize()) {
    visualize(anno, *image);
  }
}



// 随机裁剪/scale
template <typename Dtype>
void FaceDataTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image,
                                               BoundingBox<Dtype>* crop_bbox) {
  
  // image
  *image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  // 获取随机的裁剪bbox
  getCropBBox(anno, crop_bbox);
  // 转换标注：bbox/kps/生成mask信息，以及裁剪image
  TransCrop(anno,*crop_bbox,image);
}

template <typename Dtype>
void FaceDataTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
    // get a random value [0-1]
    // float prob = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // get max width & height
    float h_max, w_max;
    h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
    w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
    // if(prob > 0.5) {
    //   // 16:9
    //   h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
    //   w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
    // } else {
    //   // 9:16
    //   h_max = std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
    //   w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
    // }
    // get sampler
    if (phase_ == TRAIN) {
      if (param_.batch_sampler_size() == 0) {
        LOG(FATAL) << "In training-phase, at least one batch_sampler should be defined in random-crop augmention.";
      }
      vector<BoundingBox<Dtype> > sample_bboxes;
      vector<BatchSampler> samplers;
      for (int s = 0; s < param_.batch_sampler_size(); ++s) {
        samplers.push_back(param_.batch_sampler(s));
      }
      GenerateBatchSamples16_9(anno, samplers, &sample_bboxes, h_max, w_max);
      if (sample_bboxes.size() > 0) {
        int idx = caffe_rng_rand() % sample_bboxes.size();
        *crop_bbox = sample_bboxes[idx];
      } else {
        crop_bbox->x1_ = 0.5-w_max/2.0;
        crop_bbox->x2_ = 0.5+w_max/2.0;
        crop_bbox->y1_ = 0.5-h_max/2.0;
        crop_bbox->y2_ = 0.5+h_max/2.0;
      }
    } else {
      crop_bbox->x1_ = 0.5-w_max/2.0;
      crop_bbox->x2_ = 0.5+w_max/2.0;
      crop_bbox->y1_ = 0.5-h_max/2.0;
      crop_bbox->y2_ = 0.5+h_max/2.0;
    }
}



template <typename Dtype>
void FaceDataTransformer<Dtype>::ApplyCrop(const BoundingBox<Dtype>& crop_bbox,
    vector<LabeledBBox<Dtype> >& boxes, vector<BBoxData<Dtype> >* trans_bboxes) {
  trans_bboxes->clear();
  if (boxes.size() == 0) return;
  for (int i = 0; i < boxes.size(); ++i) {
    const LabeledBBox<Dtype>& lbox = boxes[i];
    const BoundingBox<Dtype>& gtbox = lbox.bbox;
    BoundingBox<Dtype> proj_bbox;
    // 满足发布条件
    if (gtbox.project_bbox(crop_bbox, &proj_bbox) >= param_.emit_coverage_thre()) {
      BBoxData<Dtype> out_box;
      out_box.bbox = proj_bbox;
      out_box.bindex = lbox.bindex;
      out_box.cid = lbox.cid;
      out_box.pid = lbox.pid;
      trans_bboxes->push_back(out_box);
    }
  }
}

/**
 * 完成裁剪工作
 * 1. 所有实例完成裁剪 [bbox/kps]
 * 2. 图像完成裁剪
 * 3. 为每个保留的实例生成Masks
 */
template <typename Dtype>
void FaceDataTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,const BoundingBox<Dtype>& crop_bbox,cv::Mat* image) {
  const int img_width = image->cols;
  const int img_height = image->rows;
  // modify header
  anno.img_width = (int)(img_width * (crop_bbox.get_width()));
  anno.img_height = (int)(img_height * (crop_bbox.get_height()));
  // image crop
  int w_off_int = (int)(crop_bbox.x1_ * img_width);
  int h_off_int = (int)(crop_bbox.y1_ * img_height);
  int crop_w_int = (int)(img_width * (crop_bbox.get_width()));
  int crop_h_int = (int)(img_height * (crop_bbox.get_height()));
  cv::Rect roi(w_off_int, h_off_int, crop_w_int, crop_h_int);
  cv::Mat image_back = image->clone();
  *image = image_back(roi);
  // scan all instances, delete the crop boxes
  int num_keep = 0;
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    BoundingBox<Dtype> proj_bbox;
    // 实例保留
    if (gt_bbox.project_bbox(crop_bbox, &proj_bbox) >= param_.emit_coverage_thre()) {
      // box更新
      it->bbox = proj_bbox;
      ++num_keep;
      ++it;
    } else {
      it = anno.instances.erase(it);
    }
  }
  anno.num_person = num_keep;
}

template <typename Dtype>
void FaceDataTransformer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void FaceDataTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,bool* doflip) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  *doflip = (dice <= param_.flip_prob());
  if (*doflip) {
    cv::flip(image, image, 1);
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
void FaceDataTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
  // bbox以及kps全部归一化
  const int image_width = anno.img_width;
  const int image_height = anno.img_height;
  // 所有实例
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    Joints& j = anno.instances[i].joint;
    bbox.x1_ /= (Dtype)image_width;
    bbox.x2_ /= (Dtype)image_width;
    bbox.y1_ /= (Dtype)image_height;
    bbox.y2_ /= (Dtype)image_height;
    for (int k = 0; k < j.joints.size(); ++k) {
      j.joints[k].x /= (Dtype)image_width;
      j.joints[k].y /= (Dtype)image_height;
    }
  }
}

template <typename Dtype>
void FaceDataTransformer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes
                                               ) {
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
void FaceDataTransformer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
  // 修改header
  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // LOG(INFO)<<"hzw"<<anno.img_width;
  // LOG(INFO)<<"hzw"<<anno.img_height;
  // resize 图像
  cv::Mat image_rsz;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;
  // resize mask
}


// 显示函数
template<typename Dtype>
void FaceDataTransformer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255,0,0,0,255,0,0,0,255,255,0,128,0,128,128,128,128,255};
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    int r = color_maps[3*(i % 6)];
    int g = color_maps[3*(i % 6) + 1];
    int b = color_maps[3*(i % 6) + 2];
    // 绘制box
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
  // LOG(INFO)<<"rows"<<image.rows;
  char imagename [256];
  sprintf(imagename, "%s/augment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

template <typename Dtype>
void FaceDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int FaceDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(FaceDataTransformer);

}  // namespace caffe
