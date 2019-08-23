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

#include "caffe/mask/unified_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

namespace caffe {

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                              vector<BBoxData<Dtype> >* bboxes,
                                              vector<KpsData<Dtype> >* kpses,
                                              vector<MaskData<Dtype> >* masks,
                                              BoundingBox<Dtype>* crop_bbox,
                                              bool* doflip) {
  // 坐标归一化
  TransformAnnoJoints(anno);
  Normalize(anno);
  // 裁剪过程
  randomCrop(anno, image, masks, crop_bbox);
  CHECK(image->data);
  CHECK_EQ(anno.instances.size(), masks->size())
          << "length of masks must equal with the length of instances.";
  // 失真
  randomDistortion(image, anno);
  // flip
  randomFlip(anno, *image, *masks, doflip);
  // resized
  fixedResize(anno, *image, *masks);
  // save to bboxes & kpses
  copy_label(anno, bboxes, kpses);
  // CHECK
  CHECK_EQ(bboxes->size(), kpses->size());
  CHECK_EQ(bboxes->size(), masks->size());
  // visualize
  if (param_.visualize()) {
    visualize(anno, *image, *masks);
  }
}

// 转换关节点数据
template <typename Dtype>
void UnifiedDataTransformer<Dtype>::TransformAnnoJoints(AnnoData<Dtype>& anno) {
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    Instance<Dtype>& ins = anno.instances[i];
    TransformJoints(ins.joint, anno.dataset);
  }
}

// 关节点数据转换方法
template <typename Dtype>
void UnifiedDataTransformer<Dtype>::TransformJoints(Joints& j, const string& dataset) {
  Joints jo = j;
  if (dataset.find("COCO") != std::string::npos) {
    int COCO_to_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(18);
    jo.isVisible.resize(18);
    for(int i = 0; i < 18; i++) {
      jo.joints[i] = (j.joints[COCO_to_1[i]-1] + j.joints[COCO_to_2[i]-1]) * 0.5;
      if(j.isVisible[COCO_to_1[i]-1] >= 2 || j.isVisible[COCO_to_2[i]-1] >= 2) {
        jo.isVisible[i] = 2;
      } else {
        jo.isVisible[i] = j.isVisible[COCO_to_1[i]-1] && j.isVisible[COCO_to_2[i]-1];
      }
    }
  } else {
    LOG(FATAL) << "Not support dataset-type: " << dataset;
  }
  j = jo;
}

// 关节点flip操作
template <typename Dtype>
void UnifiedDataTransformer<Dtype>::swapLeftRight(Joints& j) {
  int right[8] = {3,4,5, 9,10,11,15,17};
  int left[8] =  {6,7,8,12,13,14,16,18};
  CHECK_EQ(j.joints.size(), 18);
  CHECK_EQ(j.joints.size(), j.isVisible.size());
  for(int i = 0; i < 8; i++) {
    int ri = right[i] - 1;
    int li = left[i] - 1;
    Point2f temp = j.joints[ri];
    j.joints[ri] = j.joints[li];
    j.joints[li] = temp;
    int temp_v = j.isVisible[ri];
    j.isVisible[ri] = j.isVisible[li];
    j.isVisible[li] = temp_v;
  }
}

// 随机裁剪/scale
template <typename Dtype>
void UnifiedDataTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, vector<MaskData<Dtype> >* mask_data,
                                               BoundingBox<Dtype>* crop_bbox) {
  mask_data->clear();
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
  TransCrop(anno,*crop_bbox,image,mask_data);
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
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
void UnifiedDataTransformer<Dtype>::kps_crop(Joints& j, const BoundingBox<Dtype>& crop_bbox) {
  const int num = j.joints.size();
  const Dtype width = crop_bbox.get_width();
  const Dtype height = crop_bbox.get_height();
  for (int i = 0; i < num; ++i) {
    Dtype x = (j.joints[i].x - crop_bbox.x1_) / width;
    Dtype y = (j.joints[i].y - crop_bbox.y1_) / height;
    int vis;
    if (x <= 0 || x >= 1 || y <= 0 || y >= 1) {
      vis = 2;
      x = 0;
      y = 0;
    } else {
      vis = j.isVisible[i];
    }
    j.joints[i].x = x;
    j.joints[i].y = y;
    j.isVisible[i] = vis;
  }
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::ApplyCrop(const BoundingBox<Dtype>& crop_bbox,
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
void UnifiedDataTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,
                                              const BoundingBox<Dtype>& crop_bbox,
                                              cv::Mat* image,
                                              vector<MaskData<Dtype> >* mask_data) {
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
      // kps更新
      if (it->kps_included) {
        kps_crop(it->joint, crop_bbox);
        int num_kps = 0;
        for (int i = 0; i < it->joint.joints.size(); ++i) {
          if (it->joint.isVisible[i] <= 1) ++num_kps;
        }
        it->num_kps = num_kps;
        it->kps_included = (num_kps >= param_.kps_min_visible()) ? true : false;
      } else {
        it->num_kps = 0;
        it->kps_included = false;
        for (int k = 0; k < 18; ++k) {
          it->joint.joints[k].x = 0;
          it->joint.joints[k].y = 0;
          it->joint.isVisible[k] = 2;
        }
      }
      // mask
      MaskData<Dtype> mask_element;
      mask_element.bindex = it->bindex;
      mask_element.cid = it->cid;
      mask_element.pid = it->pid;
      mask_element.is_diff = it->is_diff;
      mask_element.iscrowd = it->iscrowd;
      mask_element.has_mask = it->mask_included;
      if (it->mask_included) {
        const string& mask_path = it->mask_path;
        cv::Mat mask_image = cv::imread(mask_path.c_str(),0);
        if (!mask_image.data) {
          LOG(FATAL) << "Error when open image file: " << mask_path;
        }
        mask_element.mask = mask_image(roi);
      } else {
        mask_element.mask = cv::Mat::zeros(crop_h_int, crop_w_int, CV_8UC1);
      }
      mask_data->push_back(mask_element);
      ++num_keep;
      ++it;
    } else {
      it = anno.instances.erase(it);
    }
  }
  anno.num_person = num_keep;
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
                           vector<MaskData<Dtype> >& mask_data, bool* doflip) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  *doflip = (dice <= param_.flip_prob());
  if (*doflip) {
    cv::flip(image, image, 1);
    for (int i = 0; i < mask_data.size(); ++i) {
      if (mask_data[i].has_mask) {
        cv::flip(mask_data[i].mask, mask_data[i].mask, 1);
      }
    }
    for (int i = 0; i < anno.instances.size(); ++i) {
      BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
      Joints& j = anno.instances[i].joint;
      // bbox
      Dtype temp = bbox.x2_;
      bbox.x2_ = 1.0 - bbox.x1_;
      bbox.x1_ = 1.0 - temp;
      // kps
      for (int k = 0; k < j.joints.size(); ++k) {
        j.joints[k].x = 1.0 - j.joints[k].x;
      }
      swapLeftRight(j);
    }
  }
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
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
void UnifiedDataTransformer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes,
                                               vector<KpsData<Dtype> >* kpses) {
  boxes->clear();
  kpses->clear();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    BBoxData<Dtype> bbox_element;
    KpsData<Dtype> kps_element;
    Instance<Dtype>& ins = anno.instances[i];
    // bbox
    bbox_element.bindex = ins.bindex;
    bbox_element.cid = ins.cid;
    bbox_element.pid = ins.pid;
    bbox_element.is_diff = ins.is_diff;
    bbox_element.iscrowd = ins.iscrowd;
    bbox_element.bbox = ins.bbox;
    boxes->push_back(bbox_element);
    // kps
    kps_element.bindex = ins.bindex;
    kps_element.cid = ins.cid;
    kps_element.pid = ins.pid;
    kps_element.is_diff = ins.is_diff;
    kps_element.iscrowd = ins.iscrowd;
    kps_element.has_kps = ins.kps_included;
    kps_element.num_kps = ins.num_kps;
    kps_element.joint = ins.joint;
    kpses->push_back(kps_element);
  }
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image, vector<MaskData<Dtype> >& mask_data) {
  // 修改header
  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // resize 图像
  cv::Mat image_rsz;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;
  // resize mask
  for (int i = 0; i < mask_data.size(); ++i) {
    cv::Mat& mask = mask_data[i].mask;
    cv::Mat mask_rsz;
    resize(mask, mask_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
    mask = mask_rsz;
  }
}

template<typename Dtype>
void UnifiedDataTransformer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes,
                            vector<KpsData<Dtype> >& kpses, vector<MaskData<Dtype> >& masks) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255,0,0,0,255,0,0,0,255,255,0,128,0,128,128,128,128,255};
  CHECK_EQ(boxes.size(), kpses.size());
  CHECK_EQ(boxes.size(), masks.size());
  for (int i = 0; i < boxes.size(); ++i) {
    BBoxData<Dtype>& box = boxes[i];
    KpsData<Dtype>& kps = kpses[i];
    MaskData<Dtype>& mask_e = masks[i];
    // draw boxes
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
    if (box.cid != 0) continue;
    // draw kps
    if (kps.has_kps && kps.num_kps > 0) {
      Joints& j = kps.joint;
      for(int k = 0; k < j.joints.size(); k++) {
        if(j.isVisible[k] <= 1) {
          cv::Point2f p;
          p.x = j.joints[k].x * img_vis.cols;
          p.y = j.joints[k].y * img_vis.rows;
          cv::circle(img_vis, p, 2, CV_RGB(255,255,255), -1);
        }
      }
    }
    // draw mask
    CHECK_EQ(mask_e.bindex, box.bindex);
    CHECK_EQ(mask_e.cid, box.cid);
    CHECK_EQ(mask_e.pid, box.pid);
    if (mask_e.has_mask) {
      int r = color_maps[3*(i % 6)];
      int g = color_maps[3*(i % 6) + 1];
      int b = color_maps[3*(i % 6) + 2];
      cv::Mat& mask = mask_e.mask;
      CHECK_EQ(mask.cols, img_vis.cols);
      CHECK_EQ(mask.rows, img_vis.rows);
      float alpha = 0.4;
      for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
          cv::Vec3b& rgb = img_vis.at<cv::Vec3b>(y, x);
          int mask_val = mask.at<uchar>(y,x);
          if (mask_val > 127) {
            rgb[0] = (1-alpha)*rgb[0] + alpha*b;
            rgb[1] = (1-alpha)*rgb[1] + alpha*g;
            rgb[2] = (1-alpha)*rgb[2] + alpha*r;
          }
        }
      }
    }
  }
  char imagename [256];
  sprintf(imagename, "%s/augment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

// 显示函数
template<typename Dtype>
void UnifiedDataTransformer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image, vector<MaskData<Dtype> >& mask_data) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255,0,0,0,255,0,0,0,255,255,0,128,0,128,128,128,128,255};
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    Joints& j = anno.instances[i].joint;
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
    // 绘制kps
    if (anno.instances[i].kps_included) {
      for(int k = 0; k < j.joints.size(); k++) {
        if(j.isVisible[k] <= 1) {
          cv::Point2f p;
          p.x = j.joints[k].x * img_vis.cols;
          p.y = j.joints[k].y * img_vis.rows;
          cv::circle(img_vis, p, 2, CV_RGB(255,255,255), -1);
        }
      }
    }
    // 绘制mask
    MaskData<Dtype>& mask_element = mask_data[i];
    CHECK_EQ(mask_element.bindex, anno.instances[i].bindex);
    CHECK_EQ(mask_element.cid, anno.instances[i].cid);
    CHECK_EQ(mask_element.pid, anno.instances[i].pid);
    CHECK_EQ(mask_element.is_diff, anno.instances[i].is_diff);
    CHECK_EQ(mask_element.iscrowd, anno.instances[i].iscrowd);
    CHECK_EQ(mask_element.has_mask, anno.instances[i].mask_included);
    if (mask_element.has_mask) {
      cv::Mat& mask = mask_element.mask;
      CHECK_EQ(mask.cols, img_vis.cols);
      CHECK_EQ(mask.rows, img_vis.rows);
      float alpha = 0.4;
      for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
          cv::Vec3b& rgb = img_vis.at<cv::Vec3b>(y, x);
          int mask_val = mask.at<uchar>(y,x);
          if (mask_val > 127) {
            rgb[0] = (1-alpha)*rgb[0] + alpha*b;
            rgb[1] = (1-alpha)*rgb[1] + alpha*g;
            rgb[2] = (1-alpha)*rgb[2] + alpha*r;
          }
        }
      }
    }
  }
  char imagename [256];
  sprintf(imagename, "%s/augment_%06d.jpg", param_.save_dir().c_str(), counter);
  imwrite(imagename, img_vis);
  counter++;
}

template <typename Dtype>
void UnifiedDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int UnifiedDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(UnifiedDataTransformer);

}  // namespace caffe
