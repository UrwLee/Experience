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

#include "caffe/singlepersonpose/bbox_data_transformer_singlepersonpose.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                    vector<BBoxData<Dtype> >* bboxes,
                                    BoundingBox<Dtype>* crop_bbox, bool* doflip,Dtype* transformed_heatmap,Dtype* transformed_vecmap) {

  Normalize(anno);
  TransformMetaPoints(anno);
  randomCrop(anno, image, crop_bbox);
  
    // cv::imwrite("/home/xjx/tmp/1/crop.jpg",*image);

  randomDistortion(image, anno);
    // cv::imwrite("/home/xjx/tmp/1/dis.jpg",*image);

  randomFlip(anno, *image, doflip);
 
  // cv::imwrite("/home/xjx/tmp/1/flip.jpg",*image);

  // fixedResize(anno, *image,*maskmiss);
    // cv::imwrite("/home/xjx/tmp/1/resize.jpg",*image);


  // randomly crpp
  // randomCrop(anno, image, crop_bbox,maskmiss);
  // CHECK(image->data);
  // // perform distortion
  // randomDistortion(image, anno);
  // // // flip
  // randomFlip(anno, *image, doflip,*maskmiss);
  // // resized
  // LOG(INFO)<<"123123~~~~";
  augmentation_rotate(anno,*image);
  fixedResize(anno, *image);

  
  // test(anno, *image);

  // // save to bboxes
  copy_label(anno, bboxes);
  get_mask(anno, *image,transformed_heatmap,transformed_vecmap);

  // // visualize
  if (param_.visualize()) {

    visualize(*image, *bboxes);
  }
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
}

//#############################################################

// randomly crop/scale
template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox) {
  // image
  
  *image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  // LOG(INFO)<<anno.img_path.c_str();

  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  getCropBBox(anno, crop_bbox);
  TransCrop(anno,*crop_bbox,image);
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image) {
  // image
  //*image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  vector<BatchSampler> samplers;
  for (int s = 0; s < param_.batch_sampler_size(); ++s) {
    samplers.push_back(param_.batch_sampler(s));//4个batch_sampler
  }
  int idx = caffe_rng_rand() % samplers.size();
  BoundingBox<Dtype> crop_bbox;
  SampleBBox(samplers[idx].sampler(), &crop_bbox);
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
  // cv::Mat maskmiss_back=maskmiss->clone();
  *image = image_back(roi);
  // *maskmiss=maskmiss_back(roi);
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
    
    if (phase_ == TRAIN) {
      float scale;
      caffe_rng_uniform(1, param_.min_scale(), param_.max_scale(), &scale);
      float drift ;
      caffe_rng_uniform(1, -param_.drift_scalar(),param_.drift_scalar(), &drift);
      BoundingBox<Dtype>& bbox = anno.instances[0].bbox;
      float box_w = bbox.x2_ - bbox.x1_;
      float box_h = bbox.y2_ - bbox.y1_;
      crop_bbox->x1_ = std::max(0.0,bbox.x1_ - box_w*scale/2.0 + drift);
      crop_bbox->x2_ = std::min(1.0,bbox.x2_ + box_w*scale/2.0 + drift);
      crop_bbox->y1_ = std::max(0.0,bbox.y1_ - box_h*scale/2.0 + drift);
      crop_bbox->y2_ = std::min(1.0,bbox.y2_ + box_h*scale/2.0 + drift);
      if (crop_bbox->x2_<=crop_bbox->x1_ || crop_bbox->y2_<=crop_bbox->y1_){
        crop_bbox->x1_ = 0.0;
        crop_bbox->x2_ = 1.0;
        crop_bbox->y1_ = 0.0;
        crop_bbox->y2_ = 1.0;
      }
    }else{
      crop_bbox->x1_ = 0.0;
      crop_bbox->x2_ = 1.0;
      crop_bbox->y1_ = 0.0;
      crop_bbox->y2_ = 1.0;
    }
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,
                                            const BoundingBox<Dtype>& crop_bbox,
                                            cv::Mat* image) {
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
  // cv::Mat maskmiss_back=maskmiss->clone();
  *image = image_back(roi);
  
  // *maskmiss=maskmiss_back(roi);
  // scan all instances, delete the crop boxes
  int num_keep = 0;
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    BoundingBox<Dtype> proj_bbox;
    float coverage = gt_bbox.project_bbox(crop_bbox, &proj_bbox);
    it->bbox = proj_bbox;
    if(it->kps_included){
      Joints old_kps=it->joint;
      // it->joint.joints[1].x=old_kps.joints[1].x+10;
      // LOG(INFO)<<old_kps.joints[1].x<<"$$$"<<it->joint.joints[1].x;
      for(int i=0;i<18;++i){
        if(old_kps.isVisible[i]==1){
        it->joint.joints[i].x=old_kps.joints[i].x-w_off_int;
        it->joint.joints[i].y=old_kps.joints[i].y-h_off_int;
        if(it->joint.joints[i].x<0||it->joint.joints[i].x>crop_w_int||it->joint.joints[i].y<0||it->joint.joints[i].y>crop_h_int){
           it->joint.joints[i].x=0;
           it->joint.joints[i].y=0;
           it->joint.isVisible[i]=2;
        }
        }
        // LOG(INFO)<<expand_bbox.x1_<<"$$"<<old_kps.joints[i].x<<"$$"<<it->joint.joints[i].x;
      }
    }
    ++num_keep;
    ++it;
  }
  anno.num_person = num_keep;
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
                            bool* doflip) {
  //生成０－１随机数
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  *doflip = (dice <= param_.flip_prob());
  if (*doflip) {
    cv::flip(image, image, 1);//水平翻转
    // cv::flip(maskmiss, maskmiss, 1);//水平翻转
    int num_keypoints = 18;
    for (int i = 0; i < anno.instances.size(); ++i) {
      BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
      // bbox
      Dtype temp = bbox.x2_;
      bbox.x2_ = 1.0 - bbox.x1_;
      bbox.x1_ = 1.0 - temp;
      if (anno.instances[i].kps_included){
      for(int j=0;j<num_keypoints;++j){
          if(anno.instances[i].joint.isVisible[j]==1){
            anno.instances[i].joint.joints[j].x=anno.img_width-anno.instances[i].joint.joints[j].x;
          }
        }
          // int right[8] = {3,5,7,9,11,13,15,17};
          // int left[8] =  {2,4,6,8,10,12,14,16};
          // for(int j = 0; j< 8; j++) {
          //   int ri = right[j] - 1;
          //   int li = left[j] - 1;
          //   Point2f temp = anno.instances[i].joint.joints[ri];
          //   anno.instances[i].joint.joints[ri] = anno.instances[i].joint.joints[li];
          //   anno.instances[i].joint.joints[li] = temp;
          //   int temp_v = anno.instances[i].joint.isVisible[ri];
          //   anno.instances[i].joint.isVisible[ri] = anno.instances[i].joint.isVisible[li];
          //   anno.instances[i].joint.isVisible[li] = temp_v;
          // }
      }
    }
  }
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
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
void BBoxDataSinglePersonPoseTransformer<Dtype>::TransformMetaPoints(AnnoData<Dtype>& anno){
 
  for (int i = 0; i < anno.instances.size(); ++i) {
    if (anno.instances[i].kps_included){
      TransformJoints(anno.instances[i].joint);
    }
  }
}
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::TransformJoints(Joints& j) {
  Joints jo = j;
  // COCO
 
  int COCO_to_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
  int COCO_to_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
  // 转换为18个, 包含Neck
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
  j = jo;
}

template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes) {
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
void BBoxDataSinglePersonPoseTransformer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
  // modify header
    int stride=param_.stride();
  float scale_x=float(param_.resized_width())/anno.img_width;
  float scale_y=float(param_.resized_height())/anno.img_height;

  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // resize image
  cv::Mat image_rsz;
  // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
  // CHECK(image.data);
  // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;

//   cv::Mat maskmiss_rsz;
//   resize(maskmiss, maskmiss_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
// resize(maskmiss_rsz, maskmiss_rsz, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC); 
//LOG(INFO)<<maskmiss_rsz.cols<<"*"<<maskmiss_rsz.rows<<"!!!!!!!!!!!!!"; 
// maskmiss = maskmiss_rsz;
  int num_keypoints = 18;
  for (int i = 0; i < anno.instances.size(); ++i) {
    if (anno.instances[i].kps_included){
      for(int j=0;j<num_keypoints;++j){
        // cv::circle(image, anno.instances[i].joint.joints[13], 5, cv::Scalar(0, 0, 255));
        // cv::imwrite("/home/xjx/tmp/1/14.jpg",image);
          if(anno.instances[i].joint.isVisible[j]==1){
            anno.instances[i].joint.joints[j].x=int(anno.instances[i].joint.joints[j].x*scale_x);
            anno.instances[i].joint.joints[j].y=int(anno.instances[i].joint.joints[j].y*scale_y);
           //  int font_face = cv::FONT_HERSHEY_SIMPLEX; 

           //  std::ostringstream s;
           //  LOG(INFO)<<j<<"!!!!!";
           //  s <<j+1;
           // cv::putText(image,s.str(),cvPoint(anno.instances[i].joint.joints[j].x,anno.instances[i].joint.joints[j].y), font_face, 0.5, cv::Scalar(0, 0, 255), 2, 2, 0); 
           // cv::imwrite("/home/xjx/tmp/1/16.jpg",image);
          }
        }
        // Joints old_kps=anno.instances[i].joint;
        // int change[18]={1,18, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
        // for(int t=0;t<18;++t){
        //   anno.instances[i].joint.joints[t].x=old_kps.joints[change[t]-1].x;
        //   anno.instances[i].joint.joints[t].y=old_kps.joints[change[t]-1].y;
        //   anno.instances[i].joint.isVisible[t]=old_kps.isVisible[change[t]-1];


        // }


// for (int i = 0; i < anno.instances.size(); ++i) {
//     if (anno.instances[i].kps_included){
//       for(int j=0;j<18;++j){
//         int font_face = cv::FONT_HERSHEY_SIMPLEX; 
//             std::ostringstream s;
//             LOG(INFO)<<j<<"!!!!!";
//             s <<j+1;
//            cv::putText(image,s.str(),cvPoint(anno.instances[i].joint.joints[j].x,anno.instances[i].joint.joints[j].y), font_face, 0.5, cv::Scalar(0, 0, 255), 2, 2, 0); 
//            cv::imwrite("/home/xjx/tmp/1/16.jpg",image);
          
//       }
//   }
// }
      }
    }
    

}

template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::RotatePoint(Point2f& p, Mat& R){
  Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}
// no masks
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::augmentation_rotate(AnnoData<Dtype>& anno, cv::Mat& image) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  float degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  Point2f center(image.cols/2.0, image.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, image.size(), degree).boundingRect();
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  Mat img_aug;
  warpAffine(image, img_aug, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  int num_keypoints = 18;
  image = img_aug;
  anno.img_width = image.cols;
  anno.img_height = image.rows;
  for (int i = 0; i < anno.instances.size(); ++i) {
    if (anno.instances[i].kps_included){
      for(int j=0;j<num_keypoints;++j){
        if(anno.instances[i].joint.isVisible[j]==1){
          RotatePoint(anno.instances[i].joint.joints[j], R);
          if(anno.instances[i].joint.joints[j].x<0 | anno.instances[i].joint.joints[j].y<0 | anno.instances[i].joint.joints[j].x>=image.cols|anno.instances[i].joint.joints[j].y>=image.rows){
            anno.instances[i].joint.isVisible[j]==2;
          }
        }
      }
    }
  }
}
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes) {
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
void BBoxDataSinglePersonPoseTransformer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image) {
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
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::test(AnnoData<Dtype>& anno, cv::Mat& image) {
  // static int cnt = 0;
  // cnt ++;
  int i=fabs(caffe_rng_rand()%1000);
  std::stringstream ss;
  std::string str;
  ss<<i;
  ss>>str;

  typename vector<Instance<Dtype> >::iterator it;
  int width=anno.img_width;
  int height=anno.img_height;
  int num_keypoints = 18;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype> proj_bbox = it->bbox;
   // cv::rectangle(image,cvPoint(int(proj_bbox.x1_*width),int(proj_bbox.y1_*height)),cvPoint(int(proj_bbox.x2_*width),int(proj_bbox.y2_*height)),Scalar(255,0,0),5,1,0);
   
   cv::Point p;
   if(it->kps_included){
    for(int j=0;j<num_keypoints;j=j+1){
      //LOG(INFO)<<j;
      p.x=it->joint.joints[j].x;
      p.y=it->joint.joints[j].y;
      int vis=it->joint.isVisible[j];
      // LOG(INFO)<<p.x<<"@"<<p.y<<"@@"<<vis;
      if(vis==1){
        cv::circle(image, p, 2, cv::Scalar(0, 0, 255));
                }
          }
    }
    ++it;
   }
     LOG(INFO)<<"123123~~~~";

   cv::imwrite("/home/ethan/tmp/"+str+"a.jpg",image);

   // cv::imwrite("/home/zhangming/xjx/tmp/"+str+"amask_miss.jpg",maskmiss);
}

template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::get_mask(AnnoData<Dtype>& anno, cv::Mat& image,Dtype* transformed_heatmap,Dtype* transformed_vecmap){
  int rezX = image.cols;
  int rezY = image.rows;
  int stride=param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

// LOG(INFO)<<1111;
  // resize(maskmiss, maskmiss, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);

  // 生成label-map
  // LOG(INFO)<<2222;

  generateLabelMap(transformed_vecmap,transformed_heatmap, image, anno);
}
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::generateLabelMap(Dtype* transformed_vecmap,Dtype* transformed_heatmap,
                                                  cv::Mat& img_aug, AnnoData<Dtype>& anno) {

  int stride = param_.stride();
  int grid_x = img_aug.cols / stride;
  int grid_y = img_aug.rows / stride;
  int channelOffset = grid_y * grid_x;
  // int mode = param_.mode();
  // init for maps
  // LOG(INFO)<<3333;
  int num_keypoints = 18;
  for (int g_y = 0; g_y < grid_y; g_y++) {
    for (int g_x = 0; g_x < grid_x; g_x++) {
      for (int i = 0; i < num_keypoints; i++) {
        transformed_heatmap[i * channelOffset + g_y * grid_x + g_x] = 0;
      }
    }
  }
  // Maps
  // heatmap
  

  typename vector<Instance<Dtype> >::iterator it;
  for (int i = 0; i < num_keypoints; i++) {
    for (it = anno.instances.begin(); it != anno.instances.end();) {
        if (it->kps_included){
        //for (int i = 0; i < 18; i++) {
          Point2f center=it->joint.joints[i];
          // LOG(INFO)<<center.x<<" "<<center.y;
          if(it->joint.isVisible[i]<=1) {
              putGaussianMaps(transformed_heatmap + i * channelOffset, center, param_.stride(),
                            grid_x, grid_y, param_.sigma());
            }
        }
        ++it;
    }
  }

  // int mid_1[20] = {0, 1,  2, 3, 0, 5, 6, 7,  0, 9, 10, 11, 0,  13,  14, 15, 0,17,18,19};
  // int mid_2[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20};
  // for 21 keypoints
  // int mid_1[20] = {1,2,3,4,1,6,7,8,1,10,11,12,1,14,15,16,1,18,19,20};
  // int mid_2[20] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
  // for 20 keypoints
  int mid_1[17] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 2, 6, 7, 2, 1,  1,  15, 16};
  int mid_2[17] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8, 1, 15, 16, 17, 18};
  int thre = 1;
  for(int i = 0; i < 17; i++) {
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    if (it->kps_included){
    //for(int i = 0; i < 17; i++) {
      Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = it->joint;
      if(jo.isVisible[mid_1[i]-1] <= 1 && jo.isVisible[mid_2[i]-1] <= 1) {
        putVecMaps(transformed_vecmap + 2 * i * channelOffset, transformed_vecmap + (2 * i + 1) * channelOffset,
                  count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre);
      }
    }
    // LOG(INFO)<<6666;
    ++it;
 }
}
  if(0) {
    Mat label_map;
    // 显示每个通道
    for(int i = 0; i < 17; i++){
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      for (int g_y = 0; g_y < grid_y; g_y++) {
        for (int g_x = 0; g_x < grid_x; g_x++) {
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_vecmap[2*i * channelOffset + g_y * grid_x + g_x] * 255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
      char imagename [256];
      sprintf(imagename, "%s/%02d.jpg", "/home/xjx/tmp",i);
      imwrite(imagename, label_map);
    }
  }
}

// 生成GaussMap
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::putGaussianMaps(Dtype* map, Point2f center, int stride, int grid_x, int grid_y, float sigma) {
  float start = stride / 2.0 - 0.5;
  for (int g_y = 0; g_y < grid_y; g_y++) {
    for (int g_x = 0; g_x < grid_x; g_x++) {
      // LOG(INFO)<<g_x<<" "<<g_y;
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      // 0.01
      if(exponent > 4.6052) {
        continue;
      }
      map[g_y * grid_x + g_x] += exp(-exponent);
      if(map[g_y * grid_x + g_x] > 1) {
         map[g_y * grid_x + g_x] = 1;
      }
    }
  }
}
template<typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::putVecMaps(Dtype* vecX, Dtype* vecY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre) {
  centerB = centerB / stride;
  centerA = centerA / stride;
  Point2f bc = centerB - centerA;
  // 选取线段所在的区域
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x) - thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x) + thre)), grid_x);
  int min_y = std::max( int(round(std::min(centerA.y, centerB.y) - thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y) + thre)), grid_y);
  // 计算法线方向
  float norm_bc = sqrt(bc.x * bc.x + bc.y * bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;
  // 遍历该区域
  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      // 垂直方向
      float dist = std::abs(ba.x * bc.y - ba.y * bc.x);
      // 如果小于阈值,则表明是线段区域
      // 注意count矩阵, 求取平均值
      if(dist <= thre) {
        int cnt = count.at<uchar>(g_y, g_x);
        if (cnt == 0){
          vecX[g_y * grid_x + g_x] = bc.x;
          vecY[g_y * grid_x + g_x] = bc.y;
        } else {
          vecX[g_y * grid_x + g_x] = (vecX[g_y * grid_x + g_x] * cnt + bc.x) / (cnt + 1);
          vecY[g_y * grid_x + g_x] = (vecY[g_y * grid_x + g_x] * cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
        //LOG(INFO)<<count.at<uchar>(g_y, g_x)<<'HAHAHAHAHA';
      }
    }
  }
}


template <typename Dtype>
void BBoxDataSinglePersonPoseTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int BBoxDataSinglePersonPoseTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(BBoxDataSinglePersonPoseTransformer);

}  // namespace caffe
