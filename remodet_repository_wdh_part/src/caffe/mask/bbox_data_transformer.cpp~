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

#include "caffe/mask/bbox_data_transformer.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"
// xml 读取
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
using namespace boost::property_tree;

#define DEBUG_WDH false

namespace caffe {

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                    vector<BBoxData<Dtype> >* bboxes,
                                    BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox, cv::Mat& bg_img) {
  /*
    in : anno 
    func: 对anno 中每个gt 使用原始图片尺寸 进行归一化 , 
  */  
  Normalize(anno); 
   // perform expand
  randomExpand(anno, image);
  // 旋转增广
  // ninetyAngle(anno, *image);
    // 抠手 gt 
  // cutGtChangeBg(anno , *image, bg_img);
  // 随机背光
  RandomBacklight(*image,  anno, int(3));
  // sun flare 
  //add_sun_flare(*image, -1, cv::Point(-1, -1), (image->rows/3), 10, 0.f, 0.5f); 

  randomDistortion(image, anno);
  // randomly crpp

  rotate90(*image,  anno);
  randomCrop(anno, image, crop_bbox,image_bbox);

  CHECK(image->data);

  // // flip
  randomFlip(anno, *image, doflip,image_bbox);
  randomXFlip(*image, anno, image_bbox);
  // // resized
  fixedResize(anno, *image);
  // // save to bboxes
  copy_label(anno, bboxes);
  // // visualize
  if (param_.visualize()) {
    visualize(*image, *bboxes);
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip) {
   // perform expand
  randomExpand(anno, image);
  // randomly crop
  randomCrop(anno, image);
  CHECK(image->data);
  // perform distortion
  randomDistortion(image, anno);
  // // flip
  randomFlip(anno, *image, doflip);
  // // resized
  fixedResize(anno, *image);

}// //###########################################################

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomExpand(AnnoData<Dtype>& anno, cv::Mat* image) {
    // 读入图片

  *image = cv::imread(anno.img_path.c_str()); // 
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  if (!param_.has_expand_param()){ // func: 不进行expand 跳出
    return;
  }
  // 读入图片
  if (!image->data) { 
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  //LOG(FATAL) << "Error when open image file: " << anno.img_path;
//
  // CHECK_EQ(image->cols, anno.img_width);
  // CHECK_EQ(image->rows, anno.img_height);

  BoundingBox<Dtype> expand_bbox;
  //读取expand参数
  const float max_expand_ratio = param_.expand_param().max_expand_ratio(); 
  const float expand_prob = param_.expand_param().prob();
  float expand_ratio;
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) { 
    return;
  }
  if (fabs(max_expand_ratio - 1.) < 1e-2) { // func: 最大expand 趋近于1 不进行 expand
    return;
  } 
  //随机选一个expand_ratio
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);

  //expand_img=ExpandImage(image,param_.expand_param(),expand_bbox);
  //随机新框大小
  const int img_width = image->cols;
  const int img_height = image->rows;
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  // modify header
  anno.img_width = width;
  anno.img_height = height;
  //LOG(INFO)<<"expand_ratio"<<expand_ratio;
  //LOG(INFO)<<"anno.img_width"<<width;

  //随机新框位置
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);

  h_off = floor(h_off); // func: 不大于自变量的最大整数, 2.6 -> 2, 
  w_off = floor(w_off);
  //记录新框相对于旧框的位置
  expand_bbox.x1_ = (-w_off / img_width);
  expand_bbox.y1_ = (-h_off / img_height);
  expand_bbox.x2_ = ((width - w_off) / img_width);
  expand_bbox.y2_ = ((height - h_off) / img_height);

  //图像转换
  cv::Mat expand_img;
  expand_img.create(height, width, image->type()); // func: 创建扩展后的 背景图
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
  expand_img.setTo(cv::Scalar(104,117,123,0.0)); // func: 填充均值
  // expand_img.setTo(cv::Scalar(mean_values[0],mean_values[1],mean_values[2],0.0));

  // expand_img.setTo(cv::Scalar(0));

  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  image->copyTo(expand_img(bbox_roi));
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
      else {
        it = anno.instances.erase(it);
      }
    // ++it;
  }
  //for test
  bool test=0;  if(test){
    int i=fabs(caffe_rng_rand()%1000);
    std::stringstream ss;
    std::string str;
    ss<<i;
    ss>>str;
    for (it = anno.instances.begin(); it != anno.instances.end();) {
      BoundingBox<Dtype> proj_bbox = it->bbox;
     cv::rectangle(*image,cvPoint(int(proj_bbox.x1_*width),int(proj_bbox.y1_*height)),cvPoint(int(proj_bbox.x2_*width),int(proj_bbox.y2_*height)),Scalar(255,0,0),1,1,0);
     ++it;
    }
    cv::imwrite("/home/xjx/xjx/AIC/"+str+".jpg",*image);
  }
}
//#############################################################


// randomly crop/scale
template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox) {
  // image
  // *image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  getCropBBox(anno, crop_bbox);  // TODO: use Part
  // int i=fabs(caffe_rng_rand()%1000);
  // std::stringstream ss;
  // std::string str;
  // ss<<i;
  // ss>>str;
  // int width = image->cols;
  // int height = image->rows;
  // typename vector<Instance<Dtype> >::iterator it;
  // for (it = anno.instances.begin(); it != anno.instances.end();) {
  //   BoundingBox<Dtype> proj_bbox = it->bbox;
  //  cv::rectangle(*image,cvPoint(int(proj_bbox.x1_*width),int(proj_bbox.y1_*height)),cvPoint(int(proj_bbox.x2_*width),int(proj_bbox.y2_*height)),Scalar(255,0,0),1,1,0);
  //  ++it;
  // }  
  // cv::imwrite("/home/zhangming/Datasets/RemoCoco/vis_aug/"+str+"after_getCropBBox.jpg",*image);
  TransCrop(anno,*crop_bbox,image,image_bbox);
  // width = image->cols;
  // height = image->rows;
  // for (it = anno.instances.begin(); it != anno.instances.end();) {
  //   BoundingBox<Dtype> proj_bbox = it->bbox;
  //  cv::rectangle(*image,cvPoint(int(proj_bbox.x1_*width),int(proj_bbox.y1_*height)),cvPoint(int(proj_bbox.x2_*width),int(proj_bbox.y2_*height)),Scalar(255,0,0),1,1,0);
  //  ++it;
  // }
  // cv::imwrite("/home/zhangming/Datasets/RemoCoco/vis_aug/"+str+"after_TransCrop.jpg",*image);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image) {
  // image
  // *image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  vector<BatchSampler> samplers;
  for (int s = 0; s < param_.batch_sampler_size(); ++s) {
    samplers.push_back(param_.batch_sampler(s));
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
  *image = image_back(roi);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
    // get a random value [0-1]
    // float prob = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // get max width & height
    float h_max, w_max;
    if(param_.has_sample_sixteennine()||param_.has_sample_ninesixteen()){
      if(param_.sample_sixteennine()){ // func: 找到最短边, 作为16:9 的处理标准
        h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
        w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
      }else if(param_.sample_ninesixteen()){
        h_max = std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
        w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
      }     
    } else{
      h_max = 1.0;
      w_max = 1.0;
    }
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

      // if (param_.has_sample_sixteennine()){
      //   if(param_.sample_sixteennine()){
      //     GenerateBatchSamples16_9(anno, samplers, &sample_bboxes, h_max, w_max);
      //   } else {
      //     GenerateBatchSamples(anno, samplers, &sample_bboxes);
      //   }
      // } else {
      //     GenerateBatchSamples(anno, samplers, &sample_bboxes);
      // }
      if (param_.sample_random()){ // default is false, if true,it means to crop accord body or part based on probability
          GenerateBatchSamplesRandom16_9(anno, samplers, &sample_bboxes, h_max, w_max);
      }else{
        if (param_.for_body()) {  // default is true, this datalayer use for body-crop
          GenerateBatchSamples16_9(anno, samplers, &sample_bboxes, h_max, w_max);
        } else {                  // this datalayer use for part-crop
        	if (param_.crop_around_gt()){ // func: 默认为false, 围绕gt 进行采样
         	    GenerateBatchSamples4PartsAroundGT16_9(anno, samplers, &sample_bboxes, h_max, w_max); 
	          }
               else {
            GenerateBatchSamples4Parts16_9(anno, samplers, &sample_bboxes, h_max, w_max);
          }
        }
      }

      if (sample_bboxes.size() > 0) { // 
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
void BBoxDataTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,
                                            const BoundingBox<Dtype>& crop_bbox,
                                            cv::Mat* image, BoundingBox<Dtype>* image_bbox) {
  const int img_width = image->cols;
  const int img_height = image->rows;
  // image crop:
  /**
   * We need to copy [] of image -> [] of bg
   */
  int wb = std::ceil(crop_bbox.get_width() * img_width);
  int hb = std::ceil(crop_bbox.get_height() * img_height);
  // LOG(INFO)<<crop_bbox.get_width()<<" "<<crop_bbox.get_height();
  cv::Mat bg(hb, wb, CV_8UC3, cv::Scalar(128, 128, 128));
  anno.img_width = wb;
  anno.img_height = hb;
  // (1) Intersection
  int pxmin = (int)(std::max(crop_bbox.x1_, Dtype(0)) * img_width);
  int pymin = (int)(std::max(crop_bbox.y1_, Dtype(0)) * img_height);
  int pxmax = std::floor(std::min(crop_bbox.x2_, Dtype(1)) * img_width) - 1;
  int pymax = std::floor(std::min(crop_bbox.y2_, Dtype(1)) * img_height) - 1;
  // LOG(INFO)<<"pxmin "<<pxmin<<" pxmax "<<pxmax<<" pymin "<<pymin<<" pymax "<<pymax<<" wb "<<wb<<" hb "<<hb ;
  // LOG(INFO)<<"img_width "<<img_width<<" img_height "<<img_height;
  // (2) patch of image
  int pwidth  = pxmax - pxmin;
  int pheight = pymax - pymin;
  cv::Rect orig_patch(pxmin, pymin, pwidth, pheight);
  // (3) patch of bg
  int xmin_bg = std::floor(crop_bbox.x1_ * img_width);
  int ymin_bg = std::floor(crop_bbox.y1_ * img_height);
  // LOG(INFO)<<"pxmin - xmin_bg "<<pxmin - xmin_bg<<" pymin - ymin_bg "<<pymin - ymin_bg<<" pwidth "<<pwidth<<" pheight "<<pheight;
  cv::Rect bg_patch(pxmin - xmin_bg, pymin - ymin_bg, pwidth, pheight);
  image_bbox->x1_ = (Dtype)(pxmin - xmin_bg)/(Dtype)wb;
  image_bbox->x2_ = (Dtype)(pxmin - xmin_bg + pwidth)/(Dtype)wb;
  image_bbox->y1_ = (Dtype)(pymin - ymin_bg)/(Dtype)hb;
  image_bbox->y2_ = (Dtype)(pymin - ymin_bg + pheight)/(Dtype)hb;
  // LOG(INFO)<<bg_patch.x<<" "<<bg_patch.y<<" "<<bg_patch.width<<" "<<bg_patch.height;
  // LOG(INFO)<<bg.cols<<" "<<bg.rows;
  cv::Mat area = bg(bg_patch);
  // LOG(INFO)<<"bbb";
  // (4) copy
  (*image)(orig_patch).copyTo(area);
  *image = bg;
  //old_code-start
  //  anno.img_width = (int)(img_width * (crop_bbox.get_width()));
  // anno.img_height = (int)(img_height * (crop_bbox.get_height()));

  // int w_off_int = (int)(crop_bbox.x1_ * img_width);
  // int h_off_int = (int)(crop_bbox.y1_ * img_height);
  // int crop_w_int = (int)(img_width * (crop_bbox.get_width()));
  // int crop_h_int = (int)(img_height * (crop_bbox.get_height()));
  // cv::Rect roi(w_off_int, h_off_int, crop_w_int, crop_h_int);
  // cv::Mat image_back = image->clone();
  // *image = image_back(roi);
  //old_code-end
  // scan all instances, delete the crop boxes
  int num_keep = 0;
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    BoundingBox<Dtype> proj_bbox;
    // keep instance/ emitting true ground truth
    Dtype emit_coverage_thre = 0;
    if(phase_ == TRAIN){
      if (param_.has_emit_coverage_thre()){
        emit_coverage_thre = param_.emit_coverage_thre();
      } else {
        Dtype area_gt = gt_bbox.compute_area();
        // LOG(INFO) << "area_gt " << area_gt;
        for (int s = 0; s < param_.emit_area_check_size(); ++s) {
          if (area_gt< param_.emit_area_check(s)){
              emit_coverage_thre = param_.emit_coverage_thre_multiple(s);
              break;
            }
        }
      }
    }else{
      emit_coverage_thre = param_.emit_coverage_thre();
    }

    if (gt_bbox.project_bbox(crop_bbox, &proj_bbox) >= emit_coverage_thre) {
        // box update
        // LOG(INFO) << "project_bbox_area " << gt_bbox.project_bbox(crop_bbox, &proj_bbox)<<", emit_coverage_thre "<<emit_coverage_thre;
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
void BBoxDataTransformer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());

}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::ninetyAngle(AnnoData<Dtype>& anno, cv::Mat& image) {
  // 1. 旋转图片
  float angle = 90.0;
  int scale = 1;
  int img_w = image.cols;
  int img_h = image.rows;
  cv::Mat_<double> M = cv::getRotationMatrix2D(Point(img_w*0.5, img_h*0.5), -angle, scale);
  double cos = M(0,0);
  double sin = M(0,1);

  LOG(INFO) << "cos, sin" <<cos <<"*"<< sin;
  int r_w = (img_h * sin) + (img_w * cos);
  int r_h = (img_h * cos) + (img_w * sin);
  LOG(INFO) << " w*H" << r_w <<"*"<<r_h;

  cv::warpAffine(image, image, M, Size(r_w, r_h));
  // //  2. 旋转box
  // for (int i = 0; i < anno.instances.size(); ++i) {
  //   BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
  // //   // bbox
  //   bbox.x1_ *= anno.img_width;
  //   bbox.x2_ *= anno.img_width;
  //   bbox.y1_ *= anno.img_height;
  //   bbox.y2_ *= anno.img_height;
  //   LOG(INFO) << bbox.x1_ <<","<< bbox.y1_ <<","<<bbox.x2_ <<","<<bbox.y2_ ;
  //   int x1_n = cos * bbox.x1_ - sin * bbox.y1_;
  //   int y1_n = sin * bbox.x1_ + cos * bbox.y1_;
  //   int x2_n = x1_n + bbox.get_width();
  //   int y2_n = y1_n + bbox.get_height();
  //   LOG(INFO) << x1_n<<","<< y1_n<<","<< x2_n<<","<< y2_n;
  //   bbox.x1_ = (int)(x1_n*1.0/r_w*1.0);
  //   bbox.x2_ = (int)(x2_n*1.0/r_w*1.0);
  //   bbox.y1_ = (int)(y1_n*1.0/r_h*1.0);
  //   bbox.y2_ = (int)(y2_n*1.0/r_h*1.0);
  //   LOG(INFO) << bbox.x1_ <<","<< bbox.y1_ <<","<<bbox.x2_ <<","<<bbox.y2_ ;
  // }  

  anno.img_width = r_w;
  anno.img_height = r_h;

}



template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
                            bool* doflip,BoundingBox<Dtype>* image_bbox) {
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
    Dtype tmp1 = image_bbox->x2_;
    image_bbox->x2_ = 1.0 - image_bbox->x1_;
    image_bbox->x1_ = 1.0 - tmp1;
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
                            bool* doflip) {
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
void BBoxDataTransformer<Dtype>::randomXFlip(cv::Mat &image, AnnoData<Dtype>& anno, BoundingBox<Dtype>* image_bbox) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  if ((dice <= param_.xflip_prob())) {
    cv::flip(image, image, 0);
    for (int i = 0; i < anno.instances.size(); ++i) {
      BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
      // bbox
      Dtype temp = bbox.y2_;
      bbox.y2_ = 1.0 - bbox.y1_;
      bbox.y1_ = 1.0 - temp;
    }
    Dtype tmp1 = image_bbox->y2_;
    image_bbox->y2_ = 1.0 - image_bbox->y1_;
    image_bbox->y1_ = 1.0 - tmp1;
  }
}
 

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
 //
  const int image_width = anno.img_width; // 原始图片尺寸
  const int image_height = anno.img_height;
  //
  for (int i = 0; i < anno.instances.size(); ++i) { // func: 对anno 中每个gt进行归一化 
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    bbox.x1_ /= (Dtype)image_width;
    bbox.x2_ /= (Dtype)image_width;
    bbox.y1_ /= (Dtype)image_height;
    bbox.y2_ /= (Dtype)image_height;
  }
}
template <typename Dtype>
void BBoxDataTransformer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes) {
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
    bbox_element.ignore_gt = ins.ignore_gt;
    boxes->push_back(bbox_element);
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
  // modify header
  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // resize image
  cv::Mat image_rsz;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;
}

// =========================
/*
  初始化函数 , 设置txt 和路径获取bg
*/
// template <typename Dtype>
// void BBoxDataTransformer<Dtype>::setupBg(){
//     bg_lines_id_ = 0;
//     string list_xml_bg = "/home/xjx/REMO_HandPose_20180912_DetHand_DataGroup/face_hand_test.txt";
//     string root_xml_bg = "/home/xjx/REMO_HandPose_20180912_DetHand_DataGroup";
//     std::ifstream infile(list_xml_bg.c_str()); 
//     std::string xmlname;
//     LOG(INFO) << " xxx "<<bg_lines_.size() ;
//     while(infile >> xmlname){
//         bg_lines_.push_back(make_pair(root_xml_bg, xmlname));
//     }
//     LOG(INFO) << bg_lines_.size();
// }

/*
  从 txt 读取图片
*/
// template <typename Dtype>
// void BBoxDataTransformer<Dtype>::readBgImg(cv::Mat& bg_img){


//   string xml_root = bg_lines_[bg_lines_id_].first;
//   string xml_path = xml_root + '/' + bg_lines_[bg_lines_id_].second;
//     // cout << "  | num : "<< bg_lines_id_ << ", " << xml_path << endl;
//   AnnoData<Dtype> anno;
//   ReadAnnoDataFromXml(0, xml_path, xml_root, &anno);

//   bg_img = cv::imread(anno.img_path.c_str());
//   if (!bg_img.data)  LOG(FATAL) << "Error when open image file: " << anno.img_path;
//   bg_lines_id_ ++;
//   if (bg_lines_id_ >= bg_lines_.size()){
//     bg_lines_id_ = 0;
//     std::random_shuffle(bg_lines_.begin(), bg_lines_.end());
//   }
// }

/*
  对于尺度小于512 288 的图像进行填充
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::borderImage(cv::Mat& image){
  /*
  BORDER_REPLICATE 重复： 就是对边界像素进行复制
  BORDER_REFLECT 反射：对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb 反射
  BORDER_REFLECT_101 反射101： 例子：gfedcb|abcdefgh|gfedcba
  BORDER_WRAP 外包装：cdefgh|abcdefgh|abcdefg
  BORDER_CONSTANT 常量复制：例子：iiiiii|abcdefgh|iiiiiii */

  cv::Size shape_img = image.size();
  // 图片尺寸比512 ,288 小进行填充
  if (shape_img.width < 512 || shape_img.height < 288){
    int add_x = std::max(0, 512 - shape_img.width);
    int add_y = std::max(0, 288 - shape_img.height);
    if(DEBUG_WDH) LOG(INFO) << " 边缘填充: " << add_y << ", " << add_x ;
    cv::copyMakeBorder(image, image, 0, add_x, 0,  add_y, BORDER_WRAP);
  }
}

/*
  获得anno 的实际坐标
  原始图像的 box是归一化的, 进行恢复
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::unNormalize(AnnoData<Dtype>& anno) {
 //
  const int image_width  = anno.img_width; // 原始图片尺寸
  const int image_height = anno.img_height;
  //
 
  if(DEBUG_WDH) LOG(INFO) << "anno num "<< anno.instances.size();
  for (int i = 0; i < anno.instances.size(); ++i) { // func: 对anno 中每个gt进行归一化 
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    bbox.clip();
    bbox.x1_ *= (Dtype)image_width;
    bbox.x2_ *= (Dtype)image_width;
    bbox.y1_ *= (Dtype)image_height;
    bbox.y2_ *= (Dtype)image_height; 
  }

   
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    Dtype xmin = (int)gt_bbox.x1_;
    Dtype xmax = (int)gt_bbox.x2_;
    Dtype ymin = (int)gt_bbox.y1_;
    Dtype ymax = (int)gt_bbox.y2_;

    if(( xmin >= xmax ) || ( ymin >= ymax ) || gt_bbox.get_height()<=1 || gt_bbox.get_width()<=1 ){
      if(DEBUG_WDH) LOG(INFO) <<"false " << gt_bbox.x1_ << " , "<< gt_bbox.y1_ << " , "<< gt_bbox.x2_<< " , "<< gt_bbox.y2_;
      it = anno.instances.erase(it);  //以迭代器为参数，删除元素3，并把数据删除后的下一个元素位置返回给迭代器。 此时不执行++it;
      anno.num_person -= 1;
    } else{ ++it; } // 
  }

  if(DEBUG_WDH)  LOG(INFO) << "anno num " << anno.instances.size();
} 
 
/*
  获得包含所有gt的最小矩形框, 此矩形框的坐标是原图坐标
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::RelatGtBbox(AnnoData<Dtype> anno, BoundingBox<Dtype>& relat_gt_bbox){
  Dtype minx=0, miny=0, maxx=0, maxy=0;
  for(int i=0; i < anno.instances.size(); ++i){ // func: find 框选所有gt的最小矩形
    BoundingBox<Dtype> bbox = anno.instances[i].bbox;
    if (minx > bbox.x1_) minx = bbox.x1_; 
    if (miny > bbox.y1_) miny = bbox.y1_;
    if (maxx < bbox.x2_) maxx = bbox.x2_;
    if (maxy < bbox.y2_) maxy = bbox.y2_;
  }
  relat_gt_bbox.x1_ = minx;
  relat_gt_bbox.y1_ = miny;
  relat_gt_bbox.x2_ = maxx;
  relat_gt_bbox.y2_ = maxy;
}

/*
  将原图片中的gt 的坐标, 变换为背景图片中的坐标, 如果原图片中的 gt包围圈 超过了背景图的大小, 将gt包围圈缩小到背景图片中, 
  同时对应的 gt 坐标变为背景坐标, 需要缩小时 进行缩小.
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::changeInstance(AnnoData<Dtype> anno, AnnoData<Dtype>& temp_anno, 
    Dtype x_relat_bbox, Dtype y_relat_bbox, 
    BoundingBox<Dtype> relat_gt_bbox, Dtype scale){
    for(int i=0; i < temp_anno.instances.size(); ++i){ // func: find 框选所有gt的最小矩形
      BoundingBox<Dtype>& bbox = temp_anno.instances[i].bbox;
      if(DEBUG_WDH) LOG(INFO) << "实际gt位置 num: " << i <<" , gt 坐标 " << bbox.x1_ << " ,"<< bbox.y1_ << " ,"<< bbox.x2_<< " ,"<< bbox.y2_;
      bbox.x1_ = bbox.x1_ - relat_gt_bbox.x1_ + x_relat_bbox;
      bbox.y1_ = bbox.y1_ - relat_gt_bbox.y1_ + y_relat_bbox;
      bbox.x2_ = bbox.x2_ - relat_gt_bbox.x1_ + x_relat_bbox;
      bbox.y2_ = bbox.y2_ - relat_gt_bbox.y1_ + y_relat_bbox;
      if((scale) < Dtype(1)){
        bbox.x1_ *= scale; 
        bbox.y1_ *= scale;
        bbox.x2_ *= scale;
        bbox.y2_ *= scale;
      }
      if(DEBUG_WDH) LOG(INFO) << "背景gt位置 num: " << i <<" , gt 坐标 " << bbox.x1_ << " ,"<< bbox.y1_ << " ,"<< bbox.x2_<< " ,"<< bbox.y2_;
  }
  // 清洗 
}

/*
  将 原图中的 gt 图片抠出来 resize之后放到背景图上, 
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::cutGtImgChangeBg(cv::Mat raw_image, cv::Mat& bg_img, AnnoData<Dtype> anno, AnnoData<Dtype> temp_anno){
  for(int i=0; i < anno.instances.size(); ++i){
    BoundingBox<Dtype> raw_box = anno.instances[i].bbox;
    BoundingBox<Dtype>& temp_box = temp_anno.instances[i].bbox;

    if(DEBUG_WDH) LOG(INFO) <<"背景gt  "<< temp_box.x1_ << ", " <<  temp_box.y1_ << " , " << temp_box.x2_ << " ,"<< temp_box.y2_ << ", "<< temp_box.get_width() << ", " <<temp_box.get_height();
    if(DEBUG_WDH) LOG(INFO) <<"原图gt  "<< raw_box.x1_ << ", " <<  raw_box.y1_ << " , " << raw_box.x2_ << " ,"<< raw_box.y2_ << ", " << raw_box.get_width() << ", " <<raw_box.get_height();
    cv::Mat raw_image_gt = raw_image(cv::Rect((int)raw_box.x1_, (int)raw_box.y1_, (int)raw_box.get_width(), (int)raw_box.get_height()));
    cv::Mat bg_img_ROI   = bg_img(cv::Rect((int)temp_box.x1_, (int)temp_box.y1_, (int)temp_box.get_width(), (int)temp_box.get_height()));
    cv::resize(raw_image_gt, raw_image_gt, cv::Size((int)temp_box.get_width(), (int)temp_box.get_height()), CV_INTER_AREA);
    if(DEBUG_WDH) LOG(INFO) <<"原图gt  "<< raw_image_gt.cols << ", " <<  raw_image_gt.rows;
    
    cv::addWeighted(bg_img_ROI, 0.0, raw_image_gt, 1.0, 0.0, bg_img_ROI);
  }
}

/*
  获得 gt包围圈相对于 背景图的坐标, 目的是计算gt 包围圈在背景图中的实际位置
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::normRelatGtBox(BoundingBox<Dtype>& temp_box, BoundingBox<Dtype> box){
  temp_box.x1_ = Dtype(0);
  temp_box.y1_ = Dtype(0);
  temp_box.x2_ = box.x2_ - box.x1_;
  temp_box.y2_ = box.y2_ - box.y1_;
}


/*
  修改temp_anno, 
    1. 将temp_anno.img_width 改为背景图的尺寸
    2. 将
*/
template <typename Dtype>
void BBoxDataTransformer<Dtype>::instancesFitBg(cv::Size shape_bg_img, BoundingBox<Dtype>& relat_gt_bbox, 
  AnnoData<Dtype>& temp_anno , AnnoData<Dtype> anno){
  temp_anno = anno ;// 初始化 为相同
  temp_anno.img_width = shape_bg_img.width;
  temp_anno.img_height = shape_bg_img.height;
  if(DEBUG_WDH) LOG(INFO) << "背景图 尺寸 " <<temp_anno.img_width << ", "<< temp_anno.img_height;

  // func: 获得temp box, 
  BoundingBox<Dtype> temp_relat_gt_bbox = relat_gt_bbox;
  normRelatGtBox(temp_relat_gt_bbox, relat_gt_bbox); 
  if(DEBUG_WDH) LOG(INFO) << "归一化 gt包围圈 " << temp_relat_gt_bbox.x1_ <<" , "<< temp_relat_gt_bbox.y1_ <<" , "<< temp_relat_gt_bbox.x2_<<" , "<< temp_relat_gt_bbox.y2_;
  // 缩放比例
  Dtype scale_w = shape_bg_img.width / temp_relat_gt_bbox.get_width();
  Dtype scale_h = shape_bg_img.height / temp_relat_gt_bbox.get_height();
  if(DEBUG_WDH) LOG(INFO) << "w 比例 " << scale_w << ", h 比例 " << scale_h;

  Dtype scale_fit_bg = std::min(scale_w, scale_h); 
  if(scale_fit_bg < Dtype(1)){
    temp_relat_gt_bbox.x1_ *= scale_fit_bg;
    temp_relat_gt_bbox.y1_ *= scale_fit_bg;
    temp_relat_gt_bbox.x2_ *= scale_fit_bg;
    temp_relat_gt_bbox.y2_ *= scale_fit_bg;
  }
  else scale_fit_bg = Dtype(1);

  Dtype x_relat_bbox, y_relat_bbox; // 随机选取 temp_relat_gt_bbox 在背景图上的坐标
  // temp_relat_gt_bbox 在背景图上的 最大位置, 
  Dtype x_bg_max = shape_bg_img.width - temp_relat_gt_bbox.get_width();
  Dtype y_bg_max = shape_bg_img.height - temp_relat_gt_bbox.get_height();
  if(DEBUG_WDH) LOG(INFO) << " gt 包围圈 坐标范围 " << x_bg_max << ", " << y_bg_max << ", " << temp_relat_gt_bbox.get_width() +x_bg_max << " , "<< temp_relat_gt_bbox.get_height() + y_bg_max;
  caffe_rng_uniform(1, std::min(Dtype(0), x_bg_max), std::max(x_bg_max, Dtype(0)), &x_relat_bbox);
  caffe_rng_uniform(1, std::min(Dtype(0), y_bg_max), std::max(y_bg_max, Dtype(0)), &y_relat_bbox);
  if(DEBUG_WDH) LOG(INFO) << " gt 包围圈 随机后位置 " << x_relat_bbox << ", " << y_relat_bbox <<", " << temp_relat_gt_bbox.get_width() + x_relat_bbox << " , "<< temp_relat_gt_bbox.get_height() + y_relat_bbox;

  // 修改gt 参数 fit 背景
  changeInstance(anno, temp_anno, x_relat_bbox, y_relat_bbox, relat_gt_bbox, scale_fit_bg);
}



template <typename Dtype>
void BBoxDataTransformer<Dtype>::cutGtChangeBg(AnnoData<Dtype>& anno , cv::Mat& raw_image, cv::Mat& bg_img){
  // 如果 超过概率
  if (false) { return; }

  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > 0.5) { return; }
  // 找到一张背景图, 获得w,h 
  // cv::Mat bg_img = cv::imread("/home/xjx/awdh_old/my_python/Data_manage/tool/automoldroid/test_augmentation/image4.jpg");  
  // cv::Mat bg_img = cv::imread("/home/xjx/Datasets/ditan/691.jpg");  


  borderImage(bg_img);   

  cv::Size shape_bg_img = bg_img.size();
  BoundingBox<Dtype> relat_gt_bbox;
  unNormalize(anno); // func: 转换为绝对坐标
  RelatGtBbox(anno, relat_gt_bbox);
  if(DEBUG_WDH) LOG(INFO)  << "gt包围圈: " << relat_gt_bbox.x1_ << " , " << relat_gt_bbox.y1_<< " , " << relat_gt_bbox.x2_<< " , " << relat_gt_bbox.y2_;

  // 随机选择 relag_gt_bbox 在新背景上的位置
  AnnoData<Dtype> temp_anno ; 
  instancesFitBg(shape_bg_img, relat_gt_bbox, temp_anno, anno);

  // 将raw_image 中的gt iou扣到 背景图的相应位置.
  cutGtImgChangeBg(raw_image, bg_img, anno, temp_anno);
  raw_image = bg_img;
  anno = temp_anno;
  Normalize(anno);  // 对新anno 进行归一化
} 
// =========================

/*
  背光增广 Backlight
*/
template<typename Dtype>
void BBoxDataTransformer<Dtype>::RandomBacklight(cv::Mat& image, AnnoData<Dtype> anno, int skip_cid){
  //cout<<anno.dataset<<endl;
  //if(anno.dataset == "RemoBlackHandWithoutFace201957") return;
  if (!param_.has_backlight_prob()) { return; }
  float prob;
  float expand_prob = param_.backlight_prob();
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) { return; }
  for (int i = 0; i < anno.instances.size(); ++i) { // func: 对anno 中每个gt进行归一化 
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    bbox.clip();
    int xmin = (int)(bbox.x1_ * anno.img_width);
    int xmax = (int)(bbox.x2_ * anno.img_width);
    int ymin = (int)(bbox.y1_ * anno.img_height);
    int ymax = (int)(bbox.y2_ * anno.img_height); 
    // LOG(INFO) << "cid: " << anno.instances[i].cid;
    if(anno.instances[i].cid == skip_cid) break;
    if(( xmin >= xmax ) || ( ymin >= ymax ) || (xmax - xmin)<=1 || (ymax - ymin)<=1 ){ 
       // LOG(INFO) <<"false " << xmin << " , "<< ymin << " , "<< xmax << " , "<< ymax;
      break;
    }   
    else{
      // LOG(INFO) << xmin<< " , "<< ymin<< " , "<< xmax<< " , "<< ymax;
      cv::Rect roi = cv::Rect(xmin, ymin, (xmax-xmin), (ymax-ymin) );
      cv::Mat gt_img = image(roi).clone();
      gama_com(0.55f, 0.6f, 0.01f, gt_img);
      // RandomBrightness(gt_img, &gt_img, 1, 100);
      AdjustBrightness(gt_img, -40, &gt_img );
      gt_img.copyTo(image(roi));
    }  
  }  
}


template <typename Dtype>
void BBoxDataTransformer<Dtype>::rotate90(cv::Mat &image, AnnoData<Dtype>& anno) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  if ((dice <= param_.rotate90_prob())) {
    cv::transpose(image, image);
    for (int i = 0 ; i < anno.instances.size() ; ++i) {
      Dtype tmp_x1, tmp_x2;
      tmp_x1 = anno.instances[i].bbox.x1_;
      tmp_x2 = anno.instances[i].bbox.x2_;
      anno.instances[i].bbox.x1_ = anno.instances[i].bbox.y1_;
      anno.instances[i].bbox.x2_ = anno.instances[i].bbox.y2_;
      anno.instances[i].bbox.y1_ = tmp_x1;
      anno.instances[i].bbox.y2_ = tmp_x2;
    }
    int tmp = anno.img_width;
    anno.img_width = anno.img_height;
    anno.img_height = tmp;
  }
}


template<typename Dtype>
void BBoxDataTransformer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes) {
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
void BBoxDataTransformer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image) {
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
void BBoxDataTransformer<Dtype>::InitRand() { 
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  } 
  // setupBg(); // func: 初始化抠手背景  
}

template <typename Dtype>
int BBoxDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}


template <typename Dtype>
void BBoxDataTransformer<Dtype>::ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                                                  AnnoData<Dtype>* anno) {
  ptree pt;
  read_xml(xml_file, pt);
  anno->img_path = root_dir + '/' + pt.get<string>("Annotations.ImagePath");
  anno->dataset = pt.get<string>("Annotations.DataSet");
  anno->img_width = pt.get<int>("Annotations.ImageWidth");
  anno->img_height = pt.get<int>("Annotations.ImageHeight");
  try {
    anno->num_person = pt.get<int>("Annotations.NumPerson");
  } catch (const ptree_error &e) {
    anno->num_person = pt.get<int>("Annotations.NumPart");
  }
  anno->instances.clear();
  for (int i = 0; i < anno->num_person; ++i) {
    Instance<Dtype> ins;
    char temp_cid[128], temp_pid[128], temp_iscrowd[128], temp_is_diff[128];
    char temp_xmin[128], temp_ymin[128], temp_xmax[128], temp_ymax[128];
    sprintf(temp_cid, "Annotations.Object_%d.cid", i+1);
    sprintf(temp_pid, "Annotations.Object_%d.pid", i+1);
    sprintf(temp_is_diff, "Annotations.Object_%d.is_diff", i+1);
    sprintf(temp_iscrowd, "Annotations.Object_%d.iscrowd", i+1);
    sprintf(temp_xmin, "Annotations.Object_%d.xmin", i+1);
    sprintf(temp_ymin, "Annotations.Object_%d.ymin", i+1);
    sprintf(temp_xmax, "Annotations.Object_%d.xmax", i+1);
    sprintf(temp_ymax, "Annotations.Object_%d.ymax", i+1);
    // bindex & cid & pid
    ins.bindex = bindex;
    ins.cid = pt.get<int>(temp_cid);  
    // filter crowd & diff
    if (ins.iscrowd || ins.is_diff) continue;
    // bbox: must be defined
    ins.bbox.x1_ = pt.get<Dtype>(temp_xmin);
    ins.bbox.y1_ = pt.get<Dtype>(temp_ymin);
    ins.bbox.x2_ = pt.get<Dtype>(temp_xmax);
    ins.bbox.y2_ = pt.get<Dtype>(temp_ymax);
    anno->instances.push_back(ins);
  }
}

INSTANTIATE_CLASS(BBoxDataTransformer);

}  // namespace caffe
