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

#include "caffe/mask/bbox_data_transformer_multi_resize.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
                                    vector<BBoxData<Dtype> >* bboxes,
                                    BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox) {
  //image is a address. *image is a pic
  Normalize(anno);
  randomExpand(anno, image);
  // randomly crpp
  randomCrop(anno, image, crop_bbox,image_bbox);
  CHECK(image->data);
  // perform distortion
  randomDistortion(image, anno);
  // // flip
  randomFlip(anno, *image, doflip,image_bbox);
  // // resized
  fixedResize(anno, *image);
  // // save to bboxes
  copy_label(anno, bboxes);
  // // visualize
   if(false){
      int image_h = image->rows;
      int image_w = image->cols;
      const cv::Point point1(image_bbox->x1_*image_w, image_bbox->y1_*image_h);
      const cv::Point point2(image_bbox->x2_*image_w, image_bbox->y2_*image_h);
      const cv::Scalar box_color(0, 0, 255);
      const int thickness = 2;
      cv::rectangle(*image, point1, point2, box_color, thickness);
      cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
      cv::imshow("image", *image);
      cv::waitKey(0);
  }
  if (param_.visualize()) {
    visualize(*image, *bboxes);
  }
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
}

template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip) {
  randomExpand(anno, image);

  // randomly cropexpand_ratio
  randomCrop(anno, image);
  CHECK(image->data);
  // perform distortion
  randomDistortion(image, anno);
  // // flip
  randomFlip(anno, *image, doflip);
  // // resized
  fixedResize(anno, *image);
}

//###########################################################
template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::randomExpand(AnnoData<Dtype>& anno, cv::Mat* image) {
    // 读入图片

*image = cv::imread(anno.img_path.c_str());
  if (!param_.has_expand_param()){
    return;
  }
  // 读入图片
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  //LOG(FATAL) << "Error when open image file: " << anno.img_path;
//
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);

  BoundingBox<Dtype> expand_bbox;
  //读取expand参数
  const float max_expand_ratio=param_.expand_param().max_expand_ratio();
  const float expand_prob=param_.expand_param().prob();
  float expand_ratio;
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    return;
  }
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
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
  anno.img_height =height;
  //LOG(INFO)<<"expand_ratio"<<expand_ratio;
  //LOG(INFO)<<"anno.img_width"<<width;

  //随机新框位置
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);

  h_off = floor(h_off);
  w_off = floor(w_off);
  //记录新框相对于旧框的位置
  expand_bbox.x1_=(-w_off/img_width);
  expand_bbox.y1_=(-h_off/img_height);
  expand_bbox.x2_=((width - w_off)/img_width);
  expand_bbox.y2_=((height - h_off)/img_height);
  //图像转换
  cv::Mat expand_img;
  expand_img.create(height, width, image->type());
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
  expand_img.setTo(cv::Scalar(104,117,123,0.0));
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
        //++it;
      }
++it;
  }
  //for test
  bool test=0;
  if(test){
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
void BBoxDataMultiResizeTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox) {
  // image
  //*image = cv::imread(anno.img_path.c_str());
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }


  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  getCropBBox(anno, crop_bbox);
  TransCrop(anno,*crop_bbox,image,image_bbox);
}

template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image) {
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
  *image = image_back(roi);
 }

template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
    // get a random value [0-1]
    float prob = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // get max width & height
    float h_max, w_max;
    h_max = 1.0;
    w_max = 1.0;
    if(param_.has_sample_sixteennine()){
      if(param_.sample_sixteennine()){
        if(anno.img_width>anno.img_height){
          h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
          w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
          // LOG(INFO)<<"16-9";
        }
        else{
          h_max=std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
          w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
          // LOG(INFO)<<"9-16";
        }
      } else {
         h_max = 1.0;
         w_max = 1.0;
      }
    } 
    if(param_.has_sample_sixteennine_one()){
      if(param_.sample_sixteennine_one()){
        CHECK_EQ(param_.wh_ratio_sixteennineone_size(), 2)<<"size of wh_ratio_sixteennineone must be 2";;
        float wh_ratio_min = param_.wh_ratio_sixteennineone(0);
        float wh_ratio_max = param_.wh_ratio_sixteennineone(1);
        CHECK_LT(wh_ratio_min, wh_ratio_max);
        // LOG(INFO)<<"USING sample_sixteennine_one";
        if(float(anno.img_width)/float(anno.img_height)>=wh_ratio_min and wh_ratio_max>=float(anno.img_width)/float(anno.img_height)){
          h_max = std::min(anno.img_height * 1.0, anno.img_width * 1.0) / anno.img_height;
          w_max = std::min(anno.img_height * 1.0, anno.img_width * 1.0) / anno.img_width;
        }
        if (float(anno.img_width)/float(anno.img_height)>wh_ratio_max){
          h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
          w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
        }
        if(float(anno.img_width)/float(anno.img_height)<wh_ratio_min) {
            h_max=std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
            w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
        }
      } 
    }

  if(param_.has_rand_sixteennineone()){
    if(param_.rand_sixteennineone()){
      CHECK_EQ(param_.prob_rand_sixteennineone_size(), 2)<<"size of prob_rand_sixteennineone must be 2";;
      float prob_min = param_.prob_rand_sixteennineone(0);
      float prob_max = param_.prob_rand_sixteennineone(1);
      CHECK_LE(prob_min, prob_max);
      // LOG(INFO)<<"using randsixteenone";
      if(prob < prob_min) {
        // 16:9
        // LOG(INFO)<<"16-9";
        h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
        w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
      } else if (prob>prob_max){
        // LOG(INFO)<<"1-1";
          h_max = std::min(anno.img_height * 1.0, anno.img_width * 1.0) / anno.img_height;
          w_max = std::min(anno.img_height * 1.0, anno.img_width * 1.0) / anno.img_width;
      } else{
        // 9:16
        // LOG(INFO)<<"9-16";
        h_max = std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
        w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
      }
    }
  }
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

      if (param_.has_sample_sixteennine() or param_.has_sample_sixteennine_one() or param_.has_rand_sixteennineone()){
        if(param_.sample_sixteennine() or param_.sample_sixteennine_one() or param_.rand_sixteennineone()){
          GenerateBatchSamples16_9(anno, samplers, &sample_bboxes, h_max, w_max);
          // LOG(INFO)<<"hmax"<<h_max<<"w_max"<<w_max;
        } else {
	   // LOG(INFO)<<"using random scale crop!!!!!!";
          GenerateBatchSamples(anno, samplers, &sample_bboxes);
        }
      } else {
        // LOG(INFO)<<"using random scale crop!!!!!!";

          GenerateBatchSamples(anno, samplers, &sample_bboxes);
      }

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
void BBoxDataMultiResizeTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,
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
  cv::Mat bg(hb, wb, CV_8UC3, cv::Scalar(104, 117, 123));
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
  int pwidth = pxmax - pxmin;
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
   if(false){

      const cv::Point point1(image_bbox->x1_*wb, image_bbox->y1_*hb);
      const cv::Point point2(image_bbox->x2_*wb, image_bbox->y2_*hb);
      const cv::Scalar box_color(0, 0, 255);
      const int thickness = 2;
      cv::rectangle(bg, point1, point2, box_color, thickness);
      cv::namedWindow( "bg", CV_WINDOW_AUTOSIZE );
      cv::imshow("bg", bg);
      cv::waitKey(0);
  }
  *image = bg;
  int num_keep = 0;
  typename vector<Instance<Dtype> >::iterator it;
  for (it = anno.instances.begin(); it != anno.instances.end();) {
    BoundingBox<Dtype>& gt_bbox = it->bbox;
    BoundingBox<Dtype> proj_bbox;
    // keep instance/ emitting true ground truth
    Dtype emit_coverage_thre = 0;
    if (param_.has_emit_coverage_thre()){
      emit_coverage_thre = param_.emit_coverage_thre();//0.25
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

    if (gt_bbox.project_bbox(crop_bbox, &proj_bbox) >= emit_coverage_thre) {
        // box update
        //计算相交面积占原gt面积的比重，若大于阈值则保留
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
void BBoxDataMultiResizeTransformer<Dtype>::randomDistortion(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
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
void BBoxDataMultiResizeTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
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
void BBoxDataMultiResizeTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
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
void BBoxDataMultiResizeTransformer<Dtype>::copy_label(AnnoData<Dtype>& anno, vector<BBoxData<Dtype> >* boxes) {
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
void BBoxDataMultiResizeTransformer<Dtype>::fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
//   // modify header
//   anno.img_width = param_.resized_width();
//   anno.img_height = param_.resized_height();
//   // resize image
//   cv::Mat image_rsz;
//   // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
//   // CHECK(image.data);
//   // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
//   if(param_.has_sample_sixteennine() and param_.sample_sixteennine()){
//   if (image.cols>image.rows){
//   resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);}//16-9
//   else{
//     resize(image, image_rsz, Size(param_.resized_height(), param_.resized_width()), INTER_LINEAR);//9-16
//   }
// }
// if (param_.has_sample_sixteennine_one() and param_.sample_sixteennine_one()){
//   if(float(image.cols)/float(image.rows)>=0.8 and 1.25>=float(image.cols)/float(image.rows)){
//     resize(image, image_rsz, Size(param_.resized_width(), param_.resized_width()), INTER_LINEAR);//1-1
//   }
//   if(float(image.cols)/float(image.rows)>1.25){
//     resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);//16-9
//   }
//   if(float(image.cols)/float(image.rows)<0.8){
//     resize(image, image_rsz, Size(param_.resized_height(), param_.resized_width()), INTER_LINEAR);//9-16
//   }
// }
// if(not (param_.has_sample_sixteennine() and param_.has_sample_sixteennine_one())){
// resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
// }
//   image = image_rsz;

  float max_size=float(max(anno.img_width,anno.img_height));
  float min_size=float(min(anno.img_width,anno.img_height));

  float scale=param_.resized_width()/max_size;
 // LOG(INFO)<<anno.img_width<<"$"<<anno.img_height;

  anno.img_width = anno.img_width*scale;
  anno.img_height = anno.img_height*scale;
   // LOG(INFO)<<anno.img_width<<"$$"<<anno.img_height;

  // resize image
  cv::Mat image_rsz;
  // CHECK(image.data);
  // LOG(INFO)<<param_.resized_width()<<" "<<param_.resized_height()<<" "<<image.cols<<" "<<image.rows;
    //  LOG(INFO)<<image.cols<<"*"<<image.rows;

  resize(image, image_rsz, Size(anno.img_width, anno.img_height), INTER_LINEAR);

  image = image_rsz;
    // LOG(INFO)<<image.cols<<"**"<<image.rows;
 // add new!!!!!#######
    for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    //LOG(INFO)<<bbox.x1_<<"&"<<bbox.y1_<<"&"<<bbox.x2_<<"&"<<bbox.y2_;

    bbox.x1_=bbox.x1_*anno.img_width/param_.resized_width();
    bbox.y1_=bbox.y1_*anno.img_height/param_.resized_width();
    bbox.x2_=bbox.x2_*anno.img_width/param_.resized_width();
    bbox.y2_=bbox.y2_*anno.img_height/param_.resized_width();
    // LOG(INFO)<<anno.img_width<<"@@"<<anno.img_height<<"@@"<<param_.resized_width();
    // LOG(INFO)<<bbox.x1_<<"*"<<bbox.y1_<<"*"<<bbox.x2_<<"*"<<bbox.y2_;
  }
  //##########
}

template<typename Dtype>
void BBoxDataMultiResizeTransformer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes) {
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
void BBoxDataMultiResizeTransformer<Dtype>::visualize(AnnoData<Dtype>& anno, cv::Mat& image) {
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
void BBoxDataMultiResizeTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == TRAIN);
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int BBoxDataMultiResizeTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(BBoxDataMultiResizeTransformer);

}  // namespace caffe
