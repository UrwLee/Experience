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
void BBoxDataTransformer<Dtype>::Pic916To11(AnnoData<Dtype>& anno, cv::Mat& image) {

  int w_off = caffe_rng_rand() % (anno.img_height - anno.img_width);
  cv::Mat newone(anno.img_height, anno.img_height, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat ROI = newone(cv::Rect(w_off, 0, anno.img_width, anno.img_height));
  cv::Mat mask(anno.img_height, anno.img_width, CV_8UC1, cv::Scalar(255));
  image.copyTo(ROI, mask);
  Dtype scale = (Dtype)anno.img_height / anno.img_width;
  cv::resize(newone, image, cv::Size(anno.img_width, anno.img_width));

  for (int i = 0; i < anno.instances.size(); ++i)
  {
    anno.instances[i].bbox.x1_ = (anno.instances[i].bbox.x1_ + w_off) / scale;
    anno.instances[i].bbox.y1_ = (anno.instances[i].bbox.y1_) / scale;
    anno.instances[i].bbox.x2_ = (anno.instances[i].bbox.x2_ + w_off) / scale;
    anno.instances[i].bbox.y2_ = (anno.instances[i].bbox.y2_) / scale;
  }
  anno.img_height = anno.img_width;
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomBlock(AnnoData<Dtype>& anno, cv::Mat* image) 
{
 
 // LOG(INFO)<<anno.img_path<<std::endl;
  
  
  std::vector<int> peopleid;
  std::vector<int> headandhair;
  BoundingBox<Dtype> cover;
  Dtype margin=-1;
  peopleid.clear();
  int block_color = param_.block_color();
  //统计人的个数以及id
  for(int i = 0;i < anno.instances.size();i++)
  { 
    if(anno.instances[i].cid == 0)
    peopleid.push_back(i);
 
  }
  

  //增广的概率
  Dtype CoverProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);

  //使用增广的判断条件
  if (param_.make_randomblock()&&phase_ == TRAIN&&CoverProb < param_.randomblock_prob())
  {
      //为保证效果仅在人数为1和2时使用增广
      if (peopleid.size()<=2)
      {
          //按照人数为1和2两种情况确定包围人的最小包围框
          BoundingBox<Dtype>  boxcoverpeople;
          if (peopleid.size()==1)
          {
            boxcoverpeople = anno.instances[peopleid[0]].bbox;
          }
          if (peopleid.size()==2)
          {
            boxcoverpeople.x1_ = std::min(anno.instances[peopleid[0]].bbox.x1_,anno.instances[peopleid[1]].bbox.x1_);
            boxcoverpeople.y1_ = std::min(anno.instances[peopleid[0]].bbox.y1_,anno.instances[peopleid[1]].bbox.y1_);
            boxcoverpeople.x2_ = std::max(anno.instances[peopleid[0]].bbox.x2_,anno.instances[peopleid[1]].bbox.x2_);
            boxcoverpeople.y2_ = std::max(anno.instances[peopleid[0]].bbox.y2_,anno.instances[peopleid[1]].bbox.y2_);
          }
          
          headandhair.clear();
          margin = 0; 
          //储存所有与最小包围框有交集的头发和头的框
          for (int i = 0; i < anno.instances.size(); ++i)
          {
            if(anno.instances[i].cid == 2||anno.instances[i].cid == 5)
            {
              //bool flagx = (anno.instances[i].bbox.x1_>boxcoverpeople.x1_ && anno.instances[i].bbox.x1_<boxcoverpeople.x2_)||(anno.instances[i].bbox.x2_>boxcoverpeople.x1_&&anno.instances[i].bbox.x2_<boxcoverpeople.x2_);
              //bool flagy = (anno.instances[i].bbox.y1_>boxcoverpeople.y1_ && anno.instances[i].bbox.y1_<boxcoverpeople.y2_)||(anno.instances[i].bbox.y2_>boxcoverpeople.y1_&&anno.instances[i].bbox.y2_<boxcoverpeople.y2_);
              Dtype iou = anno.instances[i].bbox.compute_iou(boxcoverpeople);
              //LOG(INFO)<<"iou:"<<iou;
              if (iou!=0)
                headandhair.push_back(i);

            }
          }

          //遮挡块的y1应该小于所有头发和头的框的y2
          for (int i = 0; i < headandhair.size(); ++i)
          {
            margin = std::max(margin,anno.instances[headandhair[i]].bbox.y2_);
          }
          //异常情况直接退出
          if (margin > boxcoverpeople.y2_||margin == -1) return;

          //随机生成遮挡块的高度
          Dtype margincut = ((static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX))*0.5 + 0.5)*(boxcoverpeople.y2_ - margin);
              //LOG(INFO) <<"margincut:"<<margincut<<endl;
          for (int i = 0; i < peopleid.size(); ++i)
          {
            anno.instances[peopleid[i]].bbox.y2_ =  std::min(boxcoverpeople.y2_ - margincut,anno.instances[peopleid[i]].bbox.y2_);
          }
              
          //存下遮挡块的框
          cover.x1_ = boxcoverpeople.x1_;
          cover.y1_ = boxcoverpeople.y2_ - margincut;
          cover.x2_ = boxcoverpeople.x2_;
          cover.y2_ = boxcoverpeople.y2_ ;

        //对该框进行填充
        int x1_o = int(cover.x1_*anno.img_width);
        int y1_o = int(cover.y1_*anno.img_height);
        int w = anno.img_width*cover.get_width();
        int h = anno.img_height*cover.get_height();

                if(block_color == 0)
        for (int row=y1_o; row<y1_o+static_cast<int>(h);row++)
          {
            for (int col=x1_o; col < x1_o+static_cast<int>(w);col++)
            {
              int b = 104;
              int g = 117;
              int r = 123;
              image->at<Vec3b>(row,col) = cv::Vec3b(b,g,r);
            }          
          }
        if(block_color == 1)
        for (int row=y1_o; row<y1_o+static_cast<int>(h);row++)
          {
            for (int col=x1_o; col < x1_o+static_cast<int>(w);col++)
            {
              
              int b = static_cast<int>(rand()%255);
              int g = static_cast<int>(rand()%255);
              int r = static_cast<int>(rand()%255);
              image->at<Vec3b>(row,col) = cv::Vec3b(b,g,r);
            }          
          }
        if(block_color == 2)
        for (int row=y1_o; row<y1_o+static_cast<int>(h);row++)
          {
            for (int col=x1_o; col < x1_o+static_cast<int>(w);col++)
            {
              if( static_cast<int>(rand()%2)==0)
              {
                int b = 0;
                int g = 0;
                int r = 0;
                 image->at<Vec3b>(row,col) = cv::Vec3b(b,g,r);
              }
              else
              {
                int b = 255;
                int g = 255;
                int r = 255;
                image->at<Vec3b>(row,col) = cv::Vec3b(b,g,r);
              }

            }          
          }  

    }
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomPerspective(AnnoData<Dtype>& anno, cv::Mat& image) {

  Dtype x1 = 10000.0;
  Dtype x2 = 0.0;
  Dtype y1 = 10000.0;
  Dtype y2 = 0.0;
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype> box = anno.instances[i].bbox;
    int cid = anno.instances[i].cid;
    if (cid == 0)
    {
      x1 = min(box.x1_, x1);
      x2 = max(box.x2_, x2);
      y1 = min(box.y1_, y1);
      y2 = max(box.y2_, y2);
    }
  }
  int w = anno.img_width;
  int h = anno.img_height;

  cv::Point2f xy1 = cv::Point2f(0, 0);
  cv::Point2f xy2 = cv::Point2f(w, 0);
  cv::Point2f xy3 = cv::Point2f(w, h);
  cv::Point2f xy4 = cv::Point2f(0, h);

  float WorH;
  caffe_rng_uniform(1, 0.0f, 1.f, &WorH);
  if (WorH > 0.5) {
    //H
    float scaleDecrease;
    caffe_rng_uniform(1, 0.0f, float(min(y1, (1.f - y2))), &scaleDecrease);
    scaleDecrease = min(scaleDecrease,0.08f);
    float UporDown;
    caffe_rng_uniform(1, 0.0f, 1.f, &UporDown);
    if (UporDown > 0.5) {
      //Up
      float LeftorRight;
      caffe_rng_uniform(1, 0.0f, 1.f, &LeftorRight);
      if (LeftorRight < 0.5) {
        //left
        xy1.y += scaleDecrease * h;
      } else if (0.5 <= LeftorRight) {
        //right
        xy2.y += scaleDecrease * h;
      }
    } else {
      //Down
      float LeftorRight;
      caffe_rng_uniform(1, 0.0f, 1.f, &LeftorRight);
      if (LeftorRight < 0.5) {
        //right
        xy3.y -= scaleDecrease * h;
      } else if (0.5 <= LeftorRight) {
        //left
        xy4.y -= scaleDecrease * h;
      }
    }
  } else {
    //W
    float scaleDecrease;
    caffe_rng_uniform(1, 0.0f, float(min(x1, (1.f - x2))), &scaleDecrease);
    scaleDecrease = min(scaleDecrease,0.08f);
    float LeftorRight;
    caffe_rng_uniform(1, 0.0f, 1.f, &LeftorRight);
    if (LeftorRight > 0.5) {
      //Left
      float UporDown;
      caffe_rng_uniform(1, 0.0f, 1.f, &UporDown);
      if (UporDown < 0.5) {
        //up
        xy1.x += scaleDecrease * w;
      } else if (0.5 <= UporDown ) {
        //down
        xy4.x += scaleDecrease * w;
      }
    } else {
      //right
      float UporDown;
      caffe_rng_uniform(1, 0.0f, 1.f, &UporDown);
      if (UporDown < 0.5) {
        //up
        xy2.x -= scaleDecrease * w;
      } else {
        //down
        xy3.x -= scaleDecrease * w;
      }
    }
  }

  cv::Point2f src_points[] = {
    cv::Point2f(0, 0),
    cv::Point2f(w, 0),
    cv::Point2f(w, h),
    cv::Point2f(0, h)
  };

  cv::Point2f dst_points[] = {xy1, xy2, xy3, xy4};

  cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
  cv::warpPerspective(image, image, M, cv::Size(anno.img_width, anno.img_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
    vector<BBoxData<Dtype> >* bboxes,
    BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox) {

  *image = cv::imread(anno.img_path.c_str());

  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  // 读入图片
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }

  // perform distortion
  randomDistortion(image, anno);

  if((phase_ == TRAIN)&&(anno.dataset != "BH_Hair_Head")) {
    Dtype DarkProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (DarkProb < param_.dark_prop()) {
      gama_com(param_.dark_gamma_min(), param_.dark_gamma_max(), 0.01, *image);
    }
    Dtype BlurFrontProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (BlurFrontProb < param_.blur_front_prob()) {
      blur(anno, *image);
    }
  }

  if (phase_ == TRAIN) {
    if (param_.picninesixteentooneone() && (((Dtype)anno.img_height / anno.img_width) > 1.2)) {
      Pic916To11(anno, *image);
    }
  }

  if (phase_ == TRAIN) {
    if (param_.transpose_prob() != 0.0) {
      rotate90_standPerson(*image, anno);
    }
  }
  Normalize(anno);
   if (phase_ == TRAIN)
  {
    int count_size_hair = 0;
    if (param_.make_addadditionhair_prob()!=0.0)
    {
      if (anno.dataset == "BH_Hair_Head")
      {
        count_size_hair = StoreHair(image, anno);
      }
      if(anno.dataset == "AIC_REMO")
      {
        Dtype hair_prob = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
        
        if(hair_prob < param_.make_addadditionhair_prob()&&(StoreSingleHair_.size()>8))
        {
          //LOG(INFO)<<"count_size_hair:"<<StoreSingleHair_.size();
          //cv::namedWindow("original");
          //cv::imshow("original",*image); 
          AddAdditionHair(anno, image);
           // int num_instance_original=anno.instances.size();
          //for(int i = 0;i<num_instance_original;i++)
          //{ 
          //  anno.instances[i].bbox.DrawBoundingBoxNorm(image);
          //}
          //cv::namedWindow("hair");
          //cv::imshow("hair",*image);
          //if (cv::waitKey(0) == 113)
          //LOG(FATAL)<<"quit";
        }
      }
    }
  }
  if (anno.dataset == "BH_Hair_Head")
  randomBlock(anno, image);
  // perform expand

  if (phase_ == TRAIN) {
    Dtype PerspectiveProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (PerspectiveProb < param_.perspective_prop()) {
      randomPerspective(anno, *image);
    }
  }

  if (phase_ == TRAIN) {
    Dtype PartProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (PartProb < param_.part_prob()) {
      randomExpand(anno, image);
      RandomBacklight(*image,  anno);
    }
  }


  // randomly crpp
  if (anno.dataset != "CA_Lie_ALL" && anno.dataset != "BH_Hair_Head")
  {
    randomCrop(anno, image, crop_bbox, image_bbox);
  }
  CHECK(image->data);

  if (phase_ == TRAIN) {
    int count_size = 0;
    if (param_.merge_single_person_prob() != 0.0) {
      count_size = StorePreson(image, anno);
    }

    Dtype MspProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (MspProb < param_.merge_single_person_prob() && StoreSingle_.size() != 0) {
      AddAdditionPreson(anno, image);
    }
  }
  // // flip
  if (phase_ == TRAIN) {
    if (param_.xflip_prob() != 0.0) {
      randomXFlip(*image, anno, image_bbox);
    }
    Dtype BlurFrontPartProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (BlurFrontPartProb < param_.blur_front_part_prob()) {
      partBlur(anno, *image);
    }
  }
  randomFlip(anno, *image, doflip, image_bbox);
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
void BBoxDataTransformer<Dtype>::rotate90(cv::Mat &image, AnnoData<Dtype>& anno) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  if ((dice <= param_.transpose_prob())) {
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

template <typename Dtype>
void BBoxDataTransformer<Dtype>::rotate90_standPerson(cv::Mat &image, AnnoData<Dtype>& anno) {
  float dice = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  if ((dice <= param_.transpose_prob())) {
    for (int i = 0; i < anno.instances.size(); ++i) {
      if (anno.instances[i].cid != 0){
        continue; 
      }
      Dtype img_width = anno.img_width;
      Dtype img_height = anno.img_height;
      Dtype instance_w = anno.instances[i].bbox.x2_ - anno.instances[i].bbox.x1_;
      Dtype instance_h = anno.instances[i].bbox.y2_ - anno.instances[i].bbox.y1_;
      float instance_hw_ratio = instance_h/instance_w;
      float area_ratio = (instance_h*instance_w)/img_width/img_height;
      float area_ratio_ = (float)param_.area_ratio();
      float hw_ratio = (float)param_.hw_ratio();
      if (instance_hw_ratio > hw_ratio && area_ratio < area_ratio_){
        continue;
      }
      else
      {
        return;
      }
    }
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
void BBoxDataTransformer<Dtype>::RandomBacklight(cv::Mat& image, AnnoData<Dtype> anno) {
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

    if (( xmin >= xmax ) || ( ymin >= ymax ) || (xmax - xmin) <= 1 || (ymax - ymin) <= 1 ) {
      // LOG(INFO) <<"false " << xmin << " , "<< ymin << " , "<< xmax << " , "<< ymax;
      break;
    }
    else {
      // LOG(INFO) << xmin<< " , "<< ymin<< " , "<< xmax<< " , "<< ymax;
      cv::Rect roi = cv::Rect(xmin, ymin, (xmax - xmin), (ymax - ymin) );
      cv::Mat gt_img = image(roi).clone();
      gama_com(0.55f, 0.6f, 0.01f, gt_img);
      // RandomBrightness(gt_img, &gt_img, 1, 100);
      AdjustBrightness(gt_img, -40, &gt_img );
      gt_img.copyTo(image(roi));
    }
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image, bool* doflip) {

  *image = cv::imread(anno.img_path.c_str());
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  // 读入图片
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  // perform distortion
  if (param_.neg_more_aug()){
	  randomDistortion_neg(image, anno);
	  if (phase_ == TRAIN) {
	    Dtype DarkProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
	    if (DarkProb < param_.back_dark_prop()) {
	      gama_com_neg(param_.dark_gamma_min_neg(), param_.dark_gamma_max_neg(), 0.01, *image);
	    }
	    Dtype BlurBackProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
	    if (BlurBackProb < param_.blur_back_prob()) {
	      blur(anno, *image);
	    }
	  }
  }else{
	randomDistortion(image, anno);
	  if (phase_ == TRAIN) {
	    Dtype DarkProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
	    if (DarkProb < param_.back_dark_prop()) {
	      gama_com(param_.dark_gamma_min(), param_.dark_gamma_max(), 0.01, *image);
	    }    
	    Dtype BlurBackProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
	    if (BlurBackProb < param_.blur_back_prob()) {
	      blur(anno, *image);
	    }    
	  }

  }
  // perform expand
  randomExpand(anno, image);
  // randomly crop
  randomCrop(anno, image);
  CHECK(image->data);
  // // flip
  randomFlip(anno, *image, doflip);
  // // resized
  fixedResize(anno, *image);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::blur(AnnoData<Dtype>& anno, cv::Mat& image) {
  int kernel_prob = rand() % 4;
  int array[4] = {3, 5, 7, 9};
  cv::blur(image, image, cv::Size(array[kernel_prob], array[kernel_prob]));
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::partBlur(AnnoData<Dtype>& anno, cv::Mat& image) {
  int kernel_prob = rand() % 4;
  int array[4] = {3, 5, 7, 9};
  Dtype threshold = param_.blur_size_threshold();
  for (int i = 0; i < anno.instances.size(); ++i) {
    int height = anno.img_height;
    int width = anno.img_width;
    int cid = anno.instances[i].cid;
    if (cid == 0) {
      BoundingBox<Dtype>& BdBbox = anno.instances[i].bbox;
      if (BdBbox.compute_area() > threshold) {
        cv::Rect rect(BdBbox.x1_ * width ,  BdBbox.y1_ * height , (BdBbox.x2_ - BdBbox.x1_)*width , (BdBbox.y2_ - BdBbox.y1_)*height);
        cv::Mat ROI = image(rect);
        cv::blur(ROI, ROI, cv::Size(array[kernel_prob], array[kernel_prob]));
      }
    }
  }
}

// //###########################################################
template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomExpand(AnnoData<Dtype>& anno, cv::Mat* image) {
  // 读入图片

  if (!param_.has_expand_param()) {
    return;
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
  anno.img_height = height;
  //LOG(INFO)<<"expand_ratio"<<expand_ratio;
  //LOG(INFO)<<"anno.img_width"<<width;

  //随机新框位置
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);

  h_off = floor(h_off);
  w_off = floor(w_off);
  //记录新框相对于旧框的位置
  expand_bbox.x1_ = (-w_off / img_width);
  expand_bbox.y1_ = (-h_off / img_height);
  expand_bbox.x2_ = ((width - w_off) / img_width);
  expand_bbox.y2_ = ((height - h_off) / img_height);
  //图像转换
  cv::Mat expand_img;
  expand_img.create(height, width, image->type());
  // LOG(INFO)<<"R"<<mean_values[0];
  // LOG(INFO)<<"G"<<mean_values[1];
  // LOG(INFO)<<"B"<<mean_values[2];
  expand_img.setTo(cv::Scalar(104, 117, 123, 0.0));
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
    if (gt_bbox.project_bbox(expand_bbox, &proj_bbox) >= emit_coverage_thre) {
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
  bool test = 0;
  if (test) {
    int i = fabs(caffe_rng_rand() % 1000);
    std::stringstream ss;
    std::string str;
    ss << i;
    ss >> str;
    for (it = anno.instances.begin(); it != anno.instances.end();) {
      BoundingBox<Dtype> proj_bbox = it->bbox;
      cv::rectangle(*image, cvPoint(int(proj_bbox.x1_ * width), int(proj_bbox.y1_ * height)), cvPoint(int(proj_bbox.x2_ * width), int(proj_bbox.y2_ * height)), Scalar(255, 0, 0), 1, 1, 0);
      ++it;
    }
    cv::imwrite("/home/xjx/xjx/AIC/" + str + ".jpg", *image);
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
  TransCrop(anno, *crop_bbox, image, image_bbox);
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
  float min_scale = samplers[idx].sampler().max_scale();

  Dtype BackProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
  if (BackProb < param_.back_makeborder_prob() && min_scale > 1.0 && anno.dataset=="tree") {
    float scale;
    caffe_rng_uniform(1, 0.1f, 1.0f, &scale);
    int image_w = image->cols;
    int image_h = image->rows;
    int img_width_new = std::max((int)(image_h * 16.0 / 9.0), image_w);
    int img_height_new = std::max((int)(image_w * 9.0 / 16.0), image_h);
    cv::copyMakeBorder(*image, *image, 0, img_height_new - image_h, 0, img_width_new - image_w, cv::BORDER_WRAP);
    int scale_h = scale * image->rows;
    int scale_w = scale * image->cols;
    cv::resize(*image, *image, cv::Point(scale_w, scale_h));
    cv::copyMakeBorder(*image, *image, img_height_new - scale_h, 0, img_width_new - scale_w, 0, cv::BORDER_WRAP);
    anno.img_width = image->cols;
    anno.img_height = image->rows;
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::getCropBBox(AnnoData<Dtype>& anno, BoundingBox<Dtype>* crop_bbox) {
  // get a random value [0-1]
  // float prob = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  // get max width & height
  float h_max, w_max;
  if (param_.has_sample_sixteennine() || param_.has_sample_ninesixteen()) {
    if (param_.sample_sixteennine()) {
      h_max = std::min(anno.img_height * 1.0, anno.img_width * 9.0 / 16.0) / anno.img_height;
      w_max = std::min(anno.img_height * 16.0 / 9.0, anno.img_width * 1.0) / anno.img_width;
    } else if (param_.sample_ninesixteen()) {
      h_max = std::min(anno.img_height * 1.0, anno.img_width * 16.0 / 9.0) / anno.img_height;
      w_max = std::min(anno.img_height * 9.0 / 16.0, anno.img_width * 1.0) / anno.img_width;
    }
  } else {
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
    if (param_.sample_random()) { // default is false, if true,it means to crop accord body or part based on probability
      GenerateBatchSamplesRandom16_9(anno, samplers, &sample_bboxes, h_max, w_max);
    } else {
      if (param_.for_body()) {  // default is true, this datalayer use for body-crop
        if (param_.ytop()) {
          GenerateBatchSamples16_9_ytop(anno, samplers, &sample_bboxes, h_max, w_max);
        } else if (param_.havehead()) {
          GenerateBatchSamples16_9_havehead(anno, samplers, &sample_bboxes, h_max, w_max);
        } else {
          GenerateBatchSamples16_9(anno, samplers, &sample_bboxes, h_max, w_max);
        }
      } else {                  // this datalayer use for part-crop
        if (param_.crop_around_gt()) { // func: 默认为false, 围绕gt 进行采样
          GenerateBatchSamples4PartsAroundGT16_9(anno, samplers, &sample_bboxes, h_max, w_max);
        }
        else {
          GenerateBatchSamples4Parts16_9(anno, samplers, &sample_bboxes, h_max, w_max);
        }
      }
    }

    if (sample_bboxes.size() > 0) {
      int idx = caffe_rng_rand() % sample_bboxes.size();
      *crop_bbox = sample_bboxes[idx];
    } else {
      crop_bbox->x1_ = 0.5 - w_max / 2.0;
      crop_bbox->x2_ = 0.5 + w_max / 2.0;
      crop_bbox->y1_ = 0.5 - h_max / 2.0;
      crop_bbox->y2_ = 0.5 + h_max / 2.0;
    }
  } else {
    crop_bbox->x1_ = 0.5 - w_max / 2.0;
    crop_bbox->x2_ = 0.5 + w_max / 2.0;
    crop_bbox->y1_ = 0.5 - h_max / 2.0;
    crop_bbox->y2_ = 0.5 + h_max / 2.0;
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,const BoundingBox<Dtype>& crop_bbox,cv::Mat* image, BoundingBox<Dtype>* image_bbox) {
  

  if (param_.speed_up())
  {
    Dtype wrap_w = crop_bbox.get_width()*anno.img_width;
    Dtype wrap_h = crop_bbox.get_height()*anno.img_height;

    Dtype scale_w =  wrap_w/(Dtype)param_.resized_width();
    Dtype scale_h =  wrap_h/(Dtype)param_.resized_height();
    cv::resize(*image, *image, Size((int)(image->cols/scale_w), (int)(image->rows/scale_h)),INTER_LINEAR);
  }
  const int img_width = image->cols;
  const int img_height = image->rows;
  // image crop:
  /**
   * 意思是我们采样的时候可能得到的crop_box超出边界的范围,此时用均值背景进行填充
   We need to copy [] of image -> [] of bg
   */
  
  float make_copy_border_thred = param_.make_copy_border_thred();
  float make_copy_border_prob = param_.make_copy_border_prob();
  float crop_bbox_area = crop_bbox.compute_area();
  //LOG(INFO)<<crop_bbox.compute_area()<<endl;
  int wb = std::ceil(crop_bbox.get_width() * img_width);
  int hb = std::ceil(crop_bbox.get_height() * img_height);
  float copyprob = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  //LOG(INFO)<scrop_bbox.get_width()<<" "<<crop_bbox.get_height();
  anno.img_width = wb;
  anno.img_height = hb;
  //LOG(INFO)<<crop_bbox.get_width()<<" "<<crop_bbox.get_height();
  //cv::Mat copy1 = *image;
  //int num_instance_original=anno.instances.size();
  //for(int i = 0;i<num_instance_original;i++)
  //{ 
  //  anno.instances[i].bbox.DrawBoundingBoxNorm(&copy1);
  //}
  //cv::namedWindow("Original Image");
  //cv::imshow("Original Image",copy1);
  //LOG(INFO) <<"before:"<<num_instance_original<<" instances";
  if(param_.make_copy_border()&&(copyprob<=make_copy_border_prob)&&(crop_bbox_area>make_copy_border_thred))
  {
    //LOG(INFO)<<anno.img_path<<std::endl;
    int top=std::floor(std::max(Dtype(0)-crop_bbox.y1_,Dtype(0))*img_height);
    int bottom = std::floor(std::max(crop_bbox.y2_-Dtype(1),Dtype(0))*img_height);
    int left = std::floor(std::max(Dtype(0)-crop_bbox.x1_,Dtype(0))*img_width);
    int right = std::floor(std::max(crop_bbox.x2_-Dtype(1),Dtype(0))*img_width);
    //LOG(INFO)<<"top bottom left right"<<top<<bottom<<left<<right<<std::endl;
    int pxmin = (int)(std::max(crop_bbox.x1_, Dtype(0)) * img_width);
    int pymin = (int)(std::max(crop_bbox.y1_, Dtype(0)) * img_height);
    int pxmax = std::floor(std::min(crop_bbox.x2_, Dtype(1)) * img_width) - 1;
    int pymax = std::floor(std::min(crop_bbox.y2_, Dtype(1)) * img_height) - 1;
    

    int pwidth=pxmax-pxmin;
    int pheight=pymax-pymin;

    cv::Mat image_patch(pheight, pwidth, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Rect orig_patch(pxmin, pymin, pxmax-pxmin, pymax-pymin);

    (*image)(orig_patch).copyTo(image_patch);
    cv::Mat expand_img;
    copyMakeBorder(image_patch,expand_img,top,bottom,left,right,BORDER_WRAP);
    *image=expand_img;

    int num_h = std::ceil(top/static_cast<float>(pheight))+std::ceil(bottom/static_cast<float>(pheight))+1;
    int num_w = std::ceil(left/static_cast<float>(pwidth))+std::ceil(right/static_cast<float>(pwidth))+1;
    int num_instance=anno.instances.size();
    //LOG(INFO)<<"num_h:"<<num_h<<"  num_w:"<<num_w<<"   num_instance"<<num_instance<<std::endl;

    
    //int num_add=0;
    for(int i=0;i<num_instance;i++)
    {
      
      float w_begin=anno.instances[i].bbox.x2_;
      float h_begin=anno.instances[i].bbox.y2_;
      //LOG(INFO)<<w_begin<<"  <--x2  y2-->"<<h_begin<<std::endl;
      
      while(w_begin>std::ceil(crop_bbox.x1_)&&crop_bbox.x1_<0)
        w_begin-=1;
      while(h_begin>std::ceil(crop_bbox.y1_)&&crop_bbox.y1_<0)
        h_begin-=1;
      //LOG(INFO)<<"w_begin:"<<w_begin-anno.instances[i].bbox.get_width()<<"   h_begin:"<<h_begin-anno.instances[i].bbox.get_height()<<std::endl;
      
      for(int j=0;j<num_h;j++)
        for(int k=0;k<num_w;k++)
        {
            Instance<Dtype> tmp=anno.instances[i];
            tmp.bbox.y1_=h_begin+j-anno.instances[i].bbox.get_height();
            tmp.bbox.y2_=h_begin+j;
            tmp.bbox.x1_=w_begin+k-anno.instances[i].bbox.get_width();
            tmp.bbox.x2_=w_begin+k;
            //num_add++;
            //LOG(INFO)<<tmp.bbox.x2_<<" <--x2 add  y2-->"<<tmp.bbox.y2_<<std::endl;
            anno.instances.push_back(tmp);
        }

    }
    //LOG(INFO)<<"num_add:"<<num_add<<"   num_instance:"<<anno.instances.size()<<std::endl;
    anno.instances.erase(anno.instances.begin(),anno.instances.begin()+num_instance);
  }
else{
  cv::Mat bg(hb, wb, CV_8UC3, cv::Scalar(128, 128, 128));
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

}
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
        /*根据不同的gt面积设置不同的gt滤出值*/
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

    Dtype scale_crop = gt_bbox.project_bbox(crop_bbox, &proj_bbox);
    if (param_.torsowithhead_coverage() != 0.0 && it->cid == 0 && scale_crop > param_.delete_from_scale()) {
      if (anno.dataset == "AICDataWithTorse" ) {
        BoundingBox<Dtype> THproj_bbox;
        Dtype TH_crop_scale = it->THbbox.project_bbox(crop_bbox, &THproj_bbox);
        Dtype THcoverage = THproj_bbox.compute_obj_coverage(proj_bbox);
        if (THcoverage < param_.torsowithhead_coverage()) {
          scale_crop = 0.0;
        }
      } else {
        scale_crop = 0.0;
      }
    }
    if ( scale_crop >= emit_coverage_thre) {
      // box update
      // LOG(INFO) << "project_bbox_area " << gt_bbox.project_bbox(crop_bbox, &proj_bbox)<<", emit_coverage_thre "<<emit_coverage_thre;
      it->bbox = proj_bbox;
      ++num_keep;
      ++it;
    } else {
        bool do_ignore_gt = param_.do_ignore_gt();
        if(do_ignore_gt){
          //LOG(INFO)<<"----------------------";
          it->bbox = proj_bbox;
          it->cid = 100;
          //LOG(INFO)<<it->cid;
          ++num_keep;
          ++it;
        }
        else
          it = anno.instances.erase(it);
      }
      
    // if (gt_bbox.project_bbox(crop_bbox, &proj_bbox) >= emit_coverage_thre) {
    //     // box update
    //     // LOG(INFO) << "project_bbox_area " << gt_bbox.project_bbox(crop_bbox, &proj_bbox)<<", emit_coverage_thre "<<emit_coverage_thre;
    //     it->bbox = proj_bbox;
    //     ++num_keep;
    //     ++it;
    //   } else {
    //     it = anno.instances.erase(it);
    //   }
  }
  anno.num_person = num_keep;
  
  //cv::Mat copy2 = *image;
  //num_instance_original=anno.instances.size();
  //for(int i = 0;i<num_instance_original;i++)
  //{ 
  //  anno.instances[i].bbox.DrawBoundingBoxNorm(&copy2);
  //}
  //LOG(INFO) <<"after:"<<num_instance_original<<" instances";
  //cv::namedWindow("transcrop_img");
  //cv::imshow("transcrop_img",copy2);
  //if (cv::waitKey(0) == 113)
  //  LOG(FATAL)<<"quit";
}


template <typename Dtype>
void BBoxDataTransformer<Dtype>::arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y) {
  // check x1 > x2
  int num = (int)((x1 - x2) / stride);
  for (int i = 0; i < num; ++i) {
    y[i] = x2 + i * stride;
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::adjust_gama(Dtype gama, cv::Mat &image) {
  int table[256];
  Dtype ivgama = 1.0 / gama;
  cv::Mat lut(1, 256, CV_8U);
  cv::Mat out;
  unsigned char *p = lut.data;
  for (int i = 0; i < 256; ++i) {
    table[i] = pow((i / 255.0), ivgama) * 255;
    p[i] = table[i];
  }
  cv::LUT(image, lut, image);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::gama_com_neg(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image) {
  int num = (int)((max_gama - min_gama) / stride_gama);
  Dtype list_gama[num];
  arange(min_gama, max_gama, stride_gama, list_gama);
  int random_num = caffe_rng_rand() % num;
  CHECK_LT(random_num, num);
  if (list_gama[random_num] <= 0) list_gama[random_num] = 0.01f;
  Dtype gama = list_gama[random_num];
  adjust_gama(gama, image);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::gama_com(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image) {
  int num = (int)((max_gama - min_gama) / stride_gama);
  Dtype list_gama[num];
  arange(min_gama, max_gama, stride_gama, list_gama);
  int random_num = caffe_rng_rand() % num;
  CHECK_LT(random_num, num);
  if (list_gama[random_num] <= 0) list_gama[random_num] = 0.01f;
  Dtype gama = list_gama[random_num];
  adjust_gama(gama, image);
}
template <typename Dtype>
int BBoxDataTransformer<Dtype>::StoreHair(cv::Mat* image, AnnoData<Dtype>& anno) 
{

  //LOG(INFO) <<"begin";  
  // for (int i = 0; i < anno.instances.size(); ++i)
  //{
  //  LOG(INFO) <<"test"<<anno.instances[i].bbox.x1_;
  //  LOG(INFO) <<"test"<<anno.instances[i].bbox.y1_;
  //  LOG(INFO) <<"test"<<anno.instances[i].bbox.x2_;
  //  LOG(INFO) <<"test"<<anno.instances[i].bbox.y2_;
  // }
  //LOG(INFO) <<"Done";
  std::vector<int> counter;
  counter.clear();
  BoundingBox<Dtype> BdBbox_hair;
  int person_num = 0;
  int head_num = 0;
  int head_id = 0;
  bool flag = 0;
  Dtype thres = 1;
  for (int i = 0; i < anno.instances.size(); ++i) 
  {
    int cid = anno.instances[i].cid;
    if (cid == 0) person_num++;
    if (cid == 2)
    {
      if (head_num == 0) head_id = i;
      head_num++;
    }
    if (cid == 5) 
    {
      counter.push_back(i);
    }
  }
  if (head_num <= 1 && counter.size() == 1 && person_num == 1)
  {
    if (head_num == 1)
    thres = anno.instances[head_id].bbox.compute_obj_coverage(anno.instances[counter[0]].bbox);
    else
    thres = 0;
  }
  //LOG(INFO)<<thres<<endl;
    if (counter.size() == 0)
  return StoreSingleHair_.size();
  if (counter.size() == 1&&thres < 0.2)
  flag = 1;
  //LOG(INFO)<<"width:"<<anno.instances[counter[0]].bbox.get_width();
  //LOG(INFO)<<"width:"<<anno.instances[counter[0]].bbox.y2_;
  //LOG(INFO)<<"width:"<<anno.instances[counter[0]].bbox.y1_;
  //LOG(INFO)<<"img_width:"<<anno.img_width<<endl;
  //LOG(INFO)<<"height:"<<anno.instances[counter[0]].bbox.get_height()<<endl;
  //LOG(INFO)<<"img_height:"<<anno.img_height<<endl;
  if ((anno.instances[counter[0]].bbox.get_height()*anno.img_height<200))
  flag = 0;
  if (flag) 
  {
    if(head_num == 0)
      BdBbox_hair = anno.instances[counter[0]].bbox;
    else
    {
      BdBbox_hair.x1_ = std::min(anno.instances[counter[0]].bbox.x1_,anno.instances[head_id].bbox.x1_);
      BdBbox_hair.y1_ = std::min(anno.instances[counter[0]].bbox.y1_,anno.instances[head_id].bbox.y1_);
      BdBbox_hair.x2_ = std::max(anno.instances[counter[0]].bbox.x2_,anno.instances[head_id].bbox.x2_);
      BdBbox_hair.y2_ = std::max(anno.instances[counter[0]].bbox.y2_,anno.instances[head_id].bbox.y2_);
    }
    //LOG(INFO)<<thres<<endl;
    cv::Rect rect(BdBbox_hair.x1_ * image->cols,  
                  BdBbox_hair.y1_ * image->rows,
                  (BdBbox_hair.x2_ - BdBbox_hair.x1_)*image->cols, 
                  (BdBbox_hair.y2_ - BdBbox_hair.y1_)*image->rows);
    //LOG(INFO)<<anno.instances[counter[0]].bbox.x1_ * image->cols<<endl;
    //LOG(INFO)<<anno.instances[counter[0]].bbox.x1_;
    //LOG(INFO)<<y1 * image->rows<<endl;
    //LOG(INFO)<<(x2 - x1)*image->cols<<endl;
    //LOG(INFO)<<(y2 - y1)*image->rows<<endl;
    //LOG(INFO)<<"roi"<<endl;
    cv::Mat image_roi = (*image).clone()(rect);
    StoreSingleHair_.push_back(image_roi);
    //LOG(INFO)<<StoreSingleHair_.size()<<endl;
    //cv::namedWindow("hair");
    //cv::imshow("hair",image_roi);
    //if (cv::waitKey(0) == 113)
    //LOG(FATAL)<<"quit";

  }
  //LOG(INFO)<<"OK"<<endl;
  while (StoreSingleHair_.size() > param_.single_hair_size()) 
  {
    StoreSingleHair_.erase(StoreSingleHair_.begin());
  }
  return StoreSingleHair_.size();

}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::AddAdditionHair(AnnoData<Dtype>& anno, cv::Mat* image) 
//diandeng
{ 

  int human_count = 0;
  std::vector<int> counter_head;
  int cid;
  Dtype hair_up = 0.3;
  Dtype hair_lr = 0.15;
  counter_head.clear();
  std::vector<int> counter_face;
  std::vector<BoundingBox<Dtype> > covers;
  covers.clear();
  std::vector<int> hair_id;
  hair_id.clear();
  counter_face.clear();

  for (int i = 0; i < anno.instances.size();i++) 
  {                       
      cid = anno.instances[i].cid;
      if (cid == 2) 
      {
          counter_head.push_back(i);
      }
      if (cid == 3) 
      {
          counter_face.push_back(i);
      }
      if (cid == 0) 
      {
          human_count++;
      }

  }
  //在1到2人的图片上使用头发的增广，此处随机从头发库里抽取，并在头发满足高宽比>1.5的情况下才继续，最多尝试10次随机挑选
  if (human_count<=2)
  {
    int rand_ind;
    Dtype hair_ratio = 0;
    int try_times=0;
    while(hair_id.size()!=human_count)
    {
      try_times++;
      rand_ind = static_cast<int>(rand()%StoreSingleHair_.size());
      hair_ratio = Dtype(StoreSingleHair_[rand_ind].rows)/StoreSingleHair_[rand_ind].cols;
      if(hair_ratio<1.5){}
      else {hair_id.push_back(rand_ind);}
      if (try_times>10)
      {
        return;
      }
    }   

    for (int i = 0; i < counter_head.size(); ++i)
    {
      cv::Mat hair = StoreSingleHair_[hair_id[i%hair_id.size()]];
      //LOG(INFO)<<"pick!";
      //将抠出来的头发根据图中头的大小调整
      //0.2 0.1是我选择的经验数值可以更好的覆盖原图
      int hair_w = hair.cols;
      int hair_h = hair.rows;
      hair_ratio = Dtype(hair_h)/hair_w;
      Dtype head_w = anno.instances[counter_head[i]].bbox.get_width()*anno.img_width;
      Dtype head_h = anno.instances[counter_head[i]].bbox.get_height()*anno.img_height;
      int head_x1 = std::max(int(anno.instances[counter_head[i]].bbox.x1_*anno.img_width-0.2*head_w),0);
      int head_y1 = std::max(int(anno.instances[counter_head[i]].bbox.y1_*anno.img_height-0.1*head_h),0);
      hair_w = 1.5*head_w;
      hair_h = hair_ratio*hair_w;
      hair_w = std::min(hair_w,anno.img_width-head_x1);
      hair_h = std::min(hair_h,anno.img_height-head_y1);
      
      cv::Mat hair_resize;
      cv::resize(hair, hair_resize, Size(hair_w,hair_h), INTER_LINEAR);
      cv::Mat img_roi;
      BoundingBox<Dtype> hairtocover;
      //存下头发的框
      hairtocover.x1_ = float(head_x1)/anno.img_width;
      hairtocover.y1_ = float(head_y1)/anno.img_height; 
      hairtocover.x2_ = hairtocover.x1_ + float(hair_w)/anno.img_width;
      hairtocover.y2_ = hairtocover.y1_ + float(hair_h)/anno.img_height;
      covers.push_back(hairtocover);

      cv::Rect rect(head_x1,head_y1,hair_w,hair_h);
      //LOG(INFO)<<image->cols<<" and "<<image->rows;
      img_roi = (*image)(rect);
      cv::addWeighted(img_roi,0.25,hair_resize,0.75,0,hair_resize);
      hair_resize.copyTo(img_roi);
    }
    //for (int i = 0; i < covers.size(); ++i)
    //{
      //将被头发遮挡的框去除
      for (int i = 0; i < covers.size(); ++i)
      {
        typename vector<Instance<Dtype> >::iterator it;
        for (it = anno.instances.begin(); it != anno.instances.end();)
        {
          if(it->cid!=0)
          {
            Dtype iou = it->bbox.compute_iou(covers[i]);
            if(iou!=0) 
              it = anno.instances.erase(it);
            else
              it++; 
          }
          else
            it++;  
        }
      }

  }
  

}

template <typename Dtype>
int BBoxDataTransformer<Dtype>::StorePreson(cv::Mat* image, AnnoData<Dtype>& anno) {
  std::vector<int> counter;
  for (int i = 0; i < anno.instances.size(); ++i) {
    int cid = anno.instances[i].cid;
    if (cid == 0) {
      counter.push_back(i);
    }
  }
  BoundingBox<Dtype>& BdBbox = anno.instances[counter[0]].bbox;
  std::vector<Instance<Dtype> > Instances;
  if (counter.size() == 1) {
    for (int i = 0; i < anno.instances.size(); ++i) {
      BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
      if (bbox.x1_ > BdBbox.x1_ &&
          bbox.x2_ < BdBbox.x2_ &&
          bbox.y1_ > BdBbox.y1_ &&
          bbox.y2_ < BdBbox.y2_) {
        BoundingBox<Dtype> bbox_stay = bbox;
        bbox_stay.x1_ = (bbox.x1_ - BdBbox.x1_) / (BdBbox.x2_ - BdBbox.x1_);
        bbox_stay.y1_ = (bbox.y1_ - BdBbox.y1_) / (BdBbox.y2_ - BdBbox.y1_);
        bbox_stay.x2_ = (bbox.x2_ - BdBbox.x1_) / (BdBbox.x2_ - BdBbox.x1_);
        bbox_stay.y2_ = (bbox.y2_ - BdBbox.y1_) / (BdBbox.y2_ - BdBbox.y1_);
        Instance<Dtype> aa;
        aa = anno.instances[i];
        aa.bbox = bbox_stay;
        Instances.push_back(aa);
      }
    }

    //CHECK_GE(BdBbox.x1_,0);
    //CHECK_GE(BdBbox.y1_,0);
    //CHECK_GE(image->cols,BdBbox.x2_);
    //if(image->rows<BdBbox.y2_){
    //  LOG(INFO)<<anno.img_path;
    //}
    //CHECK_GE(image->rows,BdBbox.y2_);
    cv::Rect rect(BdBbox.x1_ * image->cols,  BdBbox.y1_ * image->rows,
                  (BdBbox.x2_ - BdBbox.x1_)*image->cols, (BdBbox.y2_ - BdBbox.y1_)*image->rows);
    cv::Mat image_roi = (*image).clone()(rect);
    StoreSingle_.push_back(make_pair(image_roi, Instances));
  }
  while (StoreSingle_.size() > param_.single_person_size()) {
    StoreSingle_.erase(StoreSingle_.begin());
  }
  return counter.size();
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::AddAdditionPreson(AnnoData<Dtype>& anno, cv::Mat* image) {
  Dtype add_left = param_.add_left();
  Dtype add_right = param_.add_right();
  Dtype x_closer = param_.x_closer();
  Dtype y_closer = param_.y_closer();
  Dtype size_closer = param_.size_scale();
  int loop = param_.loop();
  Dtype xmax = 0;
  Dtype xmin = 1.0;
  Dtype y_refer_min = 0.0;
  Dtype y_refer_max = 0.0;
  Dtype y_cen_min = 0.0;
  Dtype y_cen_max = 0.0;
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    int cid = anno.instances[i].cid;
    if (cid == 0) {
      if (bbox.x1_ < xmin)
      {
        xmin = bbox.x1_;
        y_refer_min = bbox.y2_ - bbox.y1_;
        y_cen_min = (bbox.y2_ + bbox.y1_) / 2.0;
      }
      if (bbox.x2_ > xmax)
      {
        xmax = bbox.x2_;
        y_refer_max = bbox.y2_ - bbox.y1_;
        y_cen_max = (bbox.y2_ + bbox.y1_) / 2.0;
      }
    }
  }
  if (xmin >= add_left) {
    int rand_ind = -StoreSingle_.size() * static_cast<Dtype>(rand()) / (RAND_MAX + 1);
    cv::Mat MergePic = StoreSingle_[rand_ind].first;
    int merge_x = MergePic.cols;
    int merge_y = MergePic.rows;

    Dtype recive_y;
    Dtype real_scale;
    Dtype recive_x;
    Dtype recive_x_position;
    Dtype recive_y_position;
    int count = 0;
    while (count < loop) {
      Dtype rand_position_x = (static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX)) * x_closer + 1.0 - x_closer;
      Dtype rand_position_y = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX) * (y_closer * 2) / 1.0 + (1.0 - y_closer);
      Dtype rand_scale = (static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX)) * (size_closer * 2) / 1.0 + (1.0 - size_closer);

      recive_y = (rand_scale * y_refer_min) * ((Dtype)image->rows);
      real_scale = recive_y / merge_y;
      recive_x = real_scale * merge_x;
      recive_x_position = rand_position_x * (image->cols * xmin - recive_x );
      recive_y_position = rand_position_y * image->rows * y_cen_min - recive_y / 2.0;

      if (recive_x_position < 0 || recive_y_position < 0 || (recive_x_position + recive_x) > image->cols * xmin ||
          int(recive_y) < 0 || int(recive_x) < 0 || (recive_y_position + recive_y) > image->rows) {
        count++;
        continue;
      }
      break;
      count++;
    }
    if (count == 50) {
      return;
    }
    cv::Mat roi = (*image)(Rect((int)recive_x_position, (int)recive_y_position, (int)recive_x, (int)recive_y));
    cv::resize(MergePic, MergePic, cv::Size((int)recive_x, (int)recive_y));
    cv::Mat mask(roi.rows, roi.cols, roi.depth(), Scalar(1));
    MergePic.copyTo(roi, mask);

    Instance<Dtype> Ins;
    Ins.bindex = anno.instances[0].bindex;
    Ins.cid = 0;
    Ins.pid = anno.instances[0].pid;
    Ins.is_diff = anno.instances[0].is_diff;
    Ins.iscrowd = anno.instances[0].iscrowd;
    Ins.mask_included = anno.instances[0].mask_included;
    Ins.kps_included = anno.instances[0].kps_included;
    BoundingBox<Dtype> BdBbox;
    BdBbox.x1_ = recive_x_position / image->cols;
    BdBbox.x2_ = (recive_x_position + recive_x) / image->cols;
    BdBbox.y1_ = recive_y_position / image->rows;
    BdBbox.y2_ = (recive_y_position + recive_y) / image->rows;
    Ins.bbox = BdBbox;
    anno.instances.push_back(Ins);

    std::vector<Instance<Dtype> >& Instances = StoreSingle_[rand_ind].second;
    for (int i = 0; i < Instances.size(); ++i) {
      Instance<Dtype> Ins_part = Instances[i];
      BoundingBox<Dtype>& PdBbox = Ins_part.bbox;

      PdBbox.x1_ = BdBbox.x1_ + PdBbox.x1_ * (BdBbox.x2_ - BdBbox.x1_);
      PdBbox.y1_ = BdBbox.y1_ + PdBbox.y1_ * (BdBbox.y2_ - BdBbox.y1_);
      PdBbox.x2_ = BdBbox.x1_ + PdBbox.x2_ * (BdBbox.x2_ - BdBbox.x1_);
      PdBbox.y2_ = BdBbox.y1_ + PdBbox.y2_ * (BdBbox.y2_ - BdBbox.y1_);
      anno.instances.push_back(Ins_part);
    }
  }
  if (xmax < add_right) {
    int rand_ind = -StoreSingle_.size() * static_cast<Dtype>(rand()) / (RAND_MAX + 1);
    cv::Mat MergePic = StoreSingle_[rand_ind].first;
    int merge_x = MergePic.cols;
    int merge_y = MergePic.rows;

    Dtype recive_y;
    Dtype real_scale;
    Dtype recive_x;
    Dtype recive_x_position;
    Dtype recive_y_position;
    int count = 0;
    while (count < loop) {
      Dtype rand_position_x = (static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX)) * x_closer;
      Dtype rand_position_y = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX) * (y_closer * 2) / 1.0 + (1.0 - y_closer);
      Dtype rand_scale = (static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX)) * (size_closer * 2) / 1.0 + (1.0 - size_closer);

      recive_y = (rand_scale * y_refer_max) * ((Dtype)image->rows);
      real_scale = recive_y / merge_y;
      recive_x = real_scale * merge_x;
      recive_x_position = image->cols * xmax + rand_position_x * (image->cols * (1.0 - xmax) - recive_x);
      recive_y_position = rand_position_y * image->rows * y_cen_max - recive_y / 2.0;

      if (recive_x_position < image->cols * xmax || recive_y_position < 0 || (recive_x_position + recive_x) > image->cols ||
          int(recive_y) < 0 || int(recive_x) < 0 || (recive_y_position + recive_y) > image->rows) {
        count++;
        continue;
      }
      break;
      count++;
    }
    if (count == 50) {
      return;
    }
    cv::Mat roi = (*image)(Rect((int)recive_x_position, (int)recive_y_position, (int)recive_x, (int)recive_y));
    cv::resize(MergePic, MergePic, cv::Size((int)recive_x, (int)recive_y));
    cv::Mat mask(roi.rows, roi.cols, roi.depth(), Scalar(1));
    MergePic.copyTo(roi, mask);

    Instance<Dtype> Ins;
    Ins.bindex = anno.instances[0].bindex;
    Ins.cid = 0;
    Ins.pid = anno.instances[0].pid;
    Ins.is_diff = anno.instances[0].is_diff;
    Ins.iscrowd = anno.instances[0].iscrowd;
    Ins.mask_included = anno.instances[0].mask_included;
    Ins.kps_included = anno.instances[0].kps_included;
    BoundingBox<Dtype> BdBbox;
    BdBbox.x1_ = recive_x_position / image->cols;
    BdBbox.x2_ = (recive_x_position + recive_x) / image->cols;
    BdBbox.y1_ = recive_y_position / image->rows;
    BdBbox.y2_ = (recive_y_position + recive_y) / image->rows;
    Ins.bbox = BdBbox;
    anno.instances.push_back(Ins);

    std::vector<Instance<Dtype> >& Instances = StoreSingle_[rand_ind].second;
    for (int i = 0; i < Instances.size(); ++i) {
      Instance<Dtype> Ins_part = Instances[i];
      BoundingBox<Dtype>& PdBbox = Ins_part.bbox;
      PdBbox.x1_ = BdBbox.x1_ + PdBbox.x1_ * (BdBbox.x2_ - BdBbox.x1_);
      PdBbox.y1_ = BdBbox.y1_ + PdBbox.y1_ * (BdBbox.y2_ - BdBbox.y1_);
      PdBbox.x2_ = BdBbox.x1_ + PdBbox.x2_ * (BdBbox.x2_ - BdBbox.x1_);
      PdBbox.y2_ = BdBbox.y1_ + PdBbox.y2_ * (BdBbox.y2_ - BdBbox.y1_);
      anno.instances.push_back(Ins_part);
    }
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomDistortion_neg(cv::Mat* image, AnnoData<Dtype>& anno) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.cols, anno.img_width);
  CHECK_EQ(img.rows, anno.img_height);
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param_neg());
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
void BBoxDataTransformer<Dtype>::randomFlip(AnnoData<Dtype>& anno, cv::Mat& image,
    bool* doflip, BoundingBox<Dtype>* image_bbox) {
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
void BBoxDataTransformer<Dtype>::Normalize(AnnoData<Dtype>& anno) {
//
  const int image_width = anno.img_width;
  const int image_height = anno.img_height;
  //
  for (int i = 0; i < anno.instances.size(); ++i) {
    Instance<Dtype>& ins = anno.instances[i];
    BoundingBox<Dtype>& bbox = ins.bbox;
    bbox.x1_ /= (Dtype)image_width;
    bbox.x2_ /= (Dtype)image_width;
    bbox.y1_ /= (Dtype)image_height;
    bbox.y2_ /= (Dtype)image_height;
    if (ins.cid == 0 && anno.dataset == "AICDataWithTorse") {
      ins.THbbox.x1_ /= (Dtype)image_width;
      ins.THbbox.y1_ /= (Dtype)image_height;
      ins.THbbox.x2_ /= (Dtype)image_width;
      ins.THbbox.y2_ /= (Dtype)image_height;
    }
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
    boxes->push_back(bbox_element);
  }
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::


fixedResize(AnnoData<Dtype>& anno, cv::Mat& image) {
  // modify header
  anno.img_width = param_.resized_width();
  anno.img_height = param_.resized_height();
  // resize image
  cv::Mat image_rsz;
  resize(image, image_rsz, Size(param_.resized_width(), param_.resized_height()), INTER_LINEAR);
  image = image_rsz;
}

template<typename Dtype>
void BBoxDataTransformer<Dtype>::visualize(cv::Mat& image, vector<BBoxData<Dtype> >& boxes) {
  cv::Mat img_vis = image.clone();
  static int counter = 0;
  static const int color_maps[18] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 128, 0, 128, 128, 128, 128, 255};
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
    int r = color_maps[3 * (cid % 6)];
    int g = color_maps[3 * (cid % 6) + 1];
    int b = color_maps[3 * (cid % 6) + 2];
    bbox_real.Draw(r, g, b, &img_vis);
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
  static const int color_maps[18] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 128, 0, 128, 128, 128, 128, 255};
  for (int i = 0; i < anno.instances.size(); ++i) {
    BoundingBox<Dtype>& bbox = anno.instances[i].bbox;
    int r = color_maps[3 * (i % 6)];
    int g = color_maps[3 * (i % 6) + 1];
    int b = color_maps[3 * (i % 6) + 2];
    // draw box
    BoundingBox<Dtype> bbox_real;
    bbox_real.x1_ = bbox.x1_ * img_vis.cols;
    bbox_real.y1_ = bbox.y1_ * img_vis.rows;
    bbox_real.x2_ = bbox.x2_ * img_vis.cols;
    bbox_real.y2_ = bbox.y2_ * img_vis.rows;
    if (anno.instances[i].iscrowd) {
      bbox_real.Draw(0, 0, 0, &img_vis);
    } else {
      bbox_real.Draw(r, g, b, &img_vis);
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
}

template <typename Dtype>
int BBoxDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
    static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}








//ClipIgnoreGT
template <typename Dtype>
void BBoxDataTransformer<Dtype>::randomCrop(AnnoData<Dtype>& anno, cv::Mat* image, BoundingBox<Dtype>* crop_bbox, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes) {
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }
  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  getCropBBox(anno, crop_bbox);  // TODO: use Part
  TransCrop(anno, *crop_bbox, image, image_bbox, ignore_bboxes);
}

template <typename Dtype>
void BBoxDataTransformer<Dtype>::Transform(AnnoData<Dtype>& anno, cv::Mat* image,
    vector<BBoxData<Dtype> >* bboxes,
    BoundingBox<Dtype>* crop_bbox, bool* doflip, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes) {

  *image = cv::imread(anno.img_path.c_str());

  CHECK_EQ(image->cols, anno.img_width);
  CHECK_EQ(image->rows, anno.img_height);
  // 读入图片
  if (!image->data) {
    LOG(FATAL) << "Error when open image file: " << anno.img_path;
  }

  // perform distortion
  randomDistortion(image, anno);

  if (phase_ == TRAIN) {
    Dtype DarkProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (DarkProb < param_.dark_prop()) {
      gama_com(param_.dark_gamma_min(), param_.dark_gamma_max(), 0.01, *image);
    }
  }

  if (phase_ == TRAIN) {
    if (param_.picninesixteentooneone() && (((Dtype)anno.img_height / anno.img_width) > 1.2)) {
      Pic916To11(anno, *image);
    }
  }

  if (phase_ == TRAIN) {
    if (param_.transpose_prob() != 0.0) {
      rotate90(*image, anno);
    }
  }
  Normalize(anno);
  // perform expand

  if (phase_ == TRAIN) {
    Dtype PartProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (PartProb < param_.part_prob()) {
      randomExpand(anno, image);
      RandomBacklight(*image,  anno);
    }
  }


  // randomly crpp
  randomCrop(anno, image, crop_bbox, image_bbox, ignore_bboxes);
  CHECK(image->data);

  if (phase_ == TRAIN) {
    int count_size = 0;
    if (param_.merge_single_person_prob() != 0.0) {
      count_size = StorePreson(image, anno);
    }

    Dtype MspProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (MspProb < param_.merge_single_person_prob() && StoreSingle_.size() != 0) {
      AddAdditionPreson(anno, image);
    }
  }
  // // flip
  if (phase_ == TRAIN) {
    if (param_.xflip_prob() != 0.0) {
      randomXFlip(*image, anno, image_bbox);
    }
  }
  randomFlip(anno, *image, doflip, image_bbox);
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
void BBoxDataTransformer<Dtype>::TransCrop(AnnoData<Dtype>& anno,
    const BoundingBox<Dtype>& crop_bbox,
    cv::Mat* image, BoundingBox<Dtype>* image_bbox, std::vector<BoundingBox<Dtype> >& ignore_bboxes) {
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
  int pwidth = pxmax - pxmin;
  int pheight = pymax - pymin;
  cv::Rect orig_patch(pxmin, pymin, pwidth, pheight);
  // (3) patch of bg
  int xmin_bg = std::floor(crop_bbox.x1_ * img_width);
  int ymin_bg = std::floor(crop_bbox.y1_ * img_height);
  // LOG(INFO)<<"pxmin - xmin_bg "<<pxmin - xmin_bg<<" pymin - ymin_bg "<<pymin - ymin_bg<<" pwidth "<<pwidth<<" pheight "<<pheight;
  cv::Rect bg_patch(pxmin - xmin_bg, pymin - ymin_bg, pwidth, pheight);
  image_bbox->x1_ = (Dtype)(pxmin - xmin_bg) / (Dtype)wb;
  image_bbox->x2_ = (Dtype)(pxmin - xmin_bg + pwidth) / (Dtype)wb;
  image_bbox->y1_ = (Dtype)(pymin - ymin_bg) / (Dtype)hb;
  image_bbox->y2_ = (Dtype)(pymin - ymin_bg + pheight) / (Dtype)hb;
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
    if (phase_ == TRAIN) {
      if (param_.has_emit_coverage_thre()) {
        emit_coverage_thre = param_.emit_coverage_thre();
      } else {
        Dtype area_gt = gt_bbox.compute_area();
        // LOG(INFO) << "area_gt " << area_gt;
        for (int s = 0; s < param_.emit_area_check_size(); ++s) {
          if (area_gt < param_.emit_area_check(s)) {
            emit_coverage_thre = param_.emit_coverage_thre_multiple(s);
            break;
          }
        }
      }
    } else {
      emit_coverage_thre = param_.emit_coverage_thre();
    }
    //For training ignoreWithTH, set delete_from_scale and torsowithhead_coverage to control which GT to be ignored
    //Because of falsealarm usually appear in 1/32 featuremap ,so only the GT==Person and scale_crop>delete_from_scale to be ignored
    //Then this person will be cut and calculate the 'THcoverage' with its THbbox and gt_bbox
    Dtype scale_crop = gt_bbox.project_bbox(crop_bbox, &proj_bbox);
    if (param_.torsowithhead_coverage() != 0.0 && it->cid == 0 && scale_crop > param_.delete_from_scale()) {
      if (anno.dataset == "AICDataWithTorse" ) {
        BoundingBox<Dtype> THproj_bbox;
        Dtype TH_crop_scale = it->THbbox.project_bbox(crop_bbox, &THproj_bbox);
        Dtype THcoverage = THproj_bbox.compute_obj_coverage(proj_bbox);
        if (THcoverage < param_.torsowithhead_coverage()) {
          scale_crop = 0.0;
        }
      } else {
        scale_crop = 0.0;
      }
    }
    if ( scale_crop >= emit_coverage_thre) {
      // box update
      // LOG(INFO) << "project_bbox_area " << gt_bbox.project_bbox(crop_bbox, &proj_bbox)<<", emit_coverage_thre "<<emit_coverage_thre;
      it->bbox = proj_bbox;
      ++num_keep;
      ++it;
    } else {
      if (it->cid == 0)
      {
        ignore_bboxes.push_back(proj_bbox);
      }
      it = anno.instances.erase(it);
    }
  }
  anno.num_person = num_keep;
}








INSTANTIATE_CLASS(BBoxDataTransformer);

}  // namespace caffe
