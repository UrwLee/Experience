// if not use OPENCV, note it.
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// if not use, note it.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
// Google log & flags
#include "gflags/gflags.h"
#include "glog/logging.h"
// caffe
#include "caffe/proto/caffe.pb.h"
#include "caffe/caffe.hpp"
// remo, note the useless classes.
#include "caffe/remo/remo_front_visualizer.hpp"
#include "caffe/remo/net_wrap.hpp"
#include "caffe/remo/frame_reader.hpp"
#include "caffe/remo/data_frame.hpp"
#include "caffe/remo/basic.hpp"
#include "caffe/remo/res_frame.hpp"
#include "caffe/remo/visualizer.hpp"

#include "caffe/mask/bbox_func.hpp"

#include "caffe/tracker/basic.hpp"
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "caffe/det/detwrap.hpp"

using namespace std;
using namespace caffe;
using std::string;
using std::vector;
namespace bfs = boost::filesystem;
void SplitString(const std::string& s, std::vector<string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}
int main(int nargc, char** args) {
   // ################################ NETWORK ################################
   // network input
   int resized_width = 512;
   int resized_height = 288;
   const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_PD.prototxt";
   const std::string caffe_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel";


   // GPU
   int gpu_id = 0;
   bool mode = true;  // use GPU
   // features
   const std::string proposals = "det_out";
     // const std::string proposals = "detection_out_3";
   // display Size
   int max_dis_size = 1280;
   // active labels
   vector<int> active_label;
   active_label.push_back(3);
   // ################################ DATA ####################################
   // CAMERA
   const bool use_camera = false; // 0
   const int cam_width = 1280;
   const int cam_height = 720;
   // VIDEO
   const bool use_video = true;
   const std::string video_file = "/home/ethan/work/doubleVideo/video_raw4.avi";
   const int start_frame = 0;
   // IMAGE LIST
   const bool use_image_list = false;
   const std::string image_dir = "/home/ethan/workspace/det_images/images";
   const std::string dst_dir   = "/home/ethan/workspace/det_images/images_drawn";
   // SINGLE IMAGE
   const bool use_image_file = false;
   const std::string image_file = "/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/AIC_SRC/keypoint_validation_images_20170911/a3c6e3a7563d0e9611ad928099a9a30ff7a297f1.jpg";
   const std::string image_save_file = "1_det.jpg";

   // IMAGE LIST TXT
   const bool use_image_list_txt = true;
   // const std::string img_list_txt = "/home/ethan/DataSets/REID/MARS-v160809/Layout/Layout_MARS_v160809_train.txt";
   // const std::string image_dir_txt = "/home/ethan/DataSets/REID/MARS-v160809";

   const std::string img_list_txt = "/media/ethan/RemovableDisk/Datasets/REMO_HandPose/HandCapture_20180313/ImgsWhole_20180313/1/aa.txt";
   const std::string image_dir_txt = "/media/ethan/RemovableDisk/Datasets/REMO_HandPose/HandCapture_20180313/ImgsWhole_20180313/1";

   const std::string dst_dir_txt   = "/home/ethan/workspace/det_images/images_drawn";
   // ################################ MAIN LOOP ################################
   // det_warpper
   caffe::DetWrapper<float> det_wrapper(network_proto,caffe_model,mode,gpu_id,proposals,max_dis_size);
   //  CAMERA
  int write_width = 960;
  int write_height = 540;
   if (use_camera) {
     cv::VideoCapture cap;
     if (!cap.open(0)) {
       LOG(FATAL) << "Failed to open webcam: " << 0;
     }
     cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
     int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
     cv::VideoWriter outputVideo; 
     outputVideo.open("minihand_test_video_tmp.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(write_width,write_height), true);
     cv::Mat cv_img;
     cap >> cv_img;
     int count = 0;
     CHECK(cv_img.data) << "Could not load image.";
     while (1) {
       ++count;
       cv::Mat image;
       cap >> image;
       caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
       vector<LabeledBBox<float> > roi;
       cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
       for(int i =0;i<roi.size();i++){
         LabeledBBox<float> l_bbox = roi[i];
         float area = (l_bbox.bbox.x2_-l_bbox.bbox.x1_)*(l_bbox.bbox.y2_-l_bbox.bbox.y1_)/float(resized_width)/float(resized_height);
         LOG(INFO)<<"area; "<<area<<"; bboxsize: "<<std::sqrt(area)<<".";
       }
       cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
       cv::imshow( "RemoDet", det_image);
       cv::waitKey(1);
       cv::Mat image_rsz;
       cv::resize(image, image_rsz, cv::Size(write_width,write_height), cv::INTER_LINEAR);
       outputVideo << image_rsz;
     }
   }
   // VIDEO
   if (use_video) {
     
     cv::VideoCapture cap;
     if (!cap.open(video_file)) {
       LOG(FATAL) << "Failed to open video: " << video_file;
     }
     int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
     cv::VideoWriter outputVideo; 
     outputVideo.open("DetBD_I_L_d3.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(write_width,write_height), true);
     int total_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
     cap.set(CV_CAP_PROP_POS_FRAMES, start_frame);
     int processed_frames = start_frame + 1;
     cv::Mat cv_img;
     cap >> cv_img;
     CHECK(cv_img.data) << "Could not load image.";
     while (1) {
       // if (processed_frames < total_frames) {
      // cap.set(1,100);
         cv::Mat image;
         cap >> image;
         caffe::DataFrame<float> data_frame(processed_frames, image, resized_width, resized_height);
         vector<LabeledBBox<float> > roi;
         cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
          for(int i =0;i<roi.size();i++){
         LabeledBBox<float> l_bbox = roi[i];
         float area = (l_bbox.bbox.x2_-l_bbox.bbox.x1_)*(l_bbox.bbox.y2_-l_bbox.bbox.y1_)/float(resized_width)/float(resized_height);
         // LOG(INFO)<<"area; "<<area<<"; bboxsize: "<<std::sqrt(area)<<" cid"<<l_bbox.cid<<".";
         }
         cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
         cv::imshow( "RemoDet", det_image);
         cv::waitKey(0);
         cv::Mat image_rsz;
         cv::resize(det_image, image_rsz, cv::Size(write_width,write_height), cv::INTER_LINEAR);
         outputVideo << image_rsz;
       // } else {
       //   LOG(INFO) << "Video processed finished.";
       //   break;
       // }
       ++processed_frames;
     }
   }
   // IMAGES LIST
   if (use_image_list) {
     vector<std::string> images;
     const boost::regex annotation_filter(".*\\.jpg");
     caffe::find_matching_files(image_dir, annotation_filter, &images);
     if (images.size() == 0) {
       LOG(FATAL) << "Error: Found no jpg files in " << image_dir;
     }
     for (int i = 0; i < images.size(); ++i) {
       const std::string image_p = image_dir + '/' + images[i];
       cv::Mat image = cv::imread(image_p);
       caffe::DataFrame<float> data_frame(i,image,resized_width,resized_height);
       vector<LabeledBBox<float> > roi;
       cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
       cv::imwrite(dst_dir + '/' + images[i], det_image);
       LOG(INFO) << i << "/" << images.size() << ": process for image: " << images[i];
     }
   }
   //  SINGLE IMAGE
   if (use_image_file) {
     cv::Mat image = cv::imread(image_file);
     caffe::DataFrame<float> data_frame(0,image,resized_width,resized_height);
     vector<LabeledBBox<float> > roi;
     cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
      LOG(INFO)<<roi.size();
      cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
      cv::imshow( "RemoDet", det_image);
      cv::waitKey(0);
      cv::imwrite(image_save_file, det_image);
   }

   if(use_image_list_txt){
      std::ifstream infile(img_list_txt.c_str());
      CHECK(infile.good()) << "Failed to open file "<< img_list_txt;
      std::vector<std::string> lines;
      std::string str_line;
      while (std::getline(infile, str_line)) {
        lines.push_back(str_line);
      }
      
      for (int i=0;i < lines.size();++i){
        std::vector<string> v;
        SplitString(lines[i], v," ");
        std::string image_path = image_dir_txt + '/' + v[0];
        cv::Mat image = cv::imread(image_path);
        cv::Mat image_pad;
        // cv::copyMakeBorder(image,image_pad,16,16,192,192,cv::BORDER_CONSTANT,cv::Scalar(0));
        cv::copyMakeBorder(image,image_pad,80,80,224,224,cv::BORDER_CONSTANT,cv::Scalar(0));
        caffe::DataFrame<float> data_frame(0,image_pad,resized_width,resized_height);
        vector<LabeledBBox<float> > roi;
        cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
        LOG(INFO)<<"image_path "<<image_path<<" "<<roi.size();
        cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
        cv::imshow( "RemoDet", det_image);
        cv::waitKey(0);
      }
   }

   LOG(INFO) << "Finished.";
   return 0;
 }
