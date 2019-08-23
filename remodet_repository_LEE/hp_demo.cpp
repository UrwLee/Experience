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
#include "caffe/hp/hp_net.hpp"

using namespace std;
using namespace caffe;
using std::string;
using std::vector;
namespace bfs = boost::filesystem;
// 获取FaceBoxes: 3
void getFaceBoxes(const vector<LabeledBBox<float> >& rois, vector<BoundingBox<float> >* face_boxes) {
  face_boxes->clear();
  for (int i = 0; i < rois.size(); ++i) {
    if (rois[i].cid != 3) continue;
    face_boxes->push_back(rois[i].bbox);
  }
}
// 获取HeadBoxes: 2
void getHeadBoxes(const vector<LabeledBBox<float> >& rois, vector<BoundingBox<float> >* head_boxes) {
  head_boxes->clear();
  for (int i = 0; i < rois.size(); ++i) {
    if (rois[i].cid != 2) continue;
    head_boxes->push_back(rois[i].bbox);
  }
}
// 获取HandBoxes: 1
void getHandBoxes(const vector<LabeledBBox<float> >& rois, vector<BoundingBox<float> >* hand_boxes) {
  hand_boxes->clear();
  for (int i = 0; i < rois.size(); ++i) {
    if (rois[i].cid != 1) continue;
    hand_boxes->push_back(rois[i].bbox);
  
  }
}

// 滤出HandBoxes
void filterHandBoxes(vector<BoundingBox<float> >& hand_boxes,
               const vector<BoundingBox<float> >& face_boxes,float cov_thre = 0.1) {
  for (vector<BoundingBox<float> >::iterator it = hand_boxes.begin(); it!= hand_boxes.end();) {
    bool cov = false;
    // FACE
    for (int i = 0; i < face_boxes.size(); ++i) {
      if (it->compute_coverage(face_boxes[i]) > cov_thre) {
        cov = true;
        break;
      }
    }
    if (cov) {  // 删除之
      it = hand_boxes.erase(it);
    } else {
      ++it;
    }
  }
}

// 滤出HandBoxes
// void filterHandHandBoxes(vector<BoundingBox<float> >& hand_boxes,float cov_thre = 0.1) {
//   for (vector<BoundingBox<float> >::iterator it = hand_boxes.begin(); it!= hand_boxes.end();) {
//     bool cov = false;
//     for (vector<BoundingBox<float> >::iterator it2 = it+1; it2!= hand_boxes.end();) {
//       if (it2->compute_coverage(*it) > cov_thre) {
//         cov = true;
//         break;
//       }
//     }
//     if (cov) {  // 删除之
//       it2 = hand_boxes.erase(it);
//     } else {
//       ++it;
//     }
//   }
// }

vector<bool> filterHandBoxesbyDistance(const vector<BoundingBox<float> >& face_boxes, const vector<BoundingBox<float> >& hand_boxes) {
  vector<bool> matrix(hand_boxes.size() * face_boxes.size(), false);
  vector<float> distance(hand_boxes.size() * face_boxes.size(), 1);
  vector<bool> chosen(hand_boxes.size(), false);  // 未匹配过
  // 距离判断
  for (int i = 0; i < hand_boxes.size(); ++i) {
    for (int j = 0; j < face_boxes.size(); ++j) {
      float dx = hand_boxes[i].get_center_x() - face_boxes[j].get_center_x();
      float dy = hand_boxes[i].get_center_y() - face_boxes[j].get_center_y();
      float dis = dx * dx + dy * dy;
      float size = face_boxes[j].compute_area();
      float ratio = hand_boxes[i].compute_area() / size;
      // bool flag = ratio > 0.33 && dis < size * 9;//default parameter
      bool flag = ratio > 0.1 && dis < size * 9;
      matrix[i * face_boxes.size() + j] = flag ? true : false;
      distance[i * face_boxes.size() + j] = dis;
    }
  }
  // 择一判断
  for (int j = 0; j < face_boxes.size(); ++j) {
    // 选择距离最近的一个
    int min_id = -1;
    float min_dis = 1e6;
    for (int i = 0; i < hand_boxes.size(); ++i) {  // 遍历所有hands
      if (chosen[i]) continue;                     // 已经匹配过,pass
      if (matrix[i * face_boxes.size() + j]) {     // 匹配
        const float dis = distance[i * face_boxes.size() + j];
        if (dis < min_dis) {
          min_dis = dis;
          min_id = i;
        }
      }
    }
    if (min_id >= 0) {
      chosen[min_id] = true;  // 匹配成功
    }
  }
  // 匹配成功的有效
  return chosen;
}

int main(int nargc, char** args) {
   // ################################ NETWORK ################################
   // network input
   int resized_width = 512;
   int resized_height = 288;
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/Base_BD_PD_HD.prototxt";
   // const std::string network_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
   const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand.prototxt";
     const std::string network_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel";
   // const std::string network_proto = "/home/zhangming/Models/FEMRelease/Release20180314_Merged_profo2.5/Base_BD_PD_HD.prototxt";
   // const std::string network_model = "/home/zhangming/Models/FEMRelease/Release20180314_Merged_profo2.5/R0314.caffemodel";
   // GPU
   int gpu_id = 0;
   bool mode = true;  // use GPU
   // features
   const std::string proposals = "det_out";
   // display Size
   int max_dis_size = 1280;
   // active labels
   vector<int> active_label;
  //  active_label.push_back(0);
   active_label.push_back(1);
   active_label.push_back(2);
  //  active_label.push_back(3);
   // ################################ DATA ####################################
   // CAMERA
   const bool use_camera = true; // 0
   const int cam_width = 1280;
   const int cam_height = 720;

   // VIDEO
   const bool use_video = true; // 0
   const string video = "/home/ethan/work/remodet_repository/test_video_hp_20180705.avi";
   // ################################ MAIN LOOP ################################
   // det_warpper
   caffe::DetWrapper<float> det_wrapper(network_proto,network_model,mode,gpu_id,proposals,max_dis_size);
   // HP
    // const string hp_network = "/home/ethan/Models/Results/HPNet/CNN_Base_V0-I96-FL/Proto/test_copy.prototxt";
    // const string hp_network = "/home/ethan/Models/Results/HPNet/test_VGG16.prototxt";
     // const string hp_network = "/home/ethan/ForZhangM/Release20180529/test_copy.prototxt";
   // const string hp_network = "/home/ethan/ForZhangM/Release20180506/Release20180506/R20180506_HP_V3.prototxt";
 // const string hp_model = "/home/ethan/ForZhangM/Release20180506/Release20180506/R20180506_HP_V3_A.caffemodel";
   // const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_V0-I96-FL/Models/CNN_Base_V0-I96-FL-ROT-20K_iter_100000.caffemodel";
   // const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_I96-FLRTScaleAug1.5-2.0R8_V1Split1_1A_iter_100000.caffemodel";
    // const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_handposeV1_Split1_VGG16_ratio9_softmax_v3_iter_100000.caffemodel";
     // const string hp_model = "/home/ethan/ForZhangM/Release20180529/CNN_Base_handposeVB04_model6-finetune-all_I96_margin3_R1-7_D30-30-90_iter200k_iter_85000.caffemodel";
     // const string hp_network = "/home/ethan/ForZhangM/Release20180529/test_model2_softmax.prototxt";
     // const string hp_model = "/home/ethan/ForZhangM/Release20180529/CNN_Base_handposeVB03_model2_I96_softmax_R1-7_D30-30-90_iter_85000.caffemodel";

     // const string hp_network = "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0.prototxt";
     // const string hp_model = "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0_A.caffemodel";
   
   // const string hp_network = "/home/ethan/Models/Results/HPNet/20180705/test_model2_softmax.prototxt";
     // const string hp_model = "/home/ethan/Models/Results/HPNet/20180705/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_Resize0.5-30-70_iter_190000.caffemodel";
// const string hp_model = "/home/ethan/Models/Results/HPNet/20180705/CNN_Base_handposeVB10-4_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_70000.caffemodel";
// const string hp_network = "/home/ethan/Models/Results/HPNet/20180705/test_model2L_softmax.prototxt";
// const string hp_model = "/home/ethan/Models/Results/HPNet/20180705/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_190000.caffemodel";

// const string hp_network = "/home/ethan/Models/Results/HPNet/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9.prototxt";
// const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_125000.caffemodel";

const string hp_network = "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0.prototxt";
const string hp_model = "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0_A.caffemodel";
   // const string hp_model = "/home/ethan/ForZhangM/Release20180506/Release20180506/R20180506_HP_V3_A.caffemodel";
   // const string hp_model = "/home/zhangming/Models/Results/HPNet/CNN_Base_V0-I96-FL-ROT-20K/Models/CNN_Base_V0-I96-FL-ROT-20K_iter_100000.caffemodel";
   caffe::HPNetWrapper hp_wrapper(hp_network, hp_model);
   //  CAMERA
   if (use_camera) {
     cv::VideoCapture cap(0);
     if (!cap.open(0)) {
       LOG(FATAL) << "Failed to open webcam: " << 0;
     }
     int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
     cv::VideoWriter outputVideo; 
     outputVideo.open("test_video_hp.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(cam_width,cam_height), true);
     cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
     cv::Mat cv_img;
     cap >> cv_img;
     int count = 0;
     CHECK(cv_img.data) << "Could not load image.";
     while (1) {
       ++count;
       cv::Mat image;
       cap >> image;
       caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
       outputVideo << image;
       // 获取hand_detector的结果
       vector<LabeledBBox<float> > rois;
       det_wrapper.get_rois(data_frame, &rois);
       vector<BoundingBox<float> > head_boxes;
       vector<BoundingBox<float> > face_boxes;
       vector<BoundingBox<float> > hand_boxes;
       getHeadBoxes(rois, &head_boxes);
       getFaceBoxes(rois, &face_boxes);
       getHandBoxes(rois, &hand_boxes);
       vector<BoundingBox<float> > hand_boxes_new;
       for(int i=0;i<hand_boxes.size();i++){
         bool keep = true;
         for(int j= i+1; j<hand_boxes.size();j++){
          float iou = hand_boxes[i].compute_iou(hand_boxes[j]);
          if (iou>0.5) keep = false;
         }
         if (keep) hand_boxes_new.push_back(hand_boxes[i]);
       }
       LOG(INFO)<<hand_boxes.size()<<" "<<hand_boxes_new.size();

       // filterHandHandBoxes(hand_boxes,0.8);
       filterHandBoxes(hand_boxes_new, face_boxes);
       
       /**
        * 进一步滤除,每个Face最多允许一个HandBox
        * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
        * (2) 如果保留多个,则只选择其中
        */
       vector<bool> active_hands = filterHandBoxesbyDistance(face_boxes, hand_boxes_new);
       // 绘制Head
      //  for (int i = 0; i < head_boxes.size(); ++i) {
      //    BoundingBox<float>& roi = head_boxes[i];
      //    float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
      //    float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
      //    float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
      //    float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
      //    x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
      //    y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
      //    x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
      //    y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
      //    x1 *= image.cols;
      //    x2 *= image.cols;
      //    y1 *= image.rows;
      //    y2 *= image.rows;
      //    const cv::Point point1(x1, y1);
      //    const cv::Point point2(x2, y2);
      //    cv::rectangle(image, point1, point2, cv::Scalar(255,0,0), 3);
      //  }
       // 绘制Face

       for (int i = 0; i < face_boxes.size(); ++i) {
         BoundingBox<float>& roi = face_boxes[i];
         float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
         float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
         float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
         float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
         x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
         y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
         x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
         y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
         x1 *= image.cols;
         x2 *= image.cols;
         y1 *= image.rows;
         y2 *= image.rows;
         const cv::Point point1(x1, y1);
         const cv::Point point2(x2, y2);
         cv::rectangle(image, point1, point2, cv::Scalar(255,0,0), 3);
       }
       // 绘制minihand网络的结果
       for (int i = 0; i < hand_boxes_new.size(); ++i) {
         BoundingBox<float>& roi = hand_boxes_new[i];
         float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
         float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
         float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
         float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
         x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
         y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
         x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
         y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
         x1 *= image.cols;
         x2 *= image.cols;
         y1 *= image.rows;
         y2 *= image.rows;
         const cv::Point point1(x1, y1);
         const cv::Point point2(x2, y2);
         
         bool flag_recog = false;
         /**
          * 获取手势
          */
         if (active_hands[i]) {
           float score;
           int label = hp_wrapper.hpmode(image, hand_boxes_new[i], &score);
           flag_recog = score>0.5;
           // if(flag_recog){
             char tmp_str[256];
             snprintf(tmp_str, 256, "%d/%.2f;%.4f", label, score,hand_boxes_new[i].compute_area());
             cv::putText(image, tmp_str, cv::Point(x1, y1),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);
            // }
           
         }
         const int r = active_hands[i] ? 0 : 255;
         const int g = active_hands[i] ? 255 : 0;
         cv::rectangle(image, point1, point2, cv::Scalar(0,g,r), 3);
         LOG(INFO)<<active_hands[i]<<" "<<r<<" "<<g;
       }
       cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
       cv::imshow( "RemoDet", image);
       cv::waitKey(1);
     }
   }
   if (use_video) {
     cv::VideoCapture cap;
     if (!cap.open(video)) {
       LOG(FATAL) << "Failed to open video: " << video;
     }
     int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
     cv::VideoWriter outputVideo; 
     outputVideo.open("test_out.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(cam_width,cam_height), true);
     cv::Mat cv_img;
     cap >> cv_img;
     CHECK(cv_img.data) << "Could not load image.";
     int count = 0;
     while (1) {
       ++count;
       cv::Mat image;
       cap >> image;
       caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
       // 获取hand_detector的结果
       vector<LabeledBBox<float> > rois;
       det_wrapper.get_rois(data_frame, &rois);
       vector<BoundingBox<float> > head_boxes;
       vector<BoundingBox<float> > face_boxes;
       vector<BoundingBox<float> > hand_boxes;
       getHeadBoxes(rois, &head_boxes);
       getFaceBoxes(rois, &face_boxes);
       getHandBoxes(rois, &hand_boxes);
       vector<BoundingBox<float> > hand_boxes_new;
       for(int i=0;i<hand_boxes.size();i++){
         bool keep = true;
         for(int j= i+1; j<hand_boxes.size();j++){
          float iou = hand_boxes[i].compute_iou(hand_boxes[j]);
          if (iou>0.5) keep = false;
         }
         if (keep) hand_boxes_new.push_back(hand_boxes[i]);
       }
       LOG(INFO)<<hand_boxes.size()<<" "<<hand_boxes_new.size();
       // filterHandHandBoxes(hand_boxes,0.8);
       filterHandBoxes(hand_boxes_new, face_boxes);
       
       /**
        * 进一步滤除,每个Face最多允许一个HandBox
        * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
        * (2) 如果保留多个,则只选择其中
        */
       vector<bool> active_hands = filterHandBoxesbyDistance(face_boxes, hand_boxes_new);
       //  for(int i=0;i<active_hands.size();i++){
       //  LOG(INFO)<<i<<" "<<active_hands[i];
       // }
       // 绘制Head
      //  for (int i = 0; i < head_boxes.size(); ++i) {
      //    BoundingBox<float>& roi = head_boxes[i];
      //    float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
      //    float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
      //    float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
      //    float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
      //    x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
      //    y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
      //    x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
      //    y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
      //    x1 *= image.cols;
      //    x2 *= image.cols;
      //    y1 *= image.rows;
      //    y2 *= image.rows;
      //    const cv::Point point1(x1, y1);
      //    const cv::Point point2(x2, y2);
      //    cv::rectangle(image, point1, point2, cv::Scalar(255,0,0), 3);
      //  }
       // 绘制Face
       for (int i = 0; i < face_boxes.size(); ++i) {
         BoundingBox<float>& roi = face_boxes[i];
         float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
         float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
         float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
         float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
         x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
         y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
         x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
         y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
         x1 *= image.cols;
         x2 *= image.cols;
         y1 *= image.rows;
         y2 *= image.rows;
         const cv::Point point1(x1, y1);
         const cv::Point point2(x2, y2);
         cv::rectangle(image, point1, point2, cv::Scalar(255,0,0), 3);
       }
       // 绘制minihand网络的结果
       for (int i = 0; i < hand_boxes_new.size(); ++i) {
         BoundingBox<float>& roi = hand_boxes_new[i];
         float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
         float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
         float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
         float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
         x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
         y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
         x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
         y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
         x1 *= image.cols;
         x2 *= image.cols;
         y1 *= image.rows;
         y2 *= image.rows;
         const cv::Point point1(x1, y1);
         const cv::Point point2(x2, y2);
         
         bool flag_recog = false;
         /**
          * 获取手势
          */
         if (active_hands[i]) {
           float score;
           int label = hp_wrapper.hpmode(image, hand_boxes_new[i], &score);
           flag_recog = score>0.5; 
           // if(flag_recog){
             char tmp_str[256];
             snprintf(tmp_str, 256, "%d/%.2f", label, score);
             cv::putText(image, tmp_str, cv::Point(x1, y1),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);
            // }
             // cv::rectangle(image, point1, point2, cv::Scalar(0,255,0), 3);
           
         }
         const int r = active_hands[i] ? 0 : 255;
         const int g = active_hands[i] ? 255 : 0;
          
     
         cv::rectangle(image, point1, point2, cv::Scalar(0,g,r), 3);

         // char tmp_str1[256];
         // snprintf(tmp_str1, 256, "%d",i);
         // cv::putText(image, tmp_str1, cv::Point(x1, y2),
         // cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255), 2);
       }
       outputVideo<<image;
       cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
       cv::imshow( "RemoDet", image);
       cv::waitKey(0);
     }
   }

   LOG(INFO) << "Finished.";
   return 0;
 }
