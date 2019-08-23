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

// 滤出HandBoxes:
// 将与Head/Face交叠的HandBoxes全部删除
void filterHandBoxes(vector<BoundingBox<float> >& hand_boxes,
               const vector<BoundingBox<float> >& head_boxes,
               const vector<BoundingBox<float> >& face_boxes) {
  for (vector<BoundingBox<float> >::iterator it = hand_boxes.begin(); it!= hand_boxes.end();) {
    bool cov = false;
    // HEAD
    for (int i = 0; i < head_boxes.size(); ++i) {
      if (it->compute_coverage(head_boxes[i]) > 0.2) {
        cov = true;
        break;
      }
    }
    // FACE
    if (! cov) {
      for (int i = 0; i < face_boxes.size(); ++i) {
        if (it->compute_coverage(face_boxes[i]) > 0.1) {
          cov = true;
          break;
        }
      }
    }
    // 结论
    if (cov) {  // 删除之
      it = hand_boxes.erase(it);
    } else {
      ++it;
    }
  }
}

vector<bool> filterHandBoxesbyDistance(const vector<BoundingBox<float> >& head_boxes, const vector<BoundingBox<float> >& hand_boxes) {
  vector<bool> matrix(hand_boxes.size() * head_boxes.size(), false);
  vector<bool> chosen(hand_boxes.size(), false);  // 未匹配过
  // 距离判断
  for (int i = 0; i < hand_boxes.size(); ++i) {
    for (int j = 0; j < head_boxes.size(); ++j) {
      float dx = hand_boxes[i].get_center_x() - head_boxes[j].get_center_x();
      float dy = hand_boxes[i].get_center_y() - head_boxes[j].get_center_y();
      float dis = dx * dx + dy * dy;
      float size = head_boxes[j].compute_area();
      matrix[i * head_boxes.size() + j] = (dis < size * 5) ? true : false;
    }
  }
  // 择一判断
  for (int j = 0; j < head_boxes.size(); ++j) {
    // 选择最大的一个
    int max_id = -1;
    float max_area = 0;
    for (int i = 0; i < hand_boxes.size(); ++i) {  // 遍历所有hands
      if (chosen[i]) continue;                     // 已经匹配过,pass
      if (matrix[i * head_boxes.size() + j]) {     // 匹配
        const float area = hand_boxes[i].compute_area();
        if (area > max_area) {
          max_area = area;
          max_id = i;
        }
      }
    }
    // 匹配的hand面积必须要大于head面积的一半以上
    const float coeff = max_area / head_boxes[j].compute_area();
    if (max_id >= 0 && coeff > 0.2) {
      chosen[max_id] = true;  // 匹配成功!
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
   const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/Base_BD_PD_HD.prototxt";
   const std::string network_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
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
   // ################################ MAIN LOOP ################################
   // det_warpper
   caffe::DetWrapper<float> det_wrapper(network_proto,network_model,mode,gpu_id,proposals,max_dis_size);
   // HP
   const string hp_network = "/home/ethan/Models/Results/HPNet/CNN_Base_V0-I96-FL/Proto/test_copy.prototxt";
   // const string hp_network = "/home/ethan/Models/Results/HPNet/test.prototxt";
   const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_V0-I96-FL/Models/CNN_Base_V0-I96-FL-ROT-20K_iter_100000.caffemodel";
   // const string hp_model = "/home/ethan/Models/Results/HPNet/CNN_Base_I96-FLRTScaleAug1.5-2.0R8_V1Split1_1A_iter_100000.caffemodel";
   caffe::HPNetWrapper hp_wrapper(hp_network, hp_model);


   //  CAMERA
   if (use_camera) {
     cv::VideoCapture cap;
     if (!cap.open(0)) {
       LOG(FATAL) << "Failed to open webcam: " << 0;
     }
     cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
     int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
     cv::VideoWriter outputVideo; 
     outputVideo.open("outvideo_hp_CNN_Base_I96-FLRTScaleAug1.5-2.0R8_V1Split1_1A.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(cam_width,cam_height), true);
     cv::Mat cv_img;
     cap >> cv_img;
     int count = 0;
     CHECK(cv_img.data) << "Could not load image.";
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
       filterHandBoxes(hand_boxes, head_boxes, face_boxes);
       /**
        * 进一步滤除,每个Head最多允许一个HandBox
        * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
        * (2) 如果保留多个,则只选择其中面积最大的一个进行分析
        */
       vector<bool> active_hands = filterHandBoxesbyDistance(head_boxes, hand_boxes);
       // 绘制Head
       for (int i = 0; i < head_boxes.size(); ++i) {
         BoundingBox<float>& roi = head_boxes[i];
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
       // 绘制Face
      //  for (int i = 0; i < face_boxes.size(); ++i) {
      //    BoundingBox<float>& roi = face_boxes[i];
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
      //    cv::rectangle(image, point1, point2, cv::Scalar(0,255,0), 3);
      //  }
       // 绘制minihand网络的结果
       for (int i = 0; i < hand_boxes.size(); ++i) {
         BoundingBox<float>& roi = hand_boxes[i];
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
         const int r = active_hands[i] ? 0 : 255;
         const int g = active_hands[i] ? 255 : 0;
         /**
          * 获取手势
          */
         if (active_hands[i]) {
           float score;
           int label = hp_wrapper.hpmode(image, hand_boxes[i], &score);
           char tmp_str[256];
           snprintf(tmp_str, 256, "%d", label);
           cv::putText(image, tmp_str, cv::Point(x1, y1),
              cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);
         }
         cv::rectangle(image, point1, point2, cv::Scalar(0,g,r), 3);
       }
       outputVideo << image;
       cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
       cv::imshow( "RemoDet", image);
       cv::waitKey(1);
     }
   }

   LOG(INFO) << "Finished.";
   return 0;
 }
