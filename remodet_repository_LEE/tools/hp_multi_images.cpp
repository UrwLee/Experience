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
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "caffe/det/detwrap.hpp"
#include "caffe/hp/hp_net_zj.hpp"
#include "caffe/smile/smile_net.hpp"
// #include "caffe/sa/sa_net.hpp"
// #include "caffe/sa/sa_base_net.hpp"

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

// 获取BodyBoxes: 0
void getBodyBoxes(const vector<LabeledBBox<float> >& rois, vector<BoundingBox<float> >* hand_boxes) {
  hand_boxes->clear();
  for (int i = 0; i < rois.size(); ++i) {
    if (rois[i].cid != 0) continue;
    hand_boxes->push_back(rois[i].bbox);
  }
}

// 滤出BodyBoxes
void filterBodyBoxes(vector<BoundingBox<float> >& body_boxes,
               const vector<BoundingBox<float> >& head_boxes) {
  for (vector<BoundingBox<float> >::iterator it = body_boxes.begin(); it!= body_boxes.end();) {
    bool cov = false;
    // HEAD
    int i = 0;
    float max1=0, max2=0, tmp1, tmp2;
    for (i = 0; i < head_boxes.size(); ++i) {
      // if (it->compute_coverage(head_boxes[i]) < 0.9) {
      //   cov = true;
      //   break;
      // }
      
      // filter bodys without head by calculate max area_overlap/area_head
      tmp1 = head_boxes[i].compute_coverage(*it);
      if (max1 < tmp1) {max1 = tmp1;}

      // filter bodys that are too close to camera by calculate max area_overlap/area_body
      tmp2 = it->compute_coverage(head_boxes[i]);
      if (max2 < tmp2) {max2 = tmp2;}
    }

    if (cov || max1<0.8 || max2>0.2 || head_boxes.size()==0) {  // 删除之
      it = body_boxes.erase(it);
      std::cout << "erase a body box." << std::endl;
    } else {
      ++it;
    }
  }
}

int filterBodyBoxes(BoundingBox<float>& body_box, const vector<BoundingBox<float> >& head_boxes) {
    bool cov = false;
    // HEAD
    int i = 0;
    float max1=0, max2=0, tmp1, tmp2;
    for (i = 0; i < head_boxes.size(); ++i) {
      // if (body_boxes.compute_coverage(head_boxes[i]) < 0.9) {
      //   cov = true;
      //   break;
      // }
      
      // filter bodys without head by calculate max area_overlap/area_head
      tmp1 = head_boxes[i].compute_coverage(body_box);
      if (max1 < tmp1) {max1 = tmp1;}

      // filter bodys that are too close to camera by calculate max area_overlap/area_body
      tmp2 = body_box.compute_coverage(head_boxes[i]);
      if (max2 < tmp2) {max2 = tmp2;}
    }

    if (cov || max1<0.8 || max2>0.1 || head_boxes.size()==0) {
      return 1;
    } else {
      return 0;
    }
}


// 滤出HandBoxes
void filterHandBoxes(vector<BoundingBox<float> >& hand_boxes,
               const vector<BoundingBox<float> >& face_boxes) {
  for (vector<BoundingBox<float> >::iterator it = hand_boxes.begin(); it!= hand_boxes.end();) {
    bool cov = false;
    // FACE
    for (int i = 0; i < face_boxes.size(); ++i) {
      if (it->compute_coverage(face_boxes[i]) > 0.1) {
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
      bool flag = ratio > 0.33 && dis < size * 19;
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

namespace fs = boost::filesystem;
int get_filenames(const std::string& dir, std::vector<std::string>& filenames)
{
    fs::path path(dir);
    if (!fs::exists(path))
    {
        return -1;
    }
  
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter!=end_iter; ++iter)
    {  
        if (fs::is_regular_file(iter->status()))
        {
            filenames.push_back(iter->path().string());
        }
  
        // if (fs::is_directory(iter->status()))  
        // {  
        //     get_filenames(iter->path().string(), filenames);  
        // } 
    }
  
    return filenames.size();
}

void test_head(cv::Mat& image, cv::Mat& image_draw, vector<BoundingBox<float> >& head_boxes, bool show_head){
  for (int i = 0; i < head_boxes.size(); ++i) {
    if (!show_head){
      break;
    }
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
  //  cv::rectangle(image_draw, point1, point2, cv::Scalar(255,0,0), 3); // blue
    cv::rectangle(image_draw, point1, point2, cv::Scalar(218,112,214), 3); // purple
  }
}

void test_face(caffe::SmileNetWrapper smile_wrapper, cv::Mat& image, cv::Mat& image_draw,
                vector<BoundingBox<float> >& face_boxes, bool show_face, bool& flag_smile, int& count2){
  for (int i = 0; i < face_boxes.size(); ++i) {
    if (!show_face){
      break;
    }
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
  //  cv::rectangle(image, point1, point2, cv::Scalar(255,0,0), 3);

    float score;
    int label = 0;//smile_wrapper.smilemode(image, face_boxes[i], &score);
    // char tmp_str[256];
    // snprintf(tmp_str, 256, "%d/%.2f", label, score);
    // cv::putText(image_draw, tmp_str, cv::Point(x1, y1),
    //     cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);
    
  //  const int r = label ? 0 : 255;
  //  const int b = label ? 255 : 0;
  //  cv::rectangle(image_draw, point1, point2, cv::Scalar(r,0,b), 3);
    if (label){
        cv::rectangle(image_draw, point1, point2, cv::Scalar(0,255,255), 3);
        flag_smile = true;
        count2 ++ ;
    } else{
        cv::rectangle(image_draw, point1, point2, cv::Scalar(255,0,0), 3);
    }
  }
}
void test_hand(caffe::HPNetWrapper hp_wrapper, 
                cv::Mat& image, cv::Mat& image_draw,
                vector<bool>& active_hands, 
                vector<BoundingBox<float> >& hand_boxes, 
                bool show_hand){
  int cc = 0;
  for (int i = 0; i < hand_boxes.size(); ++i) {
  if (!show_hand){
    break;
  }
  BoundingBox<float>& roi = hand_boxes[i];
  float kContextFactor = 1;
  float x1 = roi.get_center_x() - 1 * roi.get_width()  / 2 * kContextFactor;
  float y1 = roi.get_center_y() - 1 * roi.get_height() / 2 * kContextFactor;
  float x2 = roi.get_center_x() + 1 * roi.get_width()  / 2 * kContextFactor;
  float y2 = roi.get_center_y() + 1 * roi.get_height() / 2 * kContextFactor;
  x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
  y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
  x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
  y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
  x1 *= image.cols;
  x2 *= image.cols;
  y1 *= image.rows;
  y2 *= image.rows;
  hp_wrapper.getPoints(image, roi,x1,y1,x2,y2);
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
    snprintf(tmp_str, 256, "%d/%.2f", label, score);
    cv::putText(image_draw, tmp_str, cv::Point(x1, y1),
        cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);
    // cv::rectangle(image_draw, point1, point2, cv::Scalar(0,g,r), 3);

    // cv::Mat rsz_patch;
    // hp_wrapper.getInputPatch(image, rsz_patch, roi);
    // cv::namedWindow("hp_patch", cv::WINDOW_AUTOSIZE);
    // cv::imshow( "hp_patch", rsz_patch);

    cc++;
  }
  cv::rectangle(image_draw, point1, point2, cv::Scalar(0,g,r), 3);
}
}

void test_hand(vector<caffe::HPNetWrapper> hp_wrappers, 
                cv::Mat& image, cv::Mat& image_draw,
                vector<bool>& active_hands, 
                vector<BoundingBox<float> >& hand_boxes, 
                bool show_hand,
                vector<string> models){
  int num = hp_wrappers.size();
  int sqr = std::ceil(std::sqrt(num));
  int rows, cols, width, height;
  rows = (num%sqr) ? (num/sqr+1) : (num/sqr);
  cols = sqr;
  // width = floor((float)image_draw.cols / sqr);
  // height = floor((float)image_draw.rows / sqr);
  width = 640;
  height = 360;
  cv::Mat image_plot = cv::Mat::zeros(cv::Size(cols*width, rows*height), image_draw.type());

  // for (int k = sqr+1; k>=sqr+1; k--){
  //   if (num % k == 0){
  //     // rows = num / k;
  //     // cols = k;
  //     rows = sqr+1;
  //     cols = sqr+1;
  //     width = floor((float)image_draw.cols / cols);
  //     height = floor((float)image_draw.rows / rows);
  //     break;
  //   }
  // }

  // std::cout << rows << " " << cols << std::endl;
  // std::cout << width << " " << height << std::endl;

  for (int n=0; n<hp_wrappers.size(); ++n){
    caffe::HPNetWrapper hp_wrapper = hp_wrappers[n];
    cv::Mat image_temp = image_draw.clone();
    test_hand(hp_wrapper, image, image_temp, active_hands, hand_boxes, show_hand);

    string model = models[n];
    boost::filesystem::path filePath(model);
    string folder = filePath.stem().string();
    cv::putText(image_temp, folder, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 1);
    cv::resize(image_temp, image_temp, cv::Size(width, height), cv::INTER_LINEAR);
    int x = (n+1) % cols ? ((n+1) % cols):cols;
    int y = (n+1) % cols ? ((n+1) / cols + 1):((n+1) / cols);
    int pt_x = (x-1)*width;
    int pt_y = (y-1)*height;
    // std::cout << x << " " << y << " " << pt_x << " " << pt_y << std::endl;
    image_temp.copyTo(image_plot(cv::Rect(pt_x, pt_y, width, height)));
  }
  image_draw = image_plot;
  
}

// void test_body(caffe::SaBaseNetWrapper sa_base_wrapper, 
//                 caffe::SANetWrapper sa_wrapper, 
//                 cv::Mat& image, cv::Mat& image_draw, 
//                 vector<BoundingBox<float> >& body_boxes, 
//                 const vector<BoundingBox<float> >& head_boxes, 
//                 bool show_body){
//     Blob<float> map;
//     sa_base_wrapper.get_features(image, &map);
//     for (int i = 0; i < body_boxes.size(); ++i) {
//       if (!show_body){
//         break;
//       }
//       BoundingBox<float>& roi = body_boxes[i];
//       float x1 = roi.get_center_x() - 1 * roi.get_width() / 2;
//       float y1 = roi.get_center_y() - 1 * roi.get_height() / 2;
//       float x2 = roi.get_center_x() + 1 * roi.get_width() / 2;
//       float y2 = roi.get_center_y() + 1 * roi.get_height() / 2;
//       x1 = (x1 < 0) ? 0 : (x1 > 1 ? 1 : x1);
//       y1 = (y1 < 0) ? 0 : (y1 > 1 ? 1 : y1);
//       x2 = (x2 < 0) ? 0 : (x2 > 1 ? 1 : x2);
//       y2 = (y2 < 0) ? 0 : (y2 > 1 ? 1 : y2);
//       x1 *= image.cols;
//       x2 *= image.cols;
//       y1 *= image.rows;
//       y2 *= image.rows;
//       const cv::Point point1(x1, y1);
//       const cv::Point point2(x2, y2);

//       // 
//       Blob<float> body_feature;
//       int w = 24;
//       int h = 24;
//       float scale = 1.1;
//       get_body_features(&body_feature, &map, roi, h, w, scale);

//       float score;
//       int label = sa_wrapper.samode(&body_feature, &score);
//       if (filterBodyBoxes(body_boxes[i], head_boxes) && label==1){
//         label=-1;
//       }
//       char tmp_str[256];
//       // snprintf(tmp_str, 256, "%d/%.2f", label, score);
//       // cv::putText(image_draw, tmp_str, cv::Point(x1, y2),
//       //       cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255,0,0), 5);
      
//       if (label==-1){
//         snprintf(tmp_str, 256, "%d/%.2f", label, score);
//         cv::putText(image_draw, tmp_str, cv::Point(x1, y2),
//             cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(71,99,255), 5);
//         cv::rectangle(image_draw, point1, point2, cv::Scalar(71,99,255), 3);
//       }else if(label==1){
//         snprintf(tmp_str, 256, "%d/%.2f", label, score);
//         cv::putText(image_draw, tmp_str, cv::Point(x1, y2),
//             cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,255,255), 5);
//         cv::rectangle(image_draw, point1, point2, cv::Scalar(0,255,255), 3);
//       } else{
//         snprintf(tmp_str, 256, "%d/%.2f", label, score);
//         cv::putText(image_draw, tmp_str, cv::Point(x1, y2),
//             cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255,0,0), 5);
//         cv::rectangle(image_draw, point1, point2, cv::Scalar(255,0,0), 3);
//       }     
//     }
// }
        

// --------------------------------------------------------------------------------- //

int main(int nargc, char** args) {
   // ################################ NETWORK ################################
   // network input
   int resized_width = 512;
   int resized_height = 288;
  //  const std::string network_proto = "/home/jun/work/Protos/Base_BD_PD_HD.prototxt";
  //  const std::string network_model = "/home/jun/work/Protos/ResPoseDetTrackRelease_merge.caffemodel";

  // const std::string network_proto = "/home/jun/work/Protos/TrunkNet_PDNet_merge.prototxt";
  // const std::string network_model = "/home/jun/work/Protos/TrunkNet_PDNet_merge.caffemodel";

   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand.prototxt";
   // const std::string network_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel";
 const std::string network_proto ="/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand.prototxt";
 const std::string network_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHandDist0.5_V1.caffemodel";



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
   
  bool show_head = false;
  bool show_face = true;
  bool show_hand = true;
  bool show_body = false;

  bool use_imgs  = true;
  bool use_video = false;
  bool use_camera = true;

  bool write = true;

  // images
  // string img_root = "/home/jun/work/test_images/0522";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/test_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_iter_150000";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/test_CNN_Base_handposeVB06_Inception-model1_softmax_R0-7_D30-45-90_Br30_iter_30000";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_S1.8-2.0_iter_140000";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/test_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_iter_150000";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/test_CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_30000";
  // string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/testt_CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_130000";
  string img_root = "/home/ethan/Models/Results/HPNet/20180718/videos/test_CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_45000";


  // CAMERA
  const int cam_width = 1280;
  const int cam_height = 720;
  // VIDEO
  const string video = "/home/jun/work/554.Flv";

  // ################################ MAIN LOOP ################################
  //************************ det_warpper ********************/
  caffe::DetWrapper<float> det_wrapper(network_proto,network_model,mode,gpu_id,proposals,max_dis_size);
   
  /************************ hp ********************/
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_focalloss.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_margin2.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_margin3.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_margin4.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_margin3_ARC.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model3_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model3_focalloss.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model4_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model5_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model5_focalloss.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model5_margin3.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model6_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model6_margin3.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model7_softmax.prototxt";
  //  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model7_margin3.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_inception1_softmax.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_inception2_softmax.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ResModel1_softmax.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ResModel2_softmax.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB08_model2_I96Mins1.6Maxs1.8_softmax_R0-7_D25_Br16_12layers/Proto/test_copy.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_1.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_2.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_3.prototxt";

  const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_margin2.prototxt";
  // const string hp_network = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_margin4.prototxt";

  // const string hp_network = "/home/jun/R20180506_HP_V3.prototxt";
  // const string hp_model = "/home/jun/R20180506_HP_V3_B.caffemodel";

  // ------------------------

  //// VB 01
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model2_I96_softmax/Models/CNN_Base_handposeVB01_model2_I96_softmax_iter_100000.caffemodel";
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model2_I96_focalloss/Models/CNN_Base_handposeVB01_model2_I96_focalloss_iter_100000.caffemodel";
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model2_I96_softmax_R1-7_D45-90/Models/CNN_Base_handposeVB01_model2_I96_softmax_R1-7_D45-90_iter_80000.caffemodel";
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model2_I96_softmax_R1-7_D45-90_S1.5/Models/CNN_Base_handposeVB01_model2_I96_softmax_R1-7_D45-90_S1.5_iter_80000.caffemodel";

  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model3_I96_softmax/Models/CNN_Base_handposeVB01_model3_I96_softmax_iter_70000.caffemodel";
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB01_model3_I96_focalloss/Models/CNN_Base_handposeVB01_model3_I96_focalloss_iter_100000.caffemodel";

  //// VB 02
  //  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB02_model2_I96_softmax_R1-7_D45-90/Models/CNN_Base_handposeVB02_model2_I96_softmax_R1-7_D45-90_iter_30000.caffemodel";

  //// VB 03
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model2_I96_softmax_R1-7_D45-90/Models/CNN_Base_handposeVB03_model2_I96_softmax_R1-7_D45-90_iter_105000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model2_I96_focalloss_R1-7_D45-90/Models/CNN_Base_handposeVB03_model2_I96_focalloss_R1-7_D45-90_iter_90000.caffemodel";
  
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model2_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model2_I96_softmax_R1-7_D30-30-90_iter_100000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model4-finetune-all_I96_softmax_R1-7_D25-45-90/Models/CNN_Base_handposeVB03_model4-finetune-all_I96_softmax_R1-7_D25-45-90_iter_135000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model4-finetune-later_I96_softmax_R1-7_D25-45-90/Models/CNN_Base_handposeVB03_model4-finetune-later_I96_softmax_R1-7_D25-45-90_iter_135000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-all_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-all_I96_softmax_R1-7_D30-30-90_iter_140000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-later_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-later_I96_softmax_R1-7_D30-30-90_iter_140000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-all_I96_margin3_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-all_I96_margin3_R1-7_D30-30-90_iter_145000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-later_I96_margin3_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-later_I96_margin3_R1-7_D30-30-90_iter_145000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-all_I96_focalloss_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-all_I96_focalloss_R1-7_D30-30-90_iter_120000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB03_model5-finetune-later_I96_focalloss_R1-7_D30-30-90/Models/CNN_Base_handposeVB03_model5-finetune-later_I96_focalloss_R1-7_D30-30-90_iter_120000.caffemodel";

  //// VB 04
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model2_I96_softmax_R1-7_D25-45-90/Models/CNN_Base_handposeVB04_model2_I96_softmax_R1-7_D25-45-90_iter_125000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model2_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB04_model2_I96_softmax_R1-7_D30-30-90_iter_110000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model6-finetune-all_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB04_model6-finetune-all_I96_softmax_R1-7_D30-30-90_iter_145000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model6-finetune-later_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB04_model6-finetune-later_I96_softmax_R1-7_D30-30-90_iter_145000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model6-finetune-all_I96_margin3_R1-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB04_model6-finetune-all_I96_margin3_R1-7_D30-30-90_iter200k_iter_190000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB04_model6-finetune-later_I96_margin3_R1-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB04_model6-finetune-later_I96_margin3_R1-7_D30-30-90_iter200k_iter_200000.caffemodel";

  //// VB 05
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model2_I96_softmax_R1-7_D30-30-90/Models/CNN_Base_handposeVB05_model2_I96_softmax_R1-7_D30-30-90_iter_125000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model2_I96_softmax_R0-7_D30-30-90/Models/CNN_Base_handposeVB05_model2_I96_softmax_R0-7_D30-30-90_iter_110000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model6-finetune-all_I96_softmax_R1-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model6-finetune-all_I96_softmax_R1-7_D30-30-90_iter200k_iter_140000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model6-finetune-all_I96_margin3_R1-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model6-finetune-all_I96_margin3_R1-7_D30-30-90_iter200k_iter_140000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-all_I96_softmax_R0-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-all_I96_softmax_R0-7_D30-30-90_iter200k_iter_145000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-later_I96_softmax_R0-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-later_I96_softmax_R0-7_D30-30-90_iter200k_iter_145000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-all_I96_margin3_R0-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-all_I96_margin3_R0-7_D30-30-90_iter200k_iter_145000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-later_I96_margin3_R0-7_D30-30-90_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-later_I96_margin3_R0-7_D30-30-90_iter200k_iter_145000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-all_I96_margin3_R0-7_D30-30-90_Br30_iter200k/Models/CNN_Base_handposeVB05_model7-finetune-all_I96_margin3_R0-7_D30-30-90_Br30_iter200k_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB05_model7-finetune-all_I96_margin4_R0-7_D30-30-60_iter200k_Br16/Models/CNN_Base_handposeVB05_model7-finetune-all_I96_margin4_R0-7_D30-30-60_iter200k_Br16_iter_185000.caffemodel";

  //// VB 06
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30/Models/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_iter_150000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_S1.6-1.8/Models/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_S1.6-1.8_iter_140000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_S1.8-2.0/Models/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-90_Br30_S1.8-2.0_iter_140000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D25_Br30/Models/CNN_Base_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D25_Br30_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-60_Br30/Models/CNN_Base_CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-60_Br30_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model2_I96_margin3_R0-7_D30-30-60_Br16/Models/CNN_Base_handposeVB06_model2_I96_margin3_R0-7_D30-30-60_Br16_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-60_Br16_ROC/Models/CNN_Base_handposeVB06_model2_I96_softmax_R0-7_D30-30-60_Br16_ROC_iter_200000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k/Models/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k_iter_150000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k_S1.8-2.0/Models/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_Br30_iter200k_S1.8-2.0_iter_140000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_iter200k_Br30_S1.6-1.8/Models/CNN_Base_handposeVB06_model7-finetune-all_I96_softmax_R0-7_D30-30-90_iter200k_Br30_S1.6-1.8_iter_200000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_Inception-model1_softmax_R0-7_D30-45-90_Br30/Models/CNN_Base_handposeVB06_Inception-model1_softmax_R0-7_D30-45-90_Br30_iter_65000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB06_Inception-model2_softmax_R0-7_D30-45-90_Br30/Models/CNN_Base_handposeVB06_Inception-model2_softmax_R0-7_D30-45-90_Br30_iter_65000.caffemodel";

  //// VB 07
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB07_model7-finetune-all_I96_softmax_R0-7_D25_iter200k_Br16/Models/CNN_Base_handposeVB07_model7-finetune-all_I96_softmax_R0-7_D25_iter200k_Br16_iter_195000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16/Models/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16_blur/Models/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16_blur_iter_200000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16_ROC/Models/CNN_Base_handposeVB07_model2_I96_softmax_R0-7_D25_Br16_ROC_iter_200000.caffemodel";

  //// VB 08
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB08_model2_I96_softmax_R0-7_D25_Br16/Models/CNN_Base_handposeVB08_model2_I96_softmax_R0-7_D25_Br16_iter_195000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB08_model2_I96Mins1.6Maxs1.8_softmax_R0-7_D25_Br16_12layers/Models/CNN_Base_handposeVB08_model2_I96Mins1.6Maxs1.8_softmax_R0-7_D25_Br16_12layers_iter_200000.caffemodel";

  //// VB 09
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_Res_model1_softmax_Br16_R0-7_D25/Models/CNN_Base_handposeVB09_Res_model1_softmax_Br16_R0-7_D25_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_ResModel2_softmax_Br16_R0-7_D25/Models/CNN_Base_handposeVB09_ResModel2_softmax_Br16_R0-7_D25_iter_55000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_ResModel2_softmax_Br16_R0_D180-180-180/Models/CNN_Base_handposeVB09_ResModel2_softmax_Br16_R0_D180-180-180_iter_55000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.0/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.0_iter_135000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.3/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.3_iter_135000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.5/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.5_iter_135000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.7_iter_135000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.6-1.8_lr-3_Drop0.9_iter_135000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.3/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.3_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.5/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.5_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.7/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.7_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25_Br16_S1.8-2.0_lr-3_Drop0.9_iter_130000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.2-1.6_lr-3_Drop0.7/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.2-1.6_lr-3_Drop0.7_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.2-1.6_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.2-1.6_lr-3_Drop0.9_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.8-2.2_lr-3_Drop0.7/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.8-2.2_lr-3_Drop0.7_iter_80000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.8-2.2_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.8-2.2_lr-3_Drop0.9_iter_80000.caffemodel";

  //// VB 10
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_130000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_margin2_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0/Models/CNN_Base_handposeVB10_model2_I96_margin2_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_margin4_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0/Models/CNN_Base_handposeVB10_model2_I96_margin4_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0_iter_130000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2L_I96_margin2_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0/Models/CNN_Base_handposeVB10_model2L_I96_margin2_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0_iter_130000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2L_I96_margin4_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0/Models/CNN_Base_handposeVB10_model2L_I96_margin4_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0_iter_130000.caffemodel";

  const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_130000.caffemodel";
  
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_110000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R1-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R1-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_110000.caffemodel";

  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20/Models/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20_iter_55000.caffemodel";
  // const string hp_model = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-4_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-4_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_185000.caffemodel";


  vector<caffe::HPNetWrapper> hp_wrappers;
  string hp_networks =  
  "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0.prototxt;"\
                     
                        "/home/ethan/Models/Results/HPNet/20180718/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_model2-6cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_model2_softmax.prototxt;"\
                        "/home/ethan/ForZhangM/Release20180718_HP/Release20180718_HP.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180731/test_model2-7cls_softmax.prototxt";

  string hp_models = 
   // "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02_model2-7cls_I96_marginArc0.84-0.5-0.2_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD02_model2-7cls_I96_marginArc0.84-0.5-0.2_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9_iter_50000.caffemodel;"\,
                      "/home/ethan/ForZhangM/Release20180606/R20180606_HP_V0_A.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3_iter_170000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/ForZhangM/Release20180718_HP/Release20180718_HP.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180731/CNN_Base_handposeVD02-3_model2-7cls_I96_softmax_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9_distProb0.5_he0.5-m1_iter_120000.caffemodel";

/*
  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_marginArc0.84-0.5-0.2.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-9cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-9cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02_model2-7cls_I96_marginArc0.84-0.5-0.2_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD02_model2-7cls_I96_marginArc0.84-0.5-0.2_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9_iter_50000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD03_model2-9cls_I96_softmax_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD03_model2-9cls_I96_softmax_R0-6-custom4_Br16_S1.6-1.8_lr-3_Drop0.9_iter_100000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD03_model2-9cls_I96_softmax_R0-6-custom5_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve4/Models/CNN_Base_handposeVD03_model2-9cls_I96_softmax_R0-6-custom5_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve4_iter_100000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";
*/
/*
  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02-2_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3/Models/CNN_Base_handposeVD02-2_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3_iter_170000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3/Models/CNN_Base_handposeVD02-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3_iter_170000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";


  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3/Models/CNN_Base_handposeVD02_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3_iter_115000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD02_model2-7cls_I96_softmax_R0-6-custom3_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3/Models/CNN_Base_handposeVD02_model2-7cls_I96_softmax_R0-6-custom3_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3_iter_115000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";
*/
/*
  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_iter_150000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.75_cp0.75_sp0.75_roc0.75/Models/CNN_Base_handposeVD01-3_model2-7cls_I96_softmax_R0-6-custom2_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.75_cp0.75_sp0.75_roc0.75_iter_150000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";


  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2-7cls_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-3_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-3_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_155000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01-2_model2-7cls_I96_softmax_R0-4_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5/Models/CNN_Base_handposeVD01-2_model2-7cls_I96_softmax_R0-4_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_155000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01-2_model2-7cls_I96_softmax_R0-6-custom1_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5/Models/CNN_Base_handposeVD01-2_model2-7cls_I96_softmax_R0-6-custom1_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_155000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";
*/
/*
  string hp_networks =  "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                        "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models =  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D35-45-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5/Models/CNN_Base_handposeVD01_model2_I96_softmax_R0-7_D35-45-60_Br16_S1.6-1.8_lr-3_Drop0.9_bp0.9_BlurAve5_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                      "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax_VC.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVC01_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVC01_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20/Models/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20_iter_125000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve5/Models/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve5_iter_60000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_Resize0.0/Models/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_Resize0.0_iter_200000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-4_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-4_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_185000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-5_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-3_model2L_I96_softmax_R0-5_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_175000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB09_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_200000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-5_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-5_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_85000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-6_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-6_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_85000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-7_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-7_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_85000.caffemodel";
*/
                    //  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve5/Models/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve5_iter_15000.caffemodel;"
                    //  "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20/Models/CNN_Base_handposeVB10-3_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_BlurAve3-9-20_iter_80000.caffemodel";

/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_model2L_softmax.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_2.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_2.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_130000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_Resize0.0/Models/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_Resize0.0_iter_130000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10-2_model2_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_130000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7/Models/CNN_Base_handposeVB10_model2L_I96_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.7_iter_130000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_130000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0/Models/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0_iter_70000.caffemodel";
*/
/*
  string hp_networks = "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_1.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_2.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_3.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_1.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_2.prototxt;"\
                       "/home/ethan/Models/Results/HPNet/20180718/test_proto/test_ml01_3.prototxt;";

  string hp_models = "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_95000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_95000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9/Models/CNN_Base_handposeVB10_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_iter_95000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0/Models/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0_iter_95000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0/Models/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0_iter_95000.caffemodel;"\
                     "/home/ethan/Models/Results/HPNet/20180718/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0/Models/CNN_Base_handposeVB10-2_MultiLoss01_softmax_R0-7_D25-35-60_Br16_S1.6-1.8_lr-3_Drop0.9_L0.2-0.2-1.0_iter_95000.caffemodel";
*/
  // float a[7] = {1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7};
  float a[7] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  vector<float> scale(a, a+6);

  vector<string> networks, models;
  boost::split(networks, hp_networks, boost::is_any_of(";"), boost::token_compress_on);
  boost::split(models, hp_models, boost::is_any_of(";"), boost::token_compress_on);
  
  for (int i=0; i<models.size();i++){
    string network = networks[i];
    string model = models[i];
    caffe::HPNetWrapper wrapper(network, model, scale[i]);
    hp_wrappers.push_back(wrapper);
  }

  /*------------------------------- Smile Face ----------------------------*/
  //  plain
  //    const string sf_network = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1-shallow/Proto/test_copy.prototxt";
  //    const string sf_model = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1-shallow/Models/PlainCNN_Base_V5-I96-1-1-shallow_iter_150000.caffemodel";

   const string sf_network = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1/Proto/test_copy.prototxt";
  //  const string sf_network = "/home/jun/work/server_smile/PlainCNN_Base_sf-V1_Input96_1e-2_centerloss/Proto/test_copy.prototxt";
  //  const string sf_network = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_focalloss-v2/Proto/test_copy.prototxt";
  //  const string sf_network = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_margin_nodrop/Proto/test_copy.prototxt";

  //  train_1-1
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1/Models/PlainCNN_Base_V5-I96-1-1_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_centerloss/Models/PlainCNN_Base_V5-I96-1-1_centerloss_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_margin/Models/PlainCNN_Base_V5-I96-1-1_margin_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_focalloss-v2/Models/PlainCNN_Base_V5-I96-1-1_focalloss-v2_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_focalloss-v2_1e-2/Models/PlainCNN_Base_V5-I96-1-1_focalloss-v2_1e-2_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-1-1_focalloss-v2_1e-2_aug2/Models/PlainCNN_Base_V5-I96-1-1_focalloss-v2_1e-2_aug2_iter_150000.caffemodel";

  //  original
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_centerloss/Models/PlainCNN_Base_V5-I96-original_centerloss_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_softmax/Models/PlainCNN_Base_V5-I96-original_softmax_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_margin_nodrop/Models/PlainCNN_Base_V5-I96-original_margin_nodrop_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_aug2/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_aug2_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama1/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama1_iter_200000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5_iter_200000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama8/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama8_iter_200000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5_distort1/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5_distort1_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5_drift0.10/Models/PlainCNN_Base_V5-I96-original_focalloss-v2_1e-2_alpha0.25_gama5_drift0.10_iter_100000.caffemodel";

  const string sf_model = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_softmax/Models/PlainCNN_Base_sf-V2_Input96_1e-2_softmax_iter_150000.caffemodel";
  // const string sf_model = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_centerloss/Models/PlainCNN_Base_sf-V2_Input96_1e-2_centerloss_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_focalloss_alpha0.25_gama2/Models/PlainCNN_Base_sf-V2_Input96_1e-2_focalloss_alpha0.25_gama2_iter_150000.caffemodel";

  // rotate45
  // const string sf_model = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_softmax_rotate45/Models/PlainCNN_Base_sf-V2_Input96_1e-2_softmax_rotate45_iter_150000.caffemodel";
  // const string sf_model = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_centerloss_rotate45/Models/PlainCNN_Base_sf-V2_Input96_1e-2_centerloss_rotate45_iter_150000.caffemodel";
  //  const string sf_model   = "/home/jun/work/server_smile/PlainCNN_Base_sf-V2_Input96_1e-2_focalloss_alpha0.25_gama2_rotate45/Models/PlainCNN_Base_sf-V2_Input96_1e-2_focalloss_alpha0.25_gama2_rotate45_iter_150000.caffemodel";

   // res
//    const string sf_network = "/home/jun/work/server_smile/ResCNN_Base_V5-I96-1-1/Proto/test_copy.prototxt";
  //  const string sf_model = "/home/jun/work/server_smile/ResCNN_Base_V5-I96-1-1/Models/ResCNN_Base_V5-I96-1-1_iter_150000.caffemodel";

   /************************ SA ********************/

  ////////// conv5_5
  //  const std::string sa_base_network = "/home/jun/Models/Release/Release20180425/DarkNet_TrunkBD_PDHeadHand_NonCat_1A_merge_BaseConv5_5_copy.prototxt";
  //  const std::string sa_base_model   = "/home/jun/Models/Release/Release20180425/DarkNet_TrunkBD_PDHeadHand_NonCat_Exp2.5_1A_iter31e4_trunc1.0weightdecay5e-2_iter_34000_merge.caffemodel";

  //  const std::string sa_network = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv5_5/Proto/test_copy.prototxt";
  //  const std::string sa_model   = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv5_5/Models/CNN_Base_V0_I384_R24_1v1_conv5_5_iter_150000.caffemodel";
  //  const std::string sa_model   = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv5_5_aug2/Models/CNN_Base_V0_I384_R24_1v1_conv5_5_aug2_iter_150000.caffemodel";

  //  const std::string sa_network = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv5_5_deconvBN/Proto/test_copy.prototxt";
  //  const std::string sa_model   = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv5_5_deconvBN/Models/CNN_Base_V0_I384_R24_1v1_conv5_5_deconvBN_iter_70000.caffemodel";

  ////////// conv4_5
  //  const std::string sa_base_network = "/home/jun/Models/Release/Release20180425/DarkNet_TrunkBD_PDHeadHand_NonCat_1A_merge_BaseConv4_5_copy.prototxt";
  //  const std::string sa_base_model   = "/home/jun/Models/Release/Release20180425/DarkNet_TrunkBD_PDHeadHand_NonCat_Exp2.5_1A_iter31e4_trunc1.0weightdecay5e-2_iter_34000_merge.caffemodel";

  //  const std::string sa_network = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv4_5_aug2/Proto/test_copy.prototxt";
  //  const std::string sa_model   = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_conv4_5_aug2/Models/CNN_Base_V0_I384_R24_1v1_conv4_5_aug2_iter_150000.caffemodel";

  //  const std::string sa_network = "/home/jun/work/server_sa/CNN_Base_Release20180425_V0_I384_R24_all_conv4_5_focalloss_alpha0.25_gamma2/Proto/test_copy.prototxt";
  //  const std::string sa_model   = "/home/jun/work/server_sa/CNN_Base_Release20180425_V0_I384_R24_all_conv4_5_focalloss_alpha0.25_gamma1/Models/CNN_Base_Release20180425_V0_I384_R24_all_conv4_5_focalloss_alpha0.25_gamma1_iter_90000.caffemodel";

  ///////// convf
  const string sa_base_network = "/home/jun/Models/Release/Release20180415/TrunkNet_merge_convf_copy.prototxt";
  const string sa_base_model   = "/home/jun/Models/Release/Release20180415/TrunkNet_merge.caffemodel";

  const string sa_network = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_convf_aug2/Proto/test_copy.prototxt";
  //  const string sa_model   = "/home/jun/work/server_sa/CNN_Base_V0_I384_R24_1v1_convf_aug2/Models/CNN_Base_V0_I384_R24_1v1_convf_aug2_iter_150000.caffemodel";
  const string sa_model = "/home/jun/work/server_sa/CNN_Base_Release20180415_V0_I384_R24_all_convf_softmax/Models/CNN_Base_Release20180415_V0_I384_R24_all_convf_softmax_iter_140000.caffemodel";

  // const string sa_network = "/home/jun/work/server_sa/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.25_gamma2/Proto/test_copy.prototxt";
  // const string sa_model   = "/home/jun/work/server_sa/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.25_gamma2/Models/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.25_gamma2_iter_50000.caffemodel";
  // const string sa_model   = "/home/jun/work/server_sa/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.5_gamma1/Models/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.5_gamma1_iter_80000.caffemodel";
  // const string sa_model   = "/home/jun/work/server_sa/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.75_gamma1/Models/CNN_Base_Release20180415_V0_I384_R24_all_convf_focalloss_alpha0.75_gamma1_iter_80000.caffemodel";


   /*** path for imwrite ***/
  //  bool write = true;
  boost::filesystem::path filePath(hp_model);
  string folder = filePath.stem().string();
  string image_folder = "/home/ethan/videos/" + folder;
  if (write && boost::filesystem::exists(image_folder)){
    boost::filesystem::remove_all(image_folder);
  }
  boost::filesystem::create_directories(image_folder);

  // caffe::HPNetWrapper     hp_wrapper(hp_network, hp_model);
  // caffe::SmileNetWrapper  smile_wrapper(sf_network, sf_model);

  // caffe::SaBaseNetWrapper sa_base_wrapper(sa_base_network, sa_base_model, "conv4_5");
  // caffe::SANetWrapper     sa_wrapper(sa_network, sa_model);

  int count2 = 0;
  bool flag_smile=false;

  //  CAMERA
  if (use_camera) {
    cv::VideoCapture cap(0);
    if (!cap.open(0)) {
      LOG(FATAL) << "Failed to open webcam: " << 0;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
    cv::Mat cv_img;
    cap >> cv_img;
    int count = 0;
    CHECK(cv_img.data) << "Could not load image.";
    cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
    while (1) {
      ++count;
      cv::Mat image, image_draw;  // image for test, image_draw for imshow
      cap >> image;
      if (!image.data){break;}
      cv::resize(image, image, cv::Size(cam_width, cam_height), cv::INTER_LINEAR);
      cv::flip(image, image, 1);
      image_draw = image.clone();
      caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
      // 获取hand_detector的结果
      vector<LabeledBBox<float> > rois;
      det_wrapper.get_rois(data_frame, &rois);
      vector<BoundingBox<float> > head_boxes;
      vector<BoundingBox<float> > face_boxes;
      vector<BoundingBox<float> > hand_boxes;
      vector<BoundingBox<float> > body_boxes;
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
      hand_boxes = hand_boxes_new;
      getBodyBoxes(rois, &body_boxes);
      filterHandBoxes(hand_boxes, face_boxes);
      /**
      * 进一步滤除,每个Face最多允许一个HandBox
      * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
      * (2) 如果保留多个,则只选择其中
      */
      vector<bool> active_hands = filterHandBoxesbyDistance(face_boxes, hand_boxes);
      // 绘制Head
      test_head(image, image_draw, head_boxes, show_head);

      // 绘制Face
      // test_face(smile_wrapper, image, image_draw, face_boxes, show_face, flag_smile, count2);

      /*** 绘制 Body ***/
      // test_body(sa_base_wrapper, sa_wrapper, image, image_draw, body_boxes, head_boxes, show_body);

      // 绘制minihand网络的结果
      // test_hand(hp_wrapper, image, image_draw, active_hands, hand_boxes, show_hand);
      test_hand(hp_wrappers, image, image_draw, active_hands, hand_boxes, show_hand, models);

      /*** imwrite ***/ 
      if (write){
      flag_smile = false;
      stringstream ss;
      ss << image_folder << "/" << setw(7) << setfill('0') << count << ".jpg";
      cv::imwrite(ss.str(), image);
      }

    //  cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
      //cv::flip(image, image, 1);
      cv::imshow( "RemoDet", image_draw);
      int key = cv::waitKey(1);
      if (key=='q'){
        cv::destroyAllWindows();
        break;
      }
    }
  }else if (use_video) {
    cv::VideoCapture cap(video);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
    cv::Mat cv_img;
    cap >> cv_img;
    int count = 0;
    CHECK(cv_img.data) << "Could not load image.";
    cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
    while (1) {
      ++count;
      cv::Mat image, image_draw;
      cap >> image;
      if (!image.data){break;}
      cv::resize(image, image, cv::Size(cam_width, cam_height), cv::INTER_LINEAR);
      image_draw = image.clone();
      caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
      // 获取hand_detector的结果
      vector<LabeledBBox<float> > rois;
      det_wrapper.get_rois(data_frame, &rois);
      vector<BoundingBox<float> > head_boxes;
      vector<BoundingBox<float> > face_boxes;
      vector<BoundingBox<float> > hand_boxes;
      vector<BoundingBox<float> > body_boxes;
      getHeadBoxes(rois, &head_boxes);
      getFaceBoxes(rois, &face_boxes);
      getHandBoxes(rois, &hand_boxes);
      getBodyBoxes(rois, &body_boxes);
      filterHandBoxes(hand_boxes, face_boxes);
      /**
      * 进一步滤除,每个Face最多允许一个HandBox
      * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
      * (2) 如果保留多个,则只选择其中
      */
      vector<bool> active_hands = filterHandBoxesbyDistance(face_boxes, hand_boxes);
      // 绘制Head
      test_head(image, image_draw, head_boxes, show_head);

      // 绘制Face
      // test_face(smile_wrapper, image, image_draw, face_boxes, show_face, flag_smile, count2);

      /*** 绘制 Body ***/
      // test_body(sa_base_wrapper, sa_wrapper, image, image_draw, body_boxes, head_boxes, show_body);

      // 绘制minihand网络的结果
      test_hand(hp_wrappers, image, image_draw, active_hands, hand_boxes, show_hand, models);
      // test_hand(hp_wrapper, image, image_draw, active_hands, hand_boxes, show_hand);

      /*** imwrite ***/ 
      if (write){
      flag_smile = false;
      stringstream ss;
      ss << image_folder << "/" << setw(7) << setfill('0') << count << ".jpg";
      // cv::imwrite(ss.str(), image_draw);
      }

    //  cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
      //cv::flip(image, image, 1);
      cv::imshow( "RemoDet", image_draw);
      int key = cv::waitKey();
      if (key=='q'){
        cv::destroyAllWindows();
        break;
      }
    }
  }else if (use_imgs){
  std::vector<std::string> img_list;
  // std::string img_root = "/home/jun/work/test_images/0522";
  int num_imgs = get_filenames(img_root, img_list);
  std::sort(img_list.begin(), img_list.end());
  int k=0;
  int count = 0;
  cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
  int act=0;
  while (1){
    if (k>=num_imgs){
      k = k%num_imgs;
    }else if (k<0){
      k += num_imgs;
    }
    string name_img = img_list.at(k);
    // -----------------
      ++count;
      cv::Mat image, image_draw;
      image = cv::imread(name_img);
      if (!image.data){break;}
      cv::resize(image, image, cv::Size(cam_width, cam_height), cv::INTER_LINEAR);
      image_draw = image.clone();
      caffe::DataFrame<float> data_frame(count, image, resized_width, resized_height);
      // 获取hand_detector的结果
      vector<LabeledBBox<float> > rois;
      det_wrapper.get_rois(data_frame, &rois);
      vector<BoundingBox<float> > head_boxes;
      vector<BoundingBox<float> > face_boxes;
      vector<BoundingBox<float> > hand_boxes;
      vector<BoundingBox<float> > body_boxes;
      getHeadBoxes(rois, &head_boxes);
      getFaceBoxes(rois, &face_boxes);
      getHandBoxes(rois, &hand_boxes);
      getBodyBoxes(rois, &body_boxes);
      filterHandBoxes(hand_boxes, face_boxes);
    //  filterBodyBoxes(body_boxes, head_boxes);
      /**
      * 进一步滤除,每个Face最多允许一个HandBox
      * (1) 滤除距离:头的max(w,h)的两倍距离,超过这个距离的全部忽略
      * (2) 如果保留多个,则只选择其中
      */
      vector<bool> active_hands = filterHandBoxesbyDistance(face_boxes, hand_boxes);
      // 绘制Head
      test_head(image, image_draw, head_boxes, show_head);

      // 绘制Face
      // test_face(smile_wrapper, image, image_draw, face_boxes, show_face, flag_smile, count2);

      /*** 绘制 Body ***/
      // test_body(sa_base_wrapper, sa_wrapper, image, image_draw, body_boxes, head_boxes, show_body);

      // 绘制minihand网络的结果
      // test_hand(hp_wrapper, image, image_draw, active_hands, hand_boxes, show_hand);
      test_hand(hp_wrappers, image, image_draw, active_hands, hand_boxes, show_hand, models);

      /*** imwrite ***/ 
    //  if (write){
    //   flag_smile = false;
    //   stringstream ss;
    //   ss << image_folder << "/" << setw(7) << setfill('0') << count << ".jpg";
    //   cv::imwrite(ss.str(), image);
    //  }

      char tmp_str[256];
      snprintf(tmp_str, 256, "%d/%d", k+1, num_imgs);
      cv::putText(image_draw, tmp_str, cv::Point(0, cam_height), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5);

      //cv::flip(image, image, 1);
      // cv::putText(image_draw, folder, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 1);
      cv::imshow( "RemoDet", image_draw);
      int key = cv::waitKey(act);
      if (key==' '){
        // key = act ? cv::waitKey(0) : cv::waitKey(1);
        act = 1-act;
      }
      if (key=='q'){
        cv::destroyAllWindows();
        break;
      }

      if (key=='j' || key=='i'){
        k--;
        continue;
      }
      if (key=='u'){
        k -= 10;
        continue;
      }
      if (key=='o'){
        k += 10;
        continue;
      }
      if (key=='y'){
        k -= 100;
        continue;
      }
      if (key=='p'){
        k += 100;
        continue;
      }
      k ++;
  }

  }else{
    LOG(INFO) << "No mode was assigned to run.";
  }
  //  std::cout << "the number of smile face is " << count2 << std::endl;
  LOG(INFO) << "Number of smile face is " << count2 << ".";
  LOG(INFO) << "Finished.";
  return 0;
  }
