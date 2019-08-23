#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/remo/remo_front_visualizer.hpp"
#include "caffe/remo/basic.hpp"
#include "caffe/remo/data_frame.hpp"
#include "caffe/remo/frame_reader.hpp"
#include "caffe/remo/net_wrap.hpp"
#include "caffe/remo/res_frame.hpp"
#include "caffe/remo/visualizer.hpp"
#include "caffe/tracker/bounding_box.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

using namespace std;
using namespace caffe;

/**
 * 该程序提供了自动捕获手部图片的脚本。
 * 注意：该脚本为自动获取手部图片用于手势行为的分类。
 */

// 小于该像素的hand-box将被忽略
static const int SIZE_THRE = 32 * 32;
// 超过该范围的长宽比／宽长比的hand-box将被忽略
static const float ASPECT_RATIO = 1.2f;

// default: FALSE
// FALSE: 需要使用键盘来控制是否保存或丢弃　【对捕获到任何手部图片】
// TRUE: 无需键盘控制，捕获到的任何手部图片都将自动保存
static const bool continous_flag = true;

/**
 * 根据标注信息获得hand-box的位置
 * @param meta      [标注信息]
 * @param boxes     [hand-boxes]
 * @param num_hands [数量]
 */
void get_hand_box(const PMeta<float>& meta, std::vector<BoundingBox<float> >* boxes, int* num_hands);

/**
 * 判断hand-box是否是合格的
 * @param  box     [hand-box]
 * @param  width   [原始图像宽度]
 * @param  height  [原始图像高度]
 * @param  roi_box [原始图像上的ROI-box]
 * @return         [是或不是]
 */
bool active_box(BoundingBox<float>& box, const int width, const int height, BoundingBox<float>* roi_box);

/**
 * 主程序
 */
int main(int argc, char** argv) {
  int resized_width = 512;
  int resized_height = 288;
  // Network config
  const std::string network_proto = "/home/zhangming/Models/Release/release_2/release_2_2.prototxt";
  const std::string caffe_model = "/home/zhangming/Models/Release/release_2/release_2.caffemodel";
  // GPU
  int gpu_id = 0;
  // features
  const std::string proposals = "proposals";
  const std::string heatmaps = "resized_map";
  // display Size
  int max_dis_size = 1000;

  int cam_id = 0;
  int cam_width = 1280;
  int cam_height = 720;
  // frame_reader_
  caffe::FrameReader<float> frame_reader(cam_id,cam_width,cam_height,resized_width,resized_height);
  // net_wrapper_
  caffe::NetWrapper<float> net_wrapper(network_proto,caffe_model,gpu_id,proposals,heatmaps,max_dis_size);
  // ---------------------------------------------------------------------------
  // ----------------------------------Saving Params----------------------------
  // 采集日期与目录
  const std::string time_str = "0706";
  std::string save_dir;
  if (continous_flag) {
    save_dir = "/home/zhangming/data/REMOHANDS_TEMP/" + time_str;
  } else {
    save_dir = "/home/zhangming/data/REMOHANDS/" + time_str;
  }
  if (boost::filesystem::exists(save_dir)) {
    boost::filesystem::remove_all(save_dir);
  }
  boost::filesystem::create_directories(save_dir);
  // 初始编号
  const int init_index = 0;
  // ---------------------------------------------------------------------------
  // Running
  static int count = 0;
  while(1) {
    // pop data_frame
    DataFrame<float> curr_frame;
    if(frame_reader.pop(&curr_frame)) {
      LOG(INFO) << "Warning: The input stream is stopped by fault. please check.";
      return 1;
    }
    // run
    std::vector<PMeta<float> > metas;
    net_wrapper.get_meta(curr_frame, &metas);
    // show handboxes
    cv::Mat image = curr_frame.get_ori_image();
    // get handboxes
    for (int i = 0; i < metas.size(); ++i) {
      PMeta<float>& meta = metas[i];
      std::vector<BoundingBox<float> > boxes;
      int num_hands;
      get_hand_box(meta, &boxes, &num_hands);
      if (num_hands > 0) {
        for (int j = 0; j < num_hands; ++j) {
          BoundingBox<float>& box = boxes[j];
          BoundingBox<float> roi_box;
          if (active_box(box, image.cols, image.rows, &roi_box)) {
            if (!continous_flag) {
              // display
              cv::Mat image_copy = image.clone();
              cv::Point top_left_pt(roi_box.x1_,roi_box.y1_);
              cv::Point bottom_right_pt(roi_box.x2_,roi_box.y2_);
              cv::rectangle(image_copy, top_left_pt, bottom_right_pt, cv::Scalar(0,0,255), 2);
              cv::namedWindow("HandCap", cv::WINDOW_AUTOSIZE);
              cv::imshow( "HandCap", image_copy);
              int key = cv::waitKey(0);
              if (key == 13) {
                // saving
                cv::Rect roi((int)roi_box.x1_,(int)roi_box.y1_,(int)roi_box.get_width(),(int)roi_box.get_height());
                cv::Mat image_roi = image(roi);
                char buf[256];
                sprintf(buf, "%s/hand_remo_%08d.jpg", save_dir.c_str(),count+init_index);
                LOG(INFO) << "saving image: " << buf;
                imwrite(buf, image_roi);
                ++count;
              } else {
                LOG(INFO) << "the current image is rejected.";
              }
            } else {
              // continous run
              cv::Mat image_copy = image.clone();
              cv::Rect roi((int)roi_box.x1_,(int)roi_box.y1_,(int)roi_box.get_width(),(int)roi_box.get_height());
              cv::Mat image_roi = image_copy(roi);
              char buf[256];
              sprintf(buf, "%s/hand_remo_%08d.jpg", save_dir.c_str(),count+init_index);
              LOG(INFO) << "saving image: " << buf;
              imwrite(buf, image_roi);
              ++count;
            }
          }
        }
      } else {
        // just show the image, do nothing
        cv::namedWindow("HandCap", cv::WINDOW_AUTOSIZE);
        cv::imshow( "HandCap", image);
        cv::waitKey(1);
      }
    }
  }
  LOG(INFO) << "Finished.";

  return 0;
}

void get_hand_box(const PMeta<float>& meta, std::vector<BoundingBox<float> >* boxes, int* num_hands) {
  boxes->clear();
  *num_hands = 0;
  int re = 3, rw = 4;
  int le = 6, lw = 7;
  float scale_hand = 1;
  // RIGHT
  if (meta.kps[re].v > 0.05 && meta.kps[rw].v > 0.05) {
    BoundingBox<float> box;
    // get right hand
    float xe = meta.kps[re].x;
    float ye = meta.kps[re].y;
    float xw = meta.kps[rw].x;
    float yw = meta.kps[rw].y;
    float dx = xw - xe;
    float dy = yw - ye;
    float norm = sqrt(dx*dx+dy*dy);
    dx /= norm;
    dy /= norm;
    cv::Point2f p1,p2,p3,p4;
    p1.x = xw + norm*scale_hand*dy*0.5*9/16;
    p1.y = yw - norm*scale_hand*dx*0.5*9/16;
    p2.x = xw - norm*scale_hand*dy*0.5*9/16;
    p2.y = yw + norm*scale_hand*dx*0.5*9/16;
    p3.x = p1.x + norm*scale_hand*dx;
    p3.y = p1.y + norm*scale_hand*dy;
    p4.x = p2.x + norm*scale_hand*dx;
    p4.y = p2.y + norm*scale_hand*dy;
    // get bbox
    float xmin,ymin,xmax,ymax;
    xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    // get min square box include <...>
    float cx = (xmin+xmax)/2;
    float cy = (ymin+ymax)/2;
    float wh = std::max(xmax-xmin,(ymax-ymin)*9/16);
    xmin = cx - wh/2;
    xmax = cx + wh/2;
    ymin = cy - wh*16/9/2;
    ymax = cy + wh*16/9/2;
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    box.x1_ = xmin;
    box.y1_ = ymin;
    box.x2_ = xmax;
    box.y2_ = ymax;
    boxes->push_back(box);
    (*num_hands)++;
  }
  // LEFT
  if (meta.kps[le].v > 0.05 && meta.kps[lw].v > 0.05) {
    BoundingBox<float> box;
    float xe = meta.kps[le].x;
    float ye = meta.kps[le].y;
    float xw = meta.kps[lw].x;
    float yw = meta.kps[lw].y;
    float dx = xw - xe;
    float dy = yw - ye;
    float norm = sqrt(dx*dx+dy*dy);
    dx /= norm;
    dy /= norm;
    cv::Point2f p1,p2,p3,p4;
    p1.x = xw + norm*scale_hand*dy*0.5*9/16;
    p1.y = yw - norm*scale_hand*dx*0.5*9/16;
    p2.x = xw - norm*scale_hand*dy*0.5*9/16;
    p2.y = yw + norm*scale_hand*dx*0.5*9/16;
    p3.x = p1.x + norm*scale_hand*dx;
    p3.y = p1.y + norm*scale_hand*dy;
    p4.x = p2.x + norm*scale_hand*dx;
    p4.y = p2.y + norm*scale_hand*dy;
    // get bbox
    float xmin,ymin,xmax,ymax;
    xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    // get min square box include <...>
    float cx = (xmin+xmax)/2;
    float cy = (ymin+ymax)/2;
    float wh = std::max(xmax-xmin,(ymax-ymin)*9/16);
    xmin = cx - wh/2;
    xmax = cx + wh/2;
    ymin = cy - wh*16/9/2;
    ymax = cy + wh*16/9/2;
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    box.x1_ = xmin;
    box.y1_ = ymin;
    box.x2_ = xmax;
    box.y2_ = ymax;
    boxes->push_back(box);
    (*num_hands)++;
  }
}

bool active_box(BoundingBox<float>& box, const int width, const int height, BoundingBox<float>* roi_box) {
  box.clip();
  int roi_width = box.get_width() * width;
  int roi_height = box.get_height() * height;
  if (roi_width * roi_height < SIZE_THRE) {
    return false;
  }
  float as = (float)roi_height / (float)roi_width;
  if ((as < (float)1. / ASPECT_RATIO) || (as > ASPECT_RATIO)) {
    return false;
  }
  roi_box->x1_ = (int)(box.x1_ * width);
  roi_box->x2_ = (int)(box.x2_ * width);
  roi_box->y1_ = (int)(box.y1_ * height);
  roi_box->y2_ = (int)(box.y2_ * height);
  return true;
}
