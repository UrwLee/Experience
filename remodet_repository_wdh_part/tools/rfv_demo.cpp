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

#include <boost/shared_ptr.hpp>

using namespace std;

#define CPU_MODE false
#define GPU_MODE true

int main(int nargc, char** args) {
   // network input
   const int resized_width = 512;
   const int resized_height = 288;
   // Network config
   const std::string network_proto = "/home/zhangming/Models/Release/release_2/release_2_2.prototxt";
   const std::string caffe_model = "/home/zhangming/Models/Release/release_2/release_2.caffemodel";
   // caffe Mode
   const bool caffe_mode = GPU_MODE;
   // GPU ID
   const int gpu_id = 0;
   // features
   const std::string proposals = "proposals";
  //  const std::string heatmaps = "concat_stage2";
  //  const std::string heatmaps = "heat_vec_concat";
   const std::string heatmaps = "resized_map";
   // display Size
   int max_dis_size = 1000;
   // drawn mode
   caffe::DrawnMode mode = caffe::SKELETON_BOX;
   //caffe::DrawnMode mode = caffe::HEATMAP;
   //caffe::DrawnMode mode = caffe::VECMAP;
   // ################################ WEBCAM ####################################
    //  int cam_id = 0;
    //  int cam_width = 1280;
    //  int cam_height = 720;
    //  caffe::RemoFrontVisualizer<float> rfv(cam_id,cam_width,cam_height,
    //                                resized_width,resized_height,
    //                                network_proto,caffe_model,caffe_mode,gpu_id,
    //                                proposals,heatmaps,max_dis_size,mode);
   // ################################ VIDEO #####################################
    const std::string video_file = "/home/zhangming/video/FigureSkating1.mp4";
    // const std::string video_file = "/home/zhangming/video/AdultVideos/30.Tushy - Janice Griffith - My Fantasy of a Double Penetration.mp4";
    // const std::string video_file = "/home/zhangming/video/tutu_1.mp4";
    int frame_skip = 0;
    // // const std::string video_file = "/home/zhangming/video/1.mkv";
    // // int frame_skip = 130;
    caffe::RemoFrontVisualizer<float> rfv(video_file,frame_skip,
                                    resized_width,resized_height,
                                    network_proto,caffe_model,caffe_mode,gpu_id,
                                    proposals,heatmaps,max_dis_size,mode);
    // print network params
    // boost::shared_ptr<caffe::Net<float> > net = rfv.net_wrapper_->get_net();
    // for (int i = 0; i < net->layers().size(); ++i) {
    //   const boost::shared_ptr<caffe::Layer<float> > layer = net->layers()[i];
    //   std::cout << '\n';
    //   std::cout << "===========================================================================" << '\n';
    //   std::cout << "Layer Name: " << net->layer_names()[i] << '\n';
    //   std::cout << "Layer Type: " << layer->layer_param().type() << '\n';
    //   // print param
    //   for (int j = 0; j < layer->blobs().size(); ++j) {
    //     std::cout << "\nLayer Param " << j << " :\n";
    //     const boost::shared_ptr<caffe::Blob<float> > pb = layer->blobs()[j];
    //     for (int k = 0; k < pb->count(); ++k) {
    //       std::cout << pb->cpu_data()[k] << " ";
    //       if (k > 0 && k % 10 == 9) {
    //         std::cout << '\n';
    //       }
    //       if (k >= 10 * 3) {
    //         break;
    //       }
    //     }
    //   }
    // }

   // Running
   while(1) {
       std::vector<std::vector<float> > meta;
       int status = rfv.step(&meta);
       //-----------------------------------------------------------------------
       caffe::Blob<float> monitor;
       std::string blob_name = "detection_out";
       rfv.net_wrapper_->getFeatures(blob_name,&monitor);
       for (int i = 0; i < monitor.count(); ++i) {
         std::cout << monitor.cpu_data()[i] << " ";
         if (i > 0 && i % 10 == 9) {
           std::cout << '\n';
         }
         if (i >= 10 * 10) {
           break;
         }
       }
       break;
       //-----------------------------------------------------------------------
       if (status) {
         LOG(INFO) << "Frame reader finished.";
         break;
       }
       // get frame id
       int frames = rfv.get_frames();
       // get time & fps
       if (frames % 30 == 0) {
         // get FPS info.
         float net_fps, prj_fps;
         rfv.fps(&net_fps,&prj_fps);
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "Frame ID: " << frames << ", Network FPS: "
                   << net_fps << ", Project FPS: " << prj_fps;
         // get TIME info.
         float frame_loader,preprocess,forward,drawn,display;
         rfv.time_us(&frame_loader, &preprocess, &forward, &drawn, &display);
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "Load Frame : " << frame_loader/1000 << " ms. ";
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "PreLoader  : " << preprocess/1000 << " ms. ";
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "Forward    : " << forward/1000 << " ms. ";
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "Drawn      : " << drawn/1000 << " ms. ";
         LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << "Display    : " << display/1000 << " ms. ";
       }
    }
    LOG(INFO) << "Finished.";

    return 0;
}
