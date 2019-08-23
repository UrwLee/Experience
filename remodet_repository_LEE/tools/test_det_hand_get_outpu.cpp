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
   // Network config
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_Person.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPose_AIC_PersonFace_1A.prototxt";
   const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_PersonFaceHeadHand_1H.prototxt";
    // const std::string network_proto = "/home/ethan/ForZhangM/Release20/180124_PersonFaceHeadDet_Track_Pose/Release20180124_Merged/test_all_merge_OnlyDetPerson.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR.prototxt";
   const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H_iter_460000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_iter_500000.caffemodel";
   // const std::string caffe_model = "/home/ethan/work/PoseDetJointTrain/ResNetPoseDet_JointTrain_I_L_iter_500000_ChangeNameOneBaseNet.caffemodel";
    // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180124_PersonFaceHeadDet_Track_Pose/Release20180124_Merged/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_ReTr_F_iter_80000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000.caffemodel";
   //ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000
   //ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000
   //ResNetPose_AIC_Person_A_iter_240000
   
   // GPU
    int gpu_id = 0;
    bool mode = true;  // use GPU
   // features
    const std::string proposals = "det_out";
   // display Size
    int max_dis_size = 512;
   // active labels
    vector<int> active_label;
    active_label.push_back(0);
    active_label.push_back(1);
    active_label.push_back(2);
    active_label.push_back(3);
   // ################################ DATA ####################################
  
    // const std::string img_list_txt = "/home/ethan/DataSets/REID/Market-1501/Layout/Layout_market1501_test.txt";
    // const std::string img_list_txt_dest = "/home/ethan/DataSets/REID/Market-1501/Layout/Layout_market1501_test_Filter_I_L_Detconf0.6Nms0.5.txt";
    // const std::string image_dir_txt = "/home/ethan/DataSets/REID/Market-1501";

    // const std::string img_list_txt = "/home/ethan/DataSets/REID/DukeMTMC4ReID/Layout/Layout_DukeMTMC4ReID_traintest.txt";
    // const std::string img_list_txt_dest = "/home/ethan/DataSets/REID/DukeMTMC4ReID/Layout/Layout_DukeMTMC4ReID_traintest_Filter_I_L_Detconf0.6Nms0.5.txt";
    // const std::string image_dir_txt = "/home/ethan/DataSets/REID/DukeMTMC4ReID/ReID";
    
    // const std::string img_list_txt = "/home/ethan/DataSets/REID/iLIDS_VID/Layout/Layout_lids_vid_train.txt";
    // const std::string img_list_txt_dest = "/home/ethan/DataSets/REID/iLIDS_VID/Layout/Layout_lids_vid_train_Filter_I_L_Detconf0.6Nms0.5.txt";
    // const std::string image_dir_txt = "/home/ethan/DataSets/REID/iLIDS_VID";

     const std::string img_list_txt = "/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/train_img.txt";
    const std::string img_list_txt_dest = "/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/Hand_Det_PersonFaceHeadHand_1H_iter460000_train.txt";
    const std::string image_dir_txt = "/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/AIC_SRC/keypoint_train_images_20170902";

    float img_scale = 1.0;
   // ################################ MAIN LOOP ################################
   // det_warpper
    caffe::DetWrapper<float> det_wrapper(network_proto,caffe_model,mode,gpu_id,proposals,max_dis_size);
    std::ifstream infile(img_list_txt.c_str());
    CHECK(infile.good()) << "Failed to open file "<< img_list_txt;
    std::vector<std::string> lines;
    std::string str_line;
    while (std::getline(infile, str_line)) {
      lines.push_back(str_line);
    }
    std::ofstream fout;
    fout.open(img_list_txt_dest.c_str());
    int count = 0;
    // LOG(INFO)<<image_dir_txt;
    for (int i=0;i < lines.size();++i){
      // std::vector<string> v;
      // SplitString(lines[i], v," ");
      std::string image_path = image_dir_txt + '/' + lines[i];

      cv::Mat image = cv::imread(image_path);
      int img_org_width = image.cols;
      int img_org_height = image.rows;
      int img_big_width, img_big_height;

      if (float(img_org_width)/float(img_org_height)>16.0/9.0){
        img_big_width = img_org_width;
        img_big_height = int(9.0*float(img_org_width)/16.0);
      } else{
        img_big_width = int(float(img_org_height)*16.0/9.0);
        img_big_height = img_org_height;
      }
      
      cv::Mat bg_image(img_big_height,img_big_width, CV_8UC3, cv::Scalar(0,0,0));
      float scale = float(img_big_width)/512.0;
      cv::Rect patch_on_big(0, 0, img_org_width, img_org_height);
      cv::Mat bg_patch = bg_image(patch_on_big);
      image.copyTo(bg_patch);
      caffe::DataFrame<float> data_frame(0,bg_image,resized_width,resized_height);
      vector<LabeledBBox<float> > roi;
      cv::Mat det_image = det_wrapper.get_drawn_bboxes(data_frame, active_label, &roi);
      fout<<lines[i]<<" ";
      for(int id_roi=0;id_roi<roi.size();id_roi++){
        if(roi[id_roi].cid==1){
          caffe::BoundingBox<float> bbox = roi[id_roi].bbox;

          float xmin = bbox.x1_*scale;
          float xmax = bbox.x2_*scale;
          float ymin = bbox.y1_*scale;
          float ymax = bbox.y2_*scale;
          fout<<int(xmin)<<" ";
          fout<<int(ymin)<<" ";
          fout<<int(xmax)<<" ";
          fout<<int(ymax)<<" ";
        }
      }
      if (i<lines.size())
          fout<<"\n";
      // LOG(INFO)<<"image_path "<<image_path<<" "<<roi.size();
      // cv::namedWindow("RemoDet", cv::WINDOW_AUTOSIZE);
      // cv::imshow( "RemoDet", det_image);
      // cv::waitKey(0);
      if(i%500==0)
        LOG(INFO)<<i <<"/"<<lines.size();
    }
    fout.close();
    LOG(INFO) << "Finished.";
    return 0;
 }
