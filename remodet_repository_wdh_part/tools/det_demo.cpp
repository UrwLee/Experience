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
   // const std::string network_proto = "/home/ethan/Models/Results/Minihand/test.prototxt";
   // const std::string caffe_model = "/home/ethan/Models/Results/Minihand/R20180425_Conv3_3_B48_Deconv64_3_32Chan_D20180427AICREMO_iter_280000_rewrite.caffemodel";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_Person.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPose_AIC_PersonFace_1A.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_JointTrain_I_L_HeadFaceHand_001A.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_V1-B12_C1S-C2S_MultiScale_A3.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_JointTrain_I_L_WithFaceHead_1A.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_Person_NewName.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_V1-B48_C1S-C2S_MultiScale_A6.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test.prototxt";
   // const std::string network_proto = "/home/ethan/Models/Results/Minihand/test.prototxt";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/test_all_merge_detall.prototxt";
   // const std::string network_proto = "/home/ethan/Models/Results/Minihand/ResNetPoseDet_V1-B12-Context-NO-Aug_Use-C2-6-and-conv1-recon-Concat/Proto/test.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H.prototxt";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/test_all_merge.prototxt";
   // const std::string network_proto = "/home/ethan/Models/Results/Minihand/test.prototxt";
   // const std::string network_proto = "/home/ethan/Models/Results/DAPDet/test_FB-FPN-512x288_S8-I4-C5C6_1A.prototxt";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180425/DarkNet_FT_BDPD.prototxt";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180515_TMP/DetPose_JtTr_DarkNet20180515_TrunkBD_DataMinS0.75MaxS2.0NoExp_HisiDataOnlySmpBody_WD5e-3_FTFromPose_1A.prototxt";
    // const std::string network_proto = "/home/ethan/ForZhangM/Release20180425/DarkNet_FT_BDPD_Merge_ZM.prototxt";
    // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/ResNetPoseDet_V1-B48_C1S-C2S_MultiScale_ColDist_A13.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_PersonFaceHeadHand.prototxt";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180124_PersonFaceHeadDet_Track_Pose/Release20180124_Merged/test_all_merge_OnlyDetPerson.prototxt";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR.prototxt";
   // const  std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/test_detperson_concatreluseperate.prototxt";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H_MiniMS6A_iter_140000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_V1-B48_C1S-C2S_MultiScale_MoreNeg_A12_iter_300000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_WithFaceHead_1A_iter_400000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_V1-B48_C1S-C2S_MultiScale_A6_iter_300000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_BNTrue_iter_120000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_iter_500000.caffemodel";
   // const std::string caffe_model = "/home/ethan/work/PoseDetJointTrain/ResNetPoseDet_JointTrain_I_L_iter_500000_ChangeNameOneBaseNet.caffemodel";
 // const  std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_V1-B48_C1S-C2S_MultiScale_A6_iter_300000.caffemodel";
    // const std::string caffe_model = "/home/ethan/Models/Results/Minihand/R20180425_Conv3_3_B48_Deconv64_3_32Chan_D20180427AICREMO_iter_100000_rewrite.caffemodel";
    // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
     // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180124_PersonFaceHeadDet_Track_Pose/Release20180124_Merged/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_ReTr_F_iter_80000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H_iter_460000.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180425/DarkNet_FT_BDPD.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180515_TMP/DetPose_JtTr_DarkNet20180515_TrunkBD_DataMinS0.75MaxS2.0NoExp_HisiDataOnlySmpBody_WD5e-3_FTFromPose_1A_iter_30000.caffemodel";
      // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180425/DarkNet_FT_BDPD_Merge_ZM.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/Results/Minihand/R20180425_Conv3_3_B48_Deconv64_3_32Chan_D20180427AICREMO_iter_280000_rewrite.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/Results/Minihand/ResNetPoseDet_V1-B12-Context-NO-Aug_Use-C2-6-and-conv1-recon-Concat/Models/ResNetPoseDet_V1-B12-Context-NO-Aug_Use-C2-6-and-conv1-recon-Concat_iter_300000.caffemodel";
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPoseDet_JointTrain_I_L_HeadFaceHand_002A_iter_80000.caffemodel";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180124_PersonFaceHeadDet_Track_Pose/NecessaryModels/ResNetPoseDet_JointTrain_J_D_iter_180000.caffemodel";
   // ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000
   //ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000
   //ResNetPose_AIC_Person_A_iter_240000
   // const std::string caffe_model = "/home/ethan/Models/ResNet/caffemodels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_20000.caffemodel";
    // const std::string network_proto   = "/home/ethan/ForZhangM/Release20180515_TMP/DetPose_JtTr_DarkNet_20180514_TrunkBD_PDHeadHand_DataMinS0.75MaxS1.0Exp2.5_HisiDataOnlySmpBody_WD5e-3_1A_FTBD_Pose_merge.prototxt";
    // const std::string caffe_model ="/home/ethan/ForZhangM/Release20180515_TMP/DetPose_JtTr_DarkNet_20180514_TrunkBD_PDHeadHand_DataMinS0.75MaxS1.0Exp2.5_HisiDataOnlySmpBody_WD5e-3_1A_FTBD_Pose_iter_60000_merge.caffemodel";
   // const std::string network_proto = "/home/ethan/Models/ResNet/prototxts/test_2SSD-MA2-OHEM-PLA-LLR_Person.prototxt";
    // const std::string caffe_model = "/home/ethan/work/PoseDetJointTrain/ResNetPoseDet_JointTrain_I_L_iter_500000_ChangeNameOneBaseNet.caffemodel";
// const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/test_detperson.prototxt";
// const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
 // const std::string network_proto = "/home/ethan/ForZhangM/Release20180506/Release20180506/R20180506_Trunk_BD_PD_MiniHand_Com_V0.prototxt";
    // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180506/Release20180506/R20180506_Trunk_BD_PD_MiniHand_Com_V0.caffemodel";
   // const std::string network_proto = "/home/ethan/for_DJ/test.prototxt";
   // const std::string caffe_model = "/home/ethan/for_DJ/DetPose_JtTr_DarkNet20180514MultiScaleNoBN_FTFromPose_TrunkBD_PDFaceHandLossWMult0.25_DataMinS0.75MaxS2.0NoExp_OnlySmpBody_WD5e-3_1A_iter_finetunePD0606_iter_85000.caffemodel";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_MiniHand.prototxt";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180606/minihand_basedon_DetPose_JtTr_DarkNet20180514MultiScaleNoBN_FTFromPose_TrunkBD_PDFaceHandLossWMult0.25_DataMinS0.75MaxS2.0NoExp_OnlySmpBody_WD5e-3_1A_iter_500000_rewrite.caffemodel";

// const std::string network_proto = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNet20180717MulScaleNoBN_FTFromPose_BDPDBBoxDataMinS0.75MaxS2.0ForBodyTrue_ForFeatEachAllPriorBox_WD5e-3_1A/test_BD_feat4.prototxt";
// const std::string caffe_model = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNet20180717MulScaleNoBN_FTFromPose_BDPDBBoxDataMinS0.75MaxS2.0ForBodyTrue_ForFeatEachAllPriorBox_WD5e-3_1A/DetPose_JtTr_DarkNet20180717MulScaleNoBN_FTFromPose_BDPDBBoxDataMinS0.75MaxS2.0ForBodyTrue_ForFeatEachAllPriorBox_WD5e-3_1A_iter_70000.caffemodel";
const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand.prototxt";
const std::string caffe_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHandDist0.5_V0.caffemodel";
// const std::string network_proto = "/home/ethan/Models/Results/DetNet_Minihand/test_R20180606_Conv4_5_B48_Deconv64EltSumConv2Hand_Two32Inter_D20180530AICREMO.prototxt";
// const std::string caffe_model = "/home/ethan/Models/Results/DetNet_Minihand/R20180606_Conv4_5_B48_Deconv64EltSumConv2Hand_Two32Inter_D20180530AICREMO_Dist0.8_1A/R20180606_Conv4_5_B48_Deconv64EltSumConv2Hand_Two32Inter_D20180530AICREMO_Dist0.8_1A_iter_300000_rewrite.caffemodel";
// const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Conv4_5_B48_Deconv64EltSumConv2Hand_Two32Inter_D20180530AICREMO_Dist_merge.prototxt";
// const std::string caffe_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Conv4_5_B48_Deconv64EltSumConv2Hand_Two32Inter_D20180530AICREMO_Dist0.5_1A_iter_300000_rewrite_merge.caffemodel";
// const std::string network_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_PD_MiniHand.prototxt";
// const std::string caffe_model = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel";
// const std::string network_proto = "/home/ethan/ForZhangM/Release20180809_9_16/R20180809_9_16_Base_BD_PD_MiniHand_CCom_Convf.prototxt";
// const std::string caffe_model = "/home/ethan/ForZhangM/Release20180809_9_16/R20180809_9_16_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel";
// const std::string network_proto = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNetFPNAlike20180628FeatBDOnPoseStage1Conv4_FTFromPose_BDPoseOneLayerMinS0.75MaxS1.0Exp2.0_PDFaceHandMinS0.05MaxS2.0LossWMult0.25_WD5e-3_1A/test_BD_featmapconvf.prototxt";
// const std::string caffe_model = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNetFPNAlike20180628FeatBDOnPoseStage1Conv4_FTFromPose_BDPoseOneLayerMinS0.75MaxS1.0Exp2.0_PDFaceHandMinS0.05MaxS2.0LossWMult0.25_WD5e-3_1A/DetPose_JtTr_DarkNetFPNAlike20180628FeatBDOnPoseStage1Conv4_FTFromPose_BDPoseOneLayerMinS0.75MaxS1.0Exp2.0_PDFaceHandMinS0.05MaxS2.0LossWMult0.25_WD5e-3_1A_iter_110000.caffemodel";
   // const std::string network_proto = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/Base_BD_PD_HD.prototxt";
   // const std::string caffe_model = "/home/ethan/ForZhangM/Release20180314/Release20180314_Merged_profo2.5/ResPoseDetTrackRelease_merge.caffemodel";
// const std::string network_proto = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNetOrgConv1To5ThreeFeatmap_WD5e-3_1A/test_BD.prototxt";
// const std::string caffe_model = "/home/ethan/Models/Results/DetPose_JointTrain/DetPose_JtTr_DarkNetOrgConv1To5ThreeFeatmap_WD5e-3_1A/DetPose_JtTr_DarkNetOrgConv1To5ThreeFeatmap_WD5e-3_1A_iter_15000.caffemodel";
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
   active_label.push_back(0);
   active_label.push_back(1);
   active_label.push_back(2);
   active_label.push_back(3);
   // ################################ DATA ####################################
   // CAMERA
   const bool use_camera = true; // 0
   const int cam_width = 1280;
   const int cam_height = 720;
   // VIDEO
   const bool use_video = true;
   const std::string video_file = "/home/ethan/work/doubleVideo/c1.mp4";
    // const std::string video_file = "/home/ethan/work/doubleVideo/d1.mp4";
   // const std::string video_file = "/home/ethan/work/testvidoes/test_bd_20180626_1.avi";
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
     outputVideo.open("tmp.avi", ex, cap.get(CV_CAP_PROP_FPS), cv::Size(write_width,write_height), true);
     cv::Mat cv_img;
     cap >> cv_img;
     LOG(INFO)<<cv_img.rows<<" "<<cv_img.cols;
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
           char pstr[256];
          snprintf(pstr, 256, "%.3f", (float)l_bbox.score);
          cv::putText(det_image, pstr, cv::Point(l_bbox.bbox.x1_,l_bbox.bbox.y1_+50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255), 2);
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
