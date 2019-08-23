#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/tracker/image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"
#include "caffe/pose/mpii_image_loader.hpp"
#include "caffe/pose/mpii_image_masker.hpp"
#include "caffe/pose/coco_image_loader.hpp"
#include "caffe/pose/xml_modification.hpp"

#include "caffe/pose/mpii_save_txt.hpp"
#include "caffe/pose/coco_save_txt.hpp"
#include "caffe/pic/pic_visual_saver.hpp"

#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/mask/coco_anno_loader.hpp"

#include "caffe/hand/coco_hand_loader.hpp"
#include "caffe/hand/hand_loader.hpp"
#include "caffe/hand/mpii_hand_loader.hpp"

int main(int argc, char** argv) {

  // const std::string box_xml_dir = "/home/zhangming/data/MPII/xml_box_rscale";
  // const std::string kps_xml_dir = "/home/zhangming/data/MPII/xml_kps_rscale";
  // const std::string image_folder = "/home/zhangming/data/MPII";
  // const std::string output_folder = "/home/zhangming/data/MPII/xml_rscale";
  //
  // caffe::MpiiMasker<float> mpii_masker(box_xml_dir,kps_xml_dir,image_folder,output_folder);
  // mpii_masker.Process(true,true);

  // const std::string xml_dir = "/home/zhangming/data/MPII/xml_rscale";
  // const std::string output_file = "/home/zhangming/data/MPII/txt/mpii_info_train.txt";
  // caffe::MpiiTxtSaver<float> mpii_txt_saver(xml_dir,output_file);
  // mpii_txt_saver.Save();

  // const std::string pic_file = "/home/zhangming/data/MPII/txt/goutu.txt";
  // const std::string image_dir = "/home/zhangming/data/MPII/images";
  // const std::string output_dir = "/home/zhangming/data/MPII/pic";
  // caffe::PicVisualSaver<float> pic_saver(pic_file,image_dir,output_dir);
  // pic_saver.Save();

  // const std::string xml_dir = "/home/zhangming/data/coco/COCO/xml/train";
  // const std::string xml_dir = "/home/zhangming/data/coco/COCO/xml/val";
  // const std::string output_file = "/home/zhangming/data/coco/txt/coco_val2014_info.txt";
  // caffe::CocoTxtSaver<float> coco_txt_saver(xml_dir,output_file);
  // coco_txt_saver.Save();

  // const std::string xml_dir = "/home/zhangming/data/coco/COCO/xml/val";
  // const std::string image_dir = "/home/zhangming/data/coco";
  // const std::string output_dir = "/home/zhangming/data/coco/images_val2014";
  // caffe::CocoImageLoader<float> coco_image_loader(image_dir, xml_dir);
  // coco_image_loader.Saving(output_dir, true);

  // const std::string xml_dir = "/home/zhangming/data/coco/UNI_COCO/xml/val";
  // const std::string image_dir = "/home/zhangming/data/coco";
  // const std::string output_dir = "/home/zhangming/data/coco/vis_unified";
  // caffe::CocoAnnoLoader<float> coco_anno_loader(image_dir, xml_dir);
  // coco_anno_loader.Saving(output_dir);
  // coco_anno_loader.ShowAnnotations();

  // const std::string xml_dir = "/home/zhangming/data/coco/UNI_COCO/xml/train";
  // const std::string image_dir = "/home/zhangming/data/coco";
  // const std::string output_dir = "/home/zhangming/data/coco/hand_images/coco_train";
  // const std::string save_predix = "COCO_TRAIN";
  // caffe::CocoHandLoader<float> coco_hand_loader(image_dir, xml_dir);
  // // coco_hand_loader.ShowHand();
  // coco_hand_loader.Saving(output_dir,save_predix);

  const std::string xml_dir = "/home/zhangming/data/MPII/xml_kps";
  const std::string image_dir = "/home/zhangming/data/MPII";
  const std::string output_dir = "/home/zhangming/data/MPII/hand_images";
  const std::string save_predix = "MPII";
  caffe::MpiiHandLoader<float> mpii_hand_loader(image_dir, xml_dir);
  // mpii_hand_loader.ShowHand();
  mpii_hand_loader.Saving(output_dir,save_predix);
}
