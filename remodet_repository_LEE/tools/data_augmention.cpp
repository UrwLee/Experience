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

#include "caffe/mask/unified_data_layer.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  const std::string xml_list = "/home/zhangming/Datasets/coco/UNI_COCO/Layout/val2014.txt";
  const std::string xml_root = "/home/zhangming/Datasets/coco";
  const std::string save_dir = "/home/zhangming/data/coco/aug_unified";
  // 生成Layer
  LayerParameter layer_param;
  layer_param.set_name("unifiedDataLayer");
  layer_param.set_type("UnifiedData");
  layer_param.set_phase(caffe::TRAIN);
  UnifiedTransformationParameter* udtp = layer_param.mutable_unified_data_transform_param();
  // 设置数据转换器
  udtp->set_emit_coverage_thre(0.25);
  udtp->set_kps_min_visible(4);
  udtp->set_flip_prob(0.5);
  udtp->set_resized_width(512);
  udtp->set_resized_height(288);
  udtp->set_visualize(true);
  udtp->set_save_dir(save_dir);
  // 设置颜色失真
  DistortionParameter* disp = udtp->mutable_dis_param();
  // brightness
  disp->set_brightness_prob(0.5);
  disp->set_brightness_delta(32);
  // contrast
  disp->set_contrast_prob(0.5);
  disp->set_contrast_lower(0.5);
  disp->set_contrast_upper(1.5);
  // hue
  disp->set_hue_prob(0.5);
  disp->set_hue_delta(18);
  // saturation
  disp->set_saturation_prob(0.5);
  disp->set_saturation_lower(0.5);
  disp->set_saturation_upper(1.5);
  // 设置裁剪参数
  // 设置第一个采样器
  BatchSampler* bs = udtp->add_batch_sampler();
  bs->set_max_sample(1);
  bs->set_max_trials(50);
  Sampler* sam = bs->mutable_sampler();
  SampleConstraint* sc = bs->mutable_sample_constraint();
  sam->set_min_scale(0.5);
  sam->set_max_scale(1.0);
  sc->set_min_jaccard_overlap(0.1);
  // 设置第二个采样器 ...

  // 设置数据读入层参数
  UnifiedDataParameter* udp = layer_param.mutable_unified_data_param();
  udp->set_xml_list(xml_list);
  udp->set_xml_root(xml_root);
  udp->set_shuffle(true);
  udp->set_rand_skip(100);
  udp->set_batch_size(12);
  udp->add_mean_value(104);
  udp->add_mean_value(117);
  udp->add_mean_value(123);

  // 构造数据输入层
  boost::shared_ptr<caffe::Layer<float> > udlayer = LayerRegistry<float>::CreateLayer(layer_param);
  LOG(INFO) << "[BATCHSIZE] : " << layer_param.unified_data_param().batch_size();
  vector<Blob<float>*> bottom_vec;
  vector<Blob<float>*> top_vec;
  top_vec.push_back(new Blob<float>(12,3,288,512));
  top_vec.push_back(new Blob<float>(1,1,1,66+288*512));
  udlayer->LayerSetUp(bottom_vec,top_vec);
  // LOG(INFO) << "[BATCHSIZE]: -> ";
  LOG(INFO) << "Beging DataLayer Forward ...";
  for (int i = 0; i < 1; ++i) {
    udlayer->Forward(bottom_vec,top_vec);
    LOG(INFO) << "A mini-batch loaded.";
  }
  // LOG(INFO) << "Done..........................................................";
}
