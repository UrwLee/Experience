#ifndef CAFFE_MASK_BBOX_DATA_LAYER_HPP_
#define CAFFE_MASK_BBOX_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/mask/bbox_data_transformer.hpp"
#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {
template <typename Dtype>
class BBoxDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BBoxDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), bbox_data_transform_param_(param.unified_data_transform_param()) {}
  virtual ~BBoxDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BBoxData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  //  prefetch随机数
  shared_ptr<Caffe::RNG> prefetch_rng_;

  // 输入样本顺序随机乱序
  virtual void ShuffleLists();

  // 载入一个minibatch
  virtual void load_batch(Batch<Dtype>* batch);

  void ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                           AnnoData<Dtype>* anno);
  // 所有XML文件的集合列表 root<--->path pair
  vector<pair<std::string, std::string> > lines_;
  vector<std::string> lin_test_; 
  // 当前加载的XML编号
  int lines_id_;
  // unused.
  Blob<Dtype> transformed_label_;

  // 数据转换器参数
  UnifiedTransformationParameter bbox_data_transform_param_;

  // 数据转换器
  shared_ptr<BBoxDataTransformer<Dtype> > bbox_data_transformer_;

  // RGB颜色通道均值
  vector<Dtype> mean_values_;

  // add_parts
  bool add_parts_;

  std::map<int,int> maps_;
  bool flag_hisimap_;
  vector<int> check_area_;
  int base_bindex_;
  int ndim_label_;

  // 背景图 lines_ 
  vector<pair<std::string, std::string> > bg_lines_;
  int bg_lines_id_  ;  
  void setupBg(); // 初始化背景图列表
  void readBgImg(cv::Mat& bg_img); // 从背景图中读取图片
};

}

#endif
