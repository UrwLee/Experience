#ifndef CAFFE_MINIHAND_BBOX_DATA_LAYER_HPP_
#define CAFFE_MINIHAND_BBOX_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"
// #include "caffe/minihand/minihand_transformer.hpp"
#include "caffe/minihand/minihand_sample_transformer.hpp"

namespace caffe {
template <typename Dtype>
class MinihandDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MinihandDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), minihand_transform_param_(param.minihand_transform_param()) {
        setupBg();
      }
  virtual ~MinihandDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MinihandData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleLists();
  virtual void load_batch(Batch<Dtype>* batch);

  void ReadHandDataFromXml(const int bindex, const string& xml_file, const string& image_root,
                           HandAnnoData<Dtype>* anno);

  // 所有XML文件的集合列表 root<--->path pair
  vector<pair<std::string, std::string> > lines_;

  // 当前加载的XML编号
  int lines_id_;

  int ndim_label_; // 输出label 维度
  // 数据转换器参数
  MinihandTransformationParameter minihand_transform_param_;
  // 数据转换器
  boost::shared_ptr<MinihandSampleTransformer<Dtype> > minihand_transformer_; 
  // RGB颜色通道均值 
  vector<Dtype> mean_values_;
  int base_bindex_;
     // =====================================
    // 背景图 lines_ 
  vector<pair<std::string, std::string> > bg_lines_;
  int bg_lines_id_  ;  
  void setupBg(); // 初始化背景图列表
  void readBgImg(cv::Mat& bg_img); // 从背景图中读取图片


};

}

#endif
