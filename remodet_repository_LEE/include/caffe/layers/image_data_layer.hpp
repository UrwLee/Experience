#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该层为检测器的训练提供数据输入。
 * 该层遍历每张图片，并获取该图片的标准XML文件，读入标注。
 * 该层内部集成了数据转换器来处理数据。
 * 包括：
 * １．随机增广
 * ２．data/Label数据载入。
 * 该层与mask/unified_data_layer比较相似，可以参考mask/unified_data_layer.hpp
 * 该层具有如下输出：
 * (1) top[0]: -> [N,3,H,W]
 * (2) top[1]: -> [1,1,Ng,8]
 * 每个gt-box使用8个数字进行标记 -> <bindex,cid,pid,xmin,ymin,xmax,ymax,is_diff>
 * 注意：该层主要读取VOC/COCO图像数据集。
 */

template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  /**
   * top[0]: -> [N,3,H,W]
   * top[1]: -> [1,1,Ng,8]
   * 8 -> <bindex,cid,pid,xmin,ymin,xmax,ymax,is_diff>
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  // 随机数
  shared_ptr<Caffe::RNG> prefetch_rng_;
  // 随机乱序输入样本序列
  virtual void ShuffleImages();

  // 加载一个minibatch
  virtual void load_batch(Batch<Dtype>* batch);

  // 样本对
  // <image_path, xml_path>
  vector<std::pair<std::string, std::string> > lines_;
  // 当前样本序号
  int lines_id_;

  // rsvd.
  string part_name_to_label_file_;
  string pose_name_to_label_file_;
  string dir_name_to_label_file_;
  map<string, int> part_name_to_label_;
  map<string, int> pose_name_to_label_;
  map<string, int> dir_name_to_label_;
  map<int, string> part_label_to_name_;
  map<int, string> pose_label_to_name_;
  map<int, string> dir_label_to_name_;

  // batch id (0,1,2,3,4,...)
  int batch_id_;

  // 随机采样器序列
  vector<BatchSampler> batch_samplers_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
