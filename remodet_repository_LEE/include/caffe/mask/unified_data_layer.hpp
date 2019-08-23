#ifndef CAFFE_MASK_UNIFIED_DATA_LAYER_HPP_
#define CAFFE_MASK_UNIFIED_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/mask/unified_data_transformer.hpp"
#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

/**
 * 该层将提供一个minibatch的样本。
 * 包括：
 * １．图像数据
 * ２．所有标注信息
 */

template <typename Dtype>
class UnifiedDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit UnifiedDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), unified_data_transform_param_(param.unified_data_transform_param()) {}
  virtual ~UnifiedDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UnifiedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

  /**
   * top[0]: -> Image data (N,3,H,W)
   * top[1]: -> Labels for each Gt-Boxes  (1,1,Ng,66+H*W)
   *            <bindex,cid,pid,is_diff,iscrowd,xmin,ymin,xmax,ymax,has_kps,num_kps,18*3,has_mask,H*W>
   */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  //  prefetch随机数
  shared_ptr<Caffe::RNG> prefetch_rng_;

  // 输入样本顺序随机乱序
  virtual void ShuffleLists();

  // 载入一个minibatch
  virtual void load_batch(Batch<Dtype>* batch);

  /**
   * 从指定XML加载标注数据
   * @param bindex   [minibatch编号]
   * @param xml_file [XML文件]
   * @param root_dir [图像根目录]
   * @param anno     [返回标注数据结构]
   */
  void ReadAnnoDataFromXml(const int bindex, const string& xml_file, const string& root_dir,
                           AnnoData<Dtype>* anno);
  void ReadPartBoxesFromXml(const int bindex, const string& xml_file, const string& root_dir,
                           vector<LabeledBBox<Dtype> >* labeled_boxes);
  // 所有XML文件的集合列表
  vector<std::string> lines_;

  // 当前加载的XML编号
  int lines_id_;
  // unused.
  Blob<Dtype> transformed_label_;

  // 数据转换器参数
  UnifiedTransformationParameter unified_data_transform_param_;

  // 数据转换器
  shared_ptr<UnifiedDataTransformer<Dtype> > unified_data_transformer_;

  // RGB颜色通道均值
  vector<Dtype> mean_values_;
  // 是否加载parts
  bool add_parts_;
  std::string parts_xml_dir_;

  // 是否加载kps
  bool add_kps_;
  // 是否加载mask
  bool add_mask_;

  // top_offs
  int top_offs_;
};

}

#endif
