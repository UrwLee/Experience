#ifndef CAFFE_VISUAL_MASK_LAYER_HPP_
#define CAFFE_VISUAL_MASK_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

/**
 * 该层作为Mask的可视化层，它提供了如下方法：
 * １．可视化实例的boxes
 * ２．可视化实例的关节点
 * ３．可视化实例的Mask
 */

template <typename Dtype>
class VisualMaskLayer : public Layer<Dtype> {
 public:
  explicit VisualMaskLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VisualMask"; }

  /**
   * bottom[0]: -> image data (1,3,H,W)
   * bottom[1]: -> ROIs (1,1,Nroi,7)
   * bottom[2]: -> mask/kps (Nroi,1/18,RH,RW)
   * bottom[3]: -> kps[optional] (Nroi,18,RH,RW)
   * 注意：
   * （１）如果只可视化mask, bottom[3]不存在
   * （２）如果只可视化kps，bottom[3]不存在
   * （３）如果同时可视化mask/kps, bottom[2]->mask and bottom[3]->kps
   * （４）如果不需要可视化mask/kps，则bottom[2]/[3]均不存在
   */
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  /**
   * top[0] -> (1) ignored.
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  /**
   * 将Blob数据转换为cv::Mat (unsigned char型)数据
   * @param image [数据指针]
   * @param w     [image宽度]
   * @param h     [image高度]
   * @param out   [输出unsigned char型数据]
   */
  void cv_inv(const Dtype* image, const int w, const int h, unsigned char* out);

  // 获取时间　(us)
  double get_wall_time();

  // 关节点的置信度阈值
  Dtype kps_threshold_;
  // mask置信度阈值
  Dtype mask_threshold_;
  // 是否保存视频帧
  bool write_frames_;
  // 如果保存，指定输出路径
  string output_directory_;
  // 是否显示mask
  bool show_mask_;
  // 是否显示kps
  bool show_kps_;
  // 是否显示box置信度信息
  bool print_score_;
  // 可视化最大尺寸
  int max_dis_size_;
};

}

#endif
