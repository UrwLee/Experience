#ifndef CAFFE_VISUALIZE_BOXPOSE_LAYER_HPP_
#define CAFFE_VISUALIZE_BOXPOSE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

/**
 * 该层提供了对box/pose骨架进行可视化的方法
 */

namespace caffe {

typedef VisualizeBoxposeParameter_BPDrawType BPDrawnType;

template <typename Dtype>
class VisualizeBoxposeLayer : public Layer<Dtype> {
 public:
  explicit VisualizeBoxposeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VisualizeBoxpose"; }

  /**
   * bottom[0] -> image　[1,3,H,W]
   * bottom[1] -> heatmap [1,52,RH,RW]
   * bottom[2] -> proposals [1,1,N,61]
   */
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * top[0]: 1
   * unused.
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  /**
   * 通过GPU绘制骨架
   * @param image      [图片数据]
   * @param w          [图片的width]
   * @param h          [图片的height]
   * @param proposals  [目标提议]
   * @param vec        [每个目标对象的每条线段的单位方向向量]
   * @param num_people [目标数]
   * @param threshold  [阈值,e.g.,0.05]
   */
  void render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* proposals, const Dtype* vec,
                   const int num_people, const Dtype threshold);

  /**
   * 通过GPU绘制关键点
   * @param image      [图像]
   * @param w          [图像width]
   * @param h          [图像height]
   * @param proposals  [目标提议]
   * @param num_people [目标数]
   * @param threshold  [阈值,e.g.,0.05]
   */
  void render_points_gpu(Dtype* image, const int w, const int h, const Dtype* proposals,
                   const int num_people, const Dtype threshold);

  /**
   * 通过GPU绘制heatmaps
   * 注意：通道数0-17
   * @param image    [图像]
   * @param w        [图像width]
   * @param h        [图像height]
   * @param heatmaps [heatmaps]
   * @param nw       [heatmaps的width]
   * @param nh       [heatmaps的height]
   */
  void render_heatmaps_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                   const int nw, const int nh);

  /**
   * 通过GPU绘制vecmaps
   * 注意：通道数18-51
   * @param image    [图像]
   * @param w        [图像width]
   * @param h        [图像height]
   * @param heatmaps [heatmaps]
   * @param nw       [heatmaps的width]
   * @param nh       [heatmaps的height]
   */
  void render_vecmaps_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps, const int nw, const int nh);

  /**
   * 将Blob数据转换为cv数据格式
   * 从Dtype(float) -> unsigned char (cv::Mat)
   * @param image [图像]
   * @param w     [图像width]
   * @param h     [图像height]
   * @param out   [cv数据指针]
   */
  void cv_inv(const Dtype* image, const int w, const int h, unsigned char* out);

  /**
   * 获取系统时间信息 (us)
   */
  double get_wall_time();

  /**
   * 绘制类型，参考proto定义
   */
  BPDrawnType drawtype_;
  // 置信度阈值
  Dtype pose_threshold_;
  // 是否保存输出可视化结果
  bool write_frames_;
  // 保存路径
  string output_directory_;
  // 是否可视化
  bool visualize_;
  // 是否在图像上显示置信度信息
  bool print_score_;
};

}  // namespace caffe

#endif
