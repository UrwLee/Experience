#ifndef CAFFE_VISUALIZE_POSE_LAYER_HPP_
#define CAFFE_VISUALIZE_POSE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

/**
 * 该层提供了对骨架进行可视化的方法。
 */

namespace caffe {

template <typename Dtype>
class VisualizeposeLayer : public Layer<Dtype> {
 public:
  explicit VisualizeposeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Visualizepose"; }

  /**
   * bottom[0] -> image  [1,3,H,W]
   * bottom[1] -> heatmap [1,52,RH,RW]
   * bottom[2] -> proposals [1,N,18+1,3]
   * 3: -> <x,y,v> 归一化坐标和置信度
   * 1: 目标对象的统计信息，例如可见点数，综合置信度，平均置信度
   */
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * top[0]: -> (1) unused.
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
   * 使用GPU绘制骨架
   * @param image      [图像]
   * @param w          [图像width]
   * @param h          [图像height]
   * @param poses      [proposals]
   * @param vec        [每个对象每个线段的单位方向向量]
   * @param num_people [目标数]
   * @param threshold  [阈值]
   * @param num_parts  [18]
   */
  void render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* poses, const Dtype* vec,
                   const int num_people, const Dtype threshold, const int num_parts);

  /**
   * 通过GPU绘制heatmap: 单个关节点的map
   * @param image     [图像]
   * @param w         [图像width]
   * @param h         [图像height]
   * @param heatmaps  [heatmaps] (0-17 channels)
   * @param nw        [heatmaps的width]
   * @param nh        [heatmaps的height]
   * @param num_parts [18]
   * @param part      [需要绘制的id, from 0-17]
   */
  void render_heatmap_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                   const int nw, const int nh, const int num_parts, const int part);

  /**
   * 通过GPU绘制vecmap: 单条线段的map
   * @param image     [图像]
   * @param w         [图像width]
   * @param h         [图像height]
   * @param vecmap    [vecmaps] (18-51 channels)
   * @param nw        [heatmaps的width]
   * @param nh        [heatmaps的height]
   * @param num_limbs [17]
   * @param channel   [需要绘制的线段号，from 0-16]
   */
  void render_vecmap_gpu(Dtype* image, const int w, const int h, const Dtype* vecmap,
                   const int nw, const int nh, const int num_limbs, const int channel);

  /**
   * 从某个关节点开始绘制所有的关节点heatmaps (use GPU)
   * @param image     [图像]
   * @param w         [图像width]
   * @param h         [图像height]
   * @param heatmaps  [heatmaps] (0-17 channels)
   * @param nw        [heatmaps的width]
   * @param nh        [heatmaps的height]
   * @param num_parts [18]
   * @param from_part [从哪个id开始绘制？　0 -> 绘制所有的关节点]
   */
  void render_heatmaps_from_id_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                   const int nw, const int nh, const int num_parts, const int from_part);

  /**
   * 从某条线段开始绘制所有的vecmaps (use GPU)
   * @param image        [图像]
   * @param w            [图像width]
   * @param h            [图像height]
   * @param vecmaps      [vecmaps] (18-51 channels)
   * @param nw           [heatmaps的width]
   * @param nh           [heatmaps的height]
   * @param channel_from [从哪个id开始绘制？ 0 -> 绘制所有的线段]
   */
  void render_vecmaps_from_id_gpu(Dtype* image, const int w, const int h, const Dtype* vecmaps,
                                             const int nw, const int nh, const int channel_from);

  /**
   * 通过CPU绘制骨架
   * @param image      [图像]
   * @param w          [图像width]
   * @param h          [图像height]
   * @param poses      [proposals]
   * @param num_people [目标数]
   * @param threshold  [阈值]
   * @param num_parts  [18]
   */
  void render_pose_cpu(Dtype* image, const int w, const int h, const Dtype* poses,
                   const int num_people, const Dtype threshold, const int num_parts);

  /**
   * 通过CPU绘制某个关节点的heatmap
   * @param image     [图像]
   * @param w         [图像width]
   * @param h         [图像height]
   * @param heatmaps  [heatmaps] (0-17 channels)
   * @param nw        [heatmaps的width]
   * @param nh        [heatmaps的height]
   * @param num_parts [18]
   * @param part      [需要绘制的id, from 0-17]
   */
  void render_heatmap_cpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                   const int nw, const int nh, const int num_parts, const int part);

  /**
   * 从某个关节点开始绘制所有的关节点heatmaps (use CPU)
   * @param image     [图像]
   * @param w         [图像width]
   * @param h         [图像height]
   * @param heatmaps  [heatmaps] (0-17 channels)
   * @param nw        [heatmaps的width]
   * @param nh        [heatmaps的height]
   * @param num_parts [18]
   * @param from_part [从哪个id开始绘制？　0 -> 绘制所有的关节点]
   */
  void render_heatmaps_from_id_cpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                   const int nw, const int nh, const int num_parts, const int from_part);

  /**
   * 将Blob的数据转换为cv::Mat数据　(unsigned char)
   * @param image [图像数据Blob]
   * @param w     [图像width]
   * @param h     [图像height]
   * @param out   [cv::Mat数据格式, unsigned char]
   */
  void cv_inv(const Dtype* image, const int w, const int h, unsigned char* out);

  /**
   * 获取系统时间
   */
  double get_wall_time();

  // type: COCO
  bool is_type_coco_;
  // 18 & 17
  int num_parts_;
  int num_limbs_;
  // 绘制模式，参考proto定义
  DrawType drawtype_;

  // 绘制的关节点id
  int part_id_;
  // 从哪一个关节点开始绘制？
  int from_part_;
  // 绘制的线段id
  int vec_id_;
  // 从哪一条线段开始绘制？
  int from_vec_;
  // 阈值
  Dtype pose_threshold_;
  // 是否输出可视化图像
  bool write_frames_;
  // 输出保存路径
  string output_directory_;

  // 是否可视化
  bool visualize_;
  // 是否绘制骨架
  bool draw_skeleton_;
  // 是否打印置信度信息
  bool print_score_;
};

}

#endif
