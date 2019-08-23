#ifndef CAFFE_REMO_VISUALIZER_H
#define CAFFE_REMO_VISUALIZER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/remo/res_frame.hpp"

namespace caffe {

template <typename Dtype>
class Visualizer {

  /**
   * 该类提供了对一副图像进行各种可视化绘制的方法，包括：
   * １．绘制BBOX (包括ID/相似度/置信度/...等各类信息，可选)
   * ２．绘制骨架
   * ３．绘制Heatmaps
   * ４．绘制Vecmaps
   */

public:

  /**
   * 构造方法
   * image -> 输入图像
   * max_display_size -> 最大可视化尺寸
   */
  Visualizer(cv::Mat& image, int max_display_size);

  /**
   * 复位：重新使用一副图像进行初始化
   * @param image [输入图像]
   */
  void Init(cv::Mat& image) { image_ = image; }

  /**
   * 获取图像
   * @return [图像cv::Mat]
   */
  cv::Mat get_image() { return image_; }

  /**
   * 绘制Box: 绘制某一个box对象，及ID，并返回绘制后的图像
   * @param bbox      [待绘制的box]
   * @param id        [该box的id, 如果id == -1，则不绘制id]
   * @param out_image [绘制好的图像cv::Mat]
   */
  void draw_bbox(const BoundingBox<Dtype>& bbox, const int id, cv::Mat* out_image);

  /**
   * 绘制Box: 绘制某一个box对象，及ID，并返回绘制后的图像
   * @param r         [绘制线条颜色R]
   * @param g         [绘制线条颜色G]
   * @param b         [绘制线条颜色B]
   * @param bbox      [待绘制的box]
   * @param id        [该box的id, 如果id == -1，则不绘制id]
   * @param out_image [绘制好的图像cv::Mat]
   */
  void draw_bbox(int r, int g, int b, const BoundingBox<Dtype>& bbox, const int id, cv::Mat* out_image);

  /**
   * 绘制Box: 直接在原图上绘制某一个box对象，及ID
   * @param bbox [待绘制的box]
   * @param id   [该box的id, 如果id == -1，则不绘制id]
   */
  void draw_bbox(const BoundingBox<Dtype>& bbox, const int id);

  /**
   * 绘制Box: 直接在原图上绘制某一个box对象，及ID
   * @param r    [绘制线条颜色R]
   * @param g    [绘制线条颜色G]
   * @param b    [绘制线条颜色B]
   * @param bbox [待绘制的box]
   * @param id   [该box的id, 如果id == -1，则不绘制id]
   */
  void draw_bbox(int r, int g, int b, const BoundingBox<Dtype>& bbox, const int id);

  /**
   * 绘制Hands: 基于该帧的目标对象数据结构，绘制所有的右手，并返回绘制后的图像
   * @param meta  [目标对象的数据结构]
   * @param image [绘制后的图像cv::Mat]
   */
  void draw_hand(const PMeta<Dtype>& meta, cv::Mat* image);

  /**
   * 绘制Box: 基于该帧的目标对象数据结构，绘制Box，并返回绘制后的图像
   * @param meta      [目标对象的数据结构]
   * @param out_image [绘制后的图像cv::Mat]
   */
  void draw_bbox(const vector<PMeta<Dtype> >& meta, cv::Mat* out_image);

  /**
   * 绘制Box: 基于该帧的目标对象数据结构，绘制Box，并返回绘制后的图像
   * @param r          [绘制线条颜色R]
   * @param g          [绘制线条颜色G]
   * @param b          [绘制线条颜色B]
   * @param draw_hands [是否绘制手部box]
   * @param meta       [目标对象的数据结构]
   * @param out_image  [绘制后的图像cv::Mat]
   */
  void draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta, cv::Mat* out_image);

  /**
   * 绘制Box: 直接在原图上基于该帧的目标对象数据结构，绘制Box
   * @param meta [目标对象的数据结构]
   */
  void draw_bbox(const vector<PMeta<Dtype> >& meta);

  /**
   * 绘制Box: 直接在原图上基于该帧的目标对象数据结构，绘制Box
   * @param r          [绘制线条颜色R]
   * @param g          [绘制线条颜色G]
   * @param b          [绘制线条颜色B]
   * @param draw_hands [是否绘制手部box]
   * @param meta       [目标对象的数据结构]
   */
  void draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta);

  /**
   * 绘制Box: 在指定的图像上基于提供的目标对象数据结构，绘制Box，并将绘制后的图像返回
   * @param meta      [目标对象的数据结构]
   * @param src_image [输入的图像cv::Mat]
   * @param out_image [绘制后的图像cv::Mat]
   */
  void draw_bbox(const vector<PMeta<Dtype> >& meta, const cv::Mat& src_image, cv::Mat* out_image);

  /**
   * 绘制Box: 在指定的图像上基于提供的目标对象数据结构，绘制Box，并将绘制后的图像返回
   * @param r          [绘制线条颜色R]
   * @param g          [绘制线条颜色G]
   * @param b          [绘制线条颜色B]
   * @param draw_hands [是否绘制手部box]
   * @param meta       [目标对象的数据结构]
   * @param src_image  [输入的图像cv::Mat]
   * @param out_image  [绘制后的图像cv::Mat]
   */
  void draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta, const cv::Mat& src_image, cv::Mat* out_image);

  /**
   * 绘制骨架：在原图上基于给定的目标对象数据结构信息，绘制骨架，并将绘制后的图像返回
   * @param meta      [目标对象的数据结构]
   * @param out_image [绘制后的图像cv::Mat]
   */
  void draw_skeleton(const vector<PMeta<Dtype> >& meta, cv::Mat* out_image);

  /**
   * 绘制骨架：直接在原图上基于给定的目标对象数据结构信息，绘制骨架
   * @param meta [目标对象的数据结构]
   */
  void draw_skeleton(const vector<PMeta<Dtype> >& meta);

  /**
   * 绘制vecmaps: 在原图上基于给定的heatmaps和heatmaps的长宽进行绘制，然后将绘制的图像返回
   * 通道号：　18-51
   * @param heatmaps   [heatmaps数据指针]
   * @param map_width  [heatmaps的宽度]
   * @param map_height [heatmaps的高度]
   * @param out_image  [绘制后的图像cv::Mat]
   */
  void draw_vecmap(const Dtype* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);

  /**
   * 绘制vecmaps: 直接在原图上基于给定的heatmaps和heatmaps的长宽进行绘制
   * 通道号：　18-51
   * @param heatmaps   [heatmaps数据指针]
   * @param map_width  [heatmaps的宽度]
   * @param map_height [heatmaps的高度]
   */
  void draw_vecmap(const Dtype* heatmaps, const int map_width, const int map_height);

  /**
   * 绘制heatmaps: 在原图上基于给定的heatmaps和heatmaps的长宽进行绘制，然后将绘制的图像返回
   * 通道号：　0-17
   * @param heatmaps   [heatmaps数据指针]
   * @param map_width  [heatmaps的宽度]
   * @param map_height [heatmaps的高度]
   * @param out_image  [绘制后的图像cv::Mat]
   */
  void draw_heatmap(const Dtype* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);

  /**
   * 绘制heatmaps: 直接在原图上基于给定的heatmaps和heatmaps的长宽进行绘制
   * 通道号：　0-17
   * @param heatmaps   [heatmaps数据指针]
   * @param map_width  [heatmaps的宽度]
   * @param map_height [heatmaps的高度]
   */
  void draw_heatmap(const Dtype* heatmaps, const int map_width, const int map_height);

  /**
   * 保存：将指定图像按照指定ID保存到指定目录
   * @param image    [待保存图像]
   * @param save_dir [保存目录]
   * @param image_id [保存ID]
   */
  void save(const cv::Mat& image, const std::string& save_dir, const int image_id);

  /**
   * 保存：将原图像按照指定ID保存到指定目录
   * @param save_dir [保存目录]
   * @param image_id [保存ID]
   */
  void save(const std::string& save_dir, const int image_id);

protected:
  /**
   * 使用GPU绘制vecmaps
   * @param image    [图像原始数据指针]
   * @param w        [图像的宽]
   * @param h        [图像的高]
   * @param heatmaps [heatmaps数据指针]
   * @param nw       [heatmaps的宽]
   * @param nh       [heatmaps的高]
   * 注意：通道号18-51
   */
  void render_vecmaps_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps, const int nw, const int nh);

  /**
   * 使用GPU绘制heatmaps
   * @param image    [图像原始数据指针]
   * @param w        [图像的宽]
   * @param h        [图像的高]
   * @param heatmaps [heatmaps数据指针]
   * @param nw       [heatmaps的宽]
   * @param nh       [heatmaps的高]
   * 注意：通道号0-17
   */
  void render_heatmaps_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps, const int nw, const int nh);

  /**
   * 使用GPU绘制17条线段
   * @param image      [图像原始数据指针]
   * @param w          [图像的宽]
   * @param h          [图像的高]
   * @param proposals  [目标对象的数据指针]
   * @param vec        [每个对象的每条线段的单位向量(dx,dy)，用于定位待绘制的点的区域，满足椭圆方程约束]
   * @param num_people [目标对象的数量]
   * @param threshold  [关节点置信度阈值]
   */
  void render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* proposals, const Dtype* vec, const int num_people, const Dtype threshold);

  /**
   * 使用GPU绘制18个点
   * @param image      [图像原始数据指针]
   * @param w          [图像的宽]
   * @param h          [图像的高]
   * @param proposals  [目标对象的数据指针]
   * @param num_people [目标对象的数量]
   * @param threshold  [关节点置信度阈值]
   */
  void render_points_gpu(Dtype* image, const int w, const int h, const Dtype* proposals, const int num_people, const Dtype threshold);

  /**
   * 将cv::Mat的数据格式转换为Blob（稠密数组型）格式
   * @param image [待转换的cv::Mat]
   * @param data  [转换后的Blob数据指针]
   */
  void cv_to_blob(const cv::Mat& image, Dtype* data);

  /**
   * 将Blob（稠密数组型）格式转换为cv::Mat格式
   * @param image  [Blob型数据指针]
   * @param w      [图像的宽]
   * @param h      [图像的高]
   * @param cv_img [转换后的cv::Mat型指针]
   */
  void blob_to_cv(const Dtype* image, const int w, const int h, cv::Mat* cv_img);

  // 原始图像：　基于原始输入图像使用max_dis_size进行resized后的图像
  // 后续所有绘制方法都基于这个图像，用于降低绘制工作的复杂度
  // max_dis_size越小，绘制工作量越小
  cv::Mat image_;
};

}

#endif
