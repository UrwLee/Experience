#ifndef CAFFE_REMO_RES_FRAME_H
#define CAFFE_REMO_RES_FRAME_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/basic.hpp"

#include "caffe/remo/basic.hpp"
#include "caffe/remo/visualizer.hpp"

namespace caffe {

template <typename Dtype>
class ResFrame {
public:

  /**
   * 该类将根据图像的目标结构化数据，在图像上提供各种方式的绘制API.
   * 同时，也可以在图像上根据heat/vec-maps进行绘制
   */

  /**
   * 构造方法：
   * image -> 帧原始图像
   * id -> 帧ID
   * max_dis_size -> 最大可视化尺寸
   * meta -> 目标对象的结构化数据
   */
  ResFrame(const cv::Mat& image, const int id, const int max_dis_size,
           const vector<PMeta<Dtype> >& meta);

  /**
   * 返回原始图像
   * @return [原始图像cv::Mat]
   */
  cv::Mat& get_image() { return image_; };

  /**
   * 返回帧ID
   * @return [ID]
   */
  int get_id() { return id_; }

  /**
   * 数据结构转换：将数据结构转换为vector<>形式
   * @param meta [转换后的数据对象指针]
   */
  void get_meta(std::vector<std::vector<Dtype> >* meta);

  /**
   * 绘制Vecmap：调用visualizer在图像上绘制vecmap
   * @param  heatmaps [heatmaps的数据指针] (注意：heatmaps和vecmaps已经合并为heatmaps，heatmaps对应0-17号用到，vecmaps对应18-51号通道)
   * @param  width    [heatmaps的宽]
   * @param  height   [heatmaps的高]
   * @return          [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_vecmap(const Dtype* heatmaps, const int width, const int height);

  /**
   * 绘制Vecmap：调用visualizer在图像上绘制vecmap
   * @param  heatmaps  [heatmaps的数据指针] (注意：heatmaps和vecmaps已经合并为heatmaps，heatmaps对应0-17号用到，vecmaps对应18-51号通道)
   * @param  width     [heatmaps的宽]
   * @param  height    [heatmaps的高]
   * @param  show_bbox [是否显示BOX]
   * @param  show_id   [是否显示ID]
   * @return           [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_vecmap(const Dtype* heatmaps, const int width, const int height,
                            const bool show_bbox, const bool show_id);

  /**
   * 绘制Heatmap：调用visualizer在图像上绘制heatmap
   * @param  heatmaps [heatmaps的数据指针] (注意：heatmaps和vecmaps已经合并为heatmaps，heatmaps对应0-17号用到，vecmaps对应18-51号通道)
   * @param  width    [heatmaps的宽]
   * @param  height   [heatmaps的高]
   * @return          [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_heatmap(const Dtype* heatmaps, const int width, const int height);

  /**
   * 绘制Heatmap：调用visualizer在图像上绘制heatmap
   * @param  heatmaps  [heatmaps的数据指针] (注意：heatmaps和vecmaps已经合并为heatmaps，heatmaps对应0-17号用到，vecmaps对应18-51号通道)
   * @param  width     [heatmaps的宽]
   * @param  height    [heatmaps的高]
   * @param  show_bbox [是否显示BOX]
   * @param  show_id   [是否显示ID]
   * @return           [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_heatmap(const Dtype* heatmaps, const int width, const int height,
                             const bool show_bbox, const bool show_id);

  /**
   * 绘制Box：调用visualizer在图像上绘制Box
   * @param  show_id [是否显示ID]
   * @return         [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_bbox(const bool show_id);

  /**
   * 绘制骨架：调用visualizer在图像上绘制骨架
   * @return [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_skeleton();

  /**
   * 绘制骨架：调用visualizer在图像上绘制骨架
   * @param  show_bbox [是否显示BOX]
   * @param  show_id   [是否显示ID]
   * @return           [绘制好的cv::Mat]
   */
  cv::Mat get_drawn_skeleton(const bool show_bbox, const bool show_id);

  /**
   * 绘制骨架：调用visualizer在图像上绘制骨架
   * @return [绘制好的cv::Mat]
   */
  cv::Mat get_drawn();

protected:
  // 帧ID
  int id_;
  // 原始图像
  cv::Mat image_;
  // 目标对象的数据结构
  vector<PMeta<Dtype> > meta_;
  // 最大可视化尺寸
  int max_dis_size_;
};

}

#endif
