#ifndef CAFFE_TRACKER_VIDEO_H
#define CAFFE_TRACKER_VIDEO_H

#include "caffe/tracker/bounding_box.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 视频对象的单帧标注
 */
template <typename Dtype>
struct Frame {
  // 帧ID
  int frame_num;
  // 位置
  BoundingBox<Dtype> bbox;
};

/**
 * 视频类：　提供了视频序列数据的所有信息
 * 注意：请参考ALOV/VOT的视频序列的图片帧／标注文件的形式。
 * 该类用于描述ALOV/VOT序列数据的标注及图像集合。
 */
template <typename Dtype>
class Video {
public:
  /**
   * 加载标注
   * @param annotation_index [标注ID]
   * @param frame_num        [帧ID]
   * @param image            [图片]
   * @param box              [位置]
   */
  void LoadAnnotation(const int annotation_index, int* frame_num, cv::Mat* image,
                     BoundingBox<Dtype>* box) const;

  /**
   * 加载第一帧
   * @param first_frame [第一帧的ID]
   * @param image       [图像]
   * @param box         [位置]
   */
  void LoadFirstAnnotation(int* first_frame, cv::Mat* image,
                          BoundingBox<Dtype>* box) const;

  // Load指定帧ID的内容,返回image/box
  // draw_bounding_box -> 是否可视化
  // load_only_annotation -> 是否只加载标注
  bool LoadFrame(const int frame_num,
                const bool draw_bounding_box,
                const bool load_only_annotation,
                cv::Mat* image,
                BoundingBox<Dtype>* box) const;

  // 可视化标注
  void ShowVideo() const;

  // 视频目录地址
  std::string path_;

  // 该视频序列中的所有图片名称
  std::vector<std::string> all_frames_;

  // 对应所有的标注
  std::vector<Frame<Dtype> > annotations_;

private:
  bool FindAnnotation(const int frame_num, BoundingBox<Dtype>* box) const;
};

// 视频类别集合
// 该数据结构包含了一系列相同类别的视频集合
template <typename Dtype>
struct Category {
  std::vector<Video<Dtype> > videos;
};

}

#endif
