#ifndef CAFFE_REMO_BASIC_H
#define CAFFE_REMO_BASIC_H

#include <string>
#include <iostream>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/basic.hpp"

namespace caffe {

/**
 * 该头文件定义了目标对象的数据结构
 */

/**
 * 关节点定义：
 * x/y: 位置
 * v: 置信度
 */
template <typename Dtype>
struct KPoint {
  Dtype x;
  Dtype y;
  Dtype v;
};

/**
 * 一个Person对象的数据结构
 */
template <typename Dtype>
struct PMeta {
  // ID
  int id;
  // similarity：　当前对象与它的历史对象之间的相似度，该值应该越大越好
  Dtype similarity;
  // 当前对象与背景对象【被遮挡的对象】之间的最大相似度，理论上该值越小越好，说明更容易区分
  Dtype max_back_similarity;
  // bbox: 该目标的位置
  BoundingBox<Dtype> bbox;
  // 18个关节点的定义
  vector<KPoint<Dtype> > kps;
  // 可见关节点数量
  int num_points;
  // 该对象的置信度
  Dtype score;
};

/**
 * Define the drawn method.
 */
enum DrawnMode {
  // 无可视化
  NONE,
  // 只绘制BOX
  BOX,
  // 绘制BOX/ID
  BOX_ID,
  // 只绘制骨架
  SKELETON,
  // 绘制骨架/BOX
  SKELETON_BOX,
  // 绘制骨架/BOX/ID
  SKELETON_BOX_ID,
  // 只绘制热点图
  HEATMAP,
  // 绘制热点图/BOX
  HEATMAP_BOX,
  // 绘制热点图/BOX/ID
  HEATMAP_BOX_ID,
  // 只绘制Limb图
  VECMAP,
  // 绘制Limb图/BOX
  VECMAP_BOX,
  // 绘制Limb图/BOX/ID
  VECMAP_BOX_ID
};

/**
 * 将Blobs输出转换为PMeta格式的数据结构
 * @param proposals [网络输出的Proposals结果]
 * @param num       [proposals的数量]
 * @param meta      [输出的PMeta数据结构]
 * 实现方法：　遍历每个元素，保存到对应的数据元素中即可
 */
template <typename Dtype>
void transfer_meta(const Dtype* proposals, const int num, std::vector<PMeta<Dtype> >* meta);

}

#endif
