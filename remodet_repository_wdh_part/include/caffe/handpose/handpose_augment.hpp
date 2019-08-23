#ifndef __HANDPOSE_AUGMENT_HPP_
#define __HANDPOSE_AUGMENT_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "gflags/gflags.h"
#include "glog/logging.h"
//
#include "caffe/util/im_transforms.hpp"
#include "caffe/caffe.hpp"
#include <boost/shared_ptr.hpp>

#include "caffe/handpose/handpose_instance.hpp"
#include "caffe/handpose/bbox.hpp"
using namespace std;

/**
 * 对实例进行增广：
 * (1) 随机裁剪
 *     @ 最大裁剪以及最小裁剪范围，随机选择
 * (2) 随机翻转
 *     @ 随机左右翻转
 * (3) 随机颜色失真
 *     @ 随机颜色处理
 * (4) Resize
 *     @ 固定尺寸
 */
namespace caffe {

class HandPoseAugmenter {
  public:
    HandPoseAugmenter(const bool do_flip, const float flip_prob, const int resized_width,
                  const int resized_height,const bool save, const string& save_path,
                  const DistortionParameter& param,const float bbox_extend_min,const float bbox_extend_max, 
                  const float rotate_angle, const bool clip,const bool flag_augIntrain);
    // 增广接口
    // anno : -> 标注实例
    // image: -> 增广图像
    // id : -> 实例的ID
    // phase: -> 阶段
    void aug(HandPoseInstance& anno, cv::Mat* image, int* id, caffe::Phase phase);

  private:
    bool do_flip_;            //flip
    float flip_prob_;
    int resized_width_;       // resized
    int resized_height_;
    string save_path_;        // save
    bool save_;
    float bbox_extend_min_;
    float bbox_extend_max_;
    float rotate_angle_;
    bool clip_;
    bool flag_augIntrain_;
    DistortionParameter param_;  // color distortion (no-used)
};
}
#endif
