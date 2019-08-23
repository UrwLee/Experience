#ifndef CAFFE_TRACKER_IMAGE_LOADER_H
#define CAFFE_TRACKER_IMAGE_LOADER_H

#include "caffe/tracker/bounding_box.hpp"
#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 该类提供了对图片数据集的加载功能。
 */

// 实例标注
template <typename Dtype>
struct SFrame {
  // 实例所在的图片路径
  std::string image_path;
  // 实例的位置
  BoundingBox<Dtype> bbox;
  // 图片的长和宽
  int width;
  int height;
};

/**
 * 图片加载类
 */
template <typename Dtype>
class ImageLoader {
public:

  /**
   * image_list: -> 图片列表文件
   * image_folder: -> 图片根目录
   */
  ImageLoader(const std::string& image_list,
              const std::string& image_folder);

  /**
   * 图片加载
   * @param image_num [图片id]
   * @param image     [图片cv::Mat]
   */
  void LoadImage(const int image_num, cv::Mat* image) const;

  /**
   * 加载标注
   * @param image_num      [图片id]
   * @param annotation_num [该图片中的实例id]
   * @param image          [返回图片cv::Mat]
   * @param bbox           [返回box]
   */
  void LoadAnnotation(const int image_num,
                      const int annotation_num,
                      cv::Mat* image,
                      BoundingBox<Dtype>* bbox) const;

  /**
   * 图片显示
   */
  void ShowImages() const;

  /**
   * 显示图片和Box
   */
  void ShowAnnotations() const;

  /**
   * 随机显示图片和box
   */
  void ShowAnnotationsRand() const;

  /**
   * 显示随机增广的图片Patch
   */
  void ShowAnnotationsShift() const;

  /**
   * 目标对象的统计值
   */
  void ComputeStatistics() const;

  /**
   * 返回所有标注记录集合
   */
  const std::vector<std::vector<SFrame<Dtype> > >& get_images() {
    return annotations_;
  }

  /**
   * 加载第三方数据集标注
   * @param dst [第三方Loader]
   */
  void merge_from(ImageLoader<Dtype>* dst);

  /**
   * 获取标注记录数量
   * @return [数量]
   */
  int get_image_size() { return annotations_.size(); }

  /**
   * 获取标注实例总数
   * @return [数量]
   */
  int get_anno_size();

private:

  /**
   * 从XML文件中加载标注
   * @param annotation_file   [XML文件]
   * @param image_path        [图像根目录]
   * @param image_annotations [返回标注实例集合]
   */
  void LoadAnnotationFromXmlFile(const std::string& annotation_file,
                                 const std::string& image_path,
                                 std::vector<SFrame<Dtype> >* image_annotations);
  // 所有图片的标记记录
  // vector<SFrame>: -> 单张图片的实例集合
  // vector<vector<SFrame> >: -> 所有图片的实例集合
  std::vector<std::vector<SFrame<Dtype> > > annotations_;
};

}

#endif
