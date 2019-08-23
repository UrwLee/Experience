#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "caffe/tracker/example_generator.hpp"
#include "caffe/tracker/image_loader.hpp"
#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/bounding_box.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>

namespace caffe {

using std::vector;
using std::string;
using namespace boost::property_tree;
namespace bfs = boost::filesystem;

const bool kDoTest = false;

const double kMaxRatio = 0.66;

template <typename Dtype>
ImageLoader<Dtype>::ImageLoader(const std::string& image_list,
                                const std::string& image_folder) {
  // 获取所有的list
  LOG(INFO) << "Opening file " << image_list;
  std::ifstream infile(image_list.c_str());
  std::string filename;
  std::string labelname;
  // 所有图片的文件信息
  vector<std::pair<std::string, std::string> > lines;
  while (infile >> filename >> labelname) {
    lines.push_back(std::make_pair(filename, labelname));
  }
  CHECK(!lines.empty()) << "File is empty.";
  // 开始遍历每一行记录
  int num_annotations = 0;
  for (int i = 0; i < lines.size(); ++i) {
    // 图片的完整路径
    string image_path = image_folder + "/" + lines[i].first;
    // xml的完整路径
    string xml_path = image_folder + "/" + lines[i].second;
    vector<SFrame<Dtype> > annotations;
    // 获取所有标注
    LoadAnnotationFromXmlFile(xml_path,image_path,&annotations);
    if (annotations.size() == 0) continue;
    num_annotations += annotations.size();
    annotations_.push_back(annotations);
  }
  LOG(INFO) << "Found " << num_annotations << " annotations from "
            << lines.size() << " images.";
}

template <typename Dtype>
void ImageLoader<Dtype>::LoadAnnotationFromXmlFile(const string& annotation_file, const string& image_path,
                                                   vector<SFrame<Dtype> >* image_annotations) {
  image_annotations->clear();
  ptree pt;
  read_xml(annotation_file, pt);
  // 获取图片的尺寸
  int height = pt.get<int>("annotation.size.height");
  int width = pt.get<int>("annotation.size.width");
  CHECK(width != 0 && height != 0) << annotation_file <<
    ": no valid image width/height/channels.";
  // 遍历所有子节点
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    // 查找到person的信息
    if (v1.first == "object") {
      ptree object = v1.second;
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        if (v2.first == "bndbox") {
          ptree pt2 = v2.second;
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          LOG_IF(WARNING, xmin > width) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << annotation_file <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << annotation_file <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << annotation_file <<
              " bounding box irregular.";
          Dtype box_width = xmax - xmin;
          Dtype box_height = ymax - ymin;
          if (box_width > kMaxRatio * width && box_height > kMaxRatio * height) {
            continue;
          }
          SFrame<Dtype> annotation;
          annotation.image_path = image_path;
          annotation.width = width;
          annotation.height = height;
          annotation.bbox.x1_ = xmin;
          annotation.bbox.x2_ = xmax;
          annotation.bbox.y1_ = ymin;
          annotation.bbox.y2_ = ymax;
          image_annotations->push_back(annotation);
        }
      }
    }
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::ShowImages() const {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    LoadImage(i, &image);
    cv::namedWindow("Imageshow", cv::WINDOW_AUTOSIZE);
    cv::imshow("Imageshow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::ComputeStatistics() const {
  bool first_time = true;
  Dtype min_width;
  Dtype min_height;
  Dtype max_width;
  Dtype max_height;

  Dtype mean_width = 0;
  Dtype mean_height = 0;

  Dtype min_width_frac;
  Dtype min_height_frac;
  Dtype max_width_frac;
  Dtype max_height_frac;

  Dtype mean_width_frac = 0;
  Dtype mean_height_frac = 0;

  int n = 0;
  for (int i = 0; i < annotations_.size(); ++i) {
    const std::vector<SFrame<Dtype> >& annotations = annotations_[i];
    for (int j = 0; j < annotations.size(); ++j) {
      const SFrame<Dtype>& annotation = annotations[j];
      const Dtype width = annotation.bbox.get_width();
      const Dtype height = annotation.bbox.get_height();
      const Dtype image_width = annotation.width;
      const Dtype image_height = annotation.height;
      const Dtype width_frac = width / image_width;
      const Dtype height_frac = height / image_height;
      if (first_time) {
        min_width = width;
        min_height = height;
        max_width = width;
        max_height = height;
        min_width_frac = width_frac;
        min_height_frac = height_frac;
        max_width_frac = width_frac;
        max_height_frac = height_frac;
        first_time = false;
      } else {
        min_width = std::min(min_width, width);
        min_height = std::min(min_height, height);
        max_width = std::max(max_width, width);
        max_height = std::max(max_height, height);

        min_width_frac = std::min(min_width_frac, width_frac);
        min_height_frac = std::min(min_height_frac, height_frac);
        max_width_frac = std::max(max_width_frac, width_frac);
        max_height_frac = std::max(max_height_frac, height_frac);
      }
      // Update
      mean_width = (static_cast<Dtype>(n) * mean_width + width) / static_cast<Dtype>(n + 1);
      mean_height = (static_cast<Dtype>(n) * mean_height + height) / static_cast<Dtype>(n + 1);
      mean_width_frac = (static_cast<Dtype>(n) * mean_width_frac + width_frac) / static_cast<Dtype>(n + 1);
      mean_height_frac = (static_cast<Dtype>(n) * mean_height_frac + height_frac) / static_cast<Dtype>(n + 1);
    }
    n++;
  }
  // Print the image statistics for this dataset.
  LOG(INFO) << "Width: " << min_width << ", " << max_width << ", " << mean_width;
  LOG(INFO) << "Height: " << min_height << ", " << max_height << ", " << mean_height;
  LOG(INFO) << "Width frac: " << min_width_frac << ", " << max_width_frac << ", " << mean_width_frac;
  LOG(INFO) << "Height frac: " << min_height_frac << ", " << max_height_frac << ", " << mean_height_frac;
  LOG(INFO) << "Total: " << n << " annotations.";
}

template <typename Dtype>
void ImageLoader<Dtype>::ShowAnnotations() const {
  for (int i = 0; i < annotations_.size(); ++i) {
    const std::vector<SFrame<Dtype> >& annotations = annotations_[i];
    cv::Mat image_all;
    LoadImage(i, &image_all);
    for (int j = 0; j < annotations.size(); ++j) {
      cv::Mat image;
      BoundingBox<Dtype> bbox;
      LoadAnnotation(i, j, &image, &bbox);
      LOG(INFO) << "Width: " << bbox.get_width() << ", "
                << "Height: " << bbox.get_height();
      bbox.DrawBoundingBox(&image_all);
    }
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);
    cv::imshow("ImageShow", image_all);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::LoadImage(const int image_num,
                                  cv::Mat* image) const {
  const std::vector<SFrame<Dtype> >& annotations = annotations_[image_num];
  const SFrame<Dtype>& annotation = annotations[0];
  const string& image_file = annotation.image_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image " << image_file;
    return;
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::LoadAnnotation(const int image_num,
                                       const int annotation_num,
                                       cv::Mat* image,
                                       BoundingBox<Dtype>* bbox) const {
  const std::vector<SFrame<Dtype> >& annotations = annotations_[image_num];
  const SFrame<Dtype>& annotation = annotations[annotation_num];

  const string& image_file = annotation.image_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image " << image_file;
    return;
  }
  *bbox = annotation.bbox;
}

template <typename Dtype>
void ImageLoader<Dtype>::ShowAnnotationsRand() const {
  while (true) {
    const int image_num = rand() % annotations_.size();
    const std::vector<SFrame<Dtype> >& annotations = annotations_[image_num];
    const int annotation_num = rand() % annotations.size();
    // Load the image and annotation.
    cv::Mat image;
    BoundingBox<Dtype> bbox;
    // 加载图片及标记
    LoadAnnotation(image_num, annotation_num, &image, &bbox);
    cv::Mat image_copy;
    image.copyTo(image_copy);
    bbox.DrawBoundingBox(&image_copy);
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow("ImageShow", image_copy);                   // Show our image inside it.
    cv::waitKey(0);                                          // Wait for a keystroke in the window
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::ShowAnnotationsShift() const {
  // 创建样本生成器
  ExampleGenerator<Dtype> example_generator((Dtype)5, (Dtype)5, (Dtype)(-0.4), (Dtype)0.4);
  for (int i = 0; i < annotations_.size(); ++i) {
    const std::vector<SFrame<Dtype> >& annotations = annotations_[i];
    for (int j = 0; j < annotations.size(); ++j) {
      cv::Mat image;
      BoundingBox<Dtype> bbox;
      LoadAnnotation(i, j, &image, &bbox);
      cv::Mat image_copy;
      image.copyTo(image_copy);
      bbox.Draw(0, 255, 0, &image_copy);
      cv::namedWindow("Imageshow", cv::WINDOW_AUTOSIZE);
      cv::imshow("Imageshow", image_copy);

      example_generator.Reset(bbox, bbox, image, image);
      example_generator.set_indices(i, j);

      cv::Mat image_rand_focus;
      cv::Mat target_pad;
      BoundingBox<Dtype> bbox_gt_scaled;
      // 样本生成后,显示
      const bool visualize = true;
      // 创建一个样本
      const int kNumShifts = 1;
      for (int k = 0; k < kNumShifts; ++k) {
        example_generator.MakeTrainingExampleBBShift(visualize, &image_rand_focus,
                                                     &target_pad, &bbox_gt_scaled);
      }
    }
  }
}

template <typename Dtype>
void ImageLoader<Dtype>::merge_from(ImageLoader<Dtype>* dst) {
  const std::vector<std::vector<SFrame<Dtype> > >& dst_annos = dst->get_images();
  if (dst_annos.size() == 0) return;
  for (int i = 0; i < dst_annos.size(); ++i) {
    annotations_.push_back(dst_annos[i]);
  }
  LOG(INFO) << "Add " << dst_annos.size() << " Images.";
}

template <typename Dtype>
int ImageLoader<Dtype>::get_anno_size() {
  int num_anno = 0;
  for (int i = 0; i < annotations_.size(); ++i) {
    num_anno += annotations_[i].size();
  }
  return num_anno;
}

INSTANTIATE_CLASS(ImageLoader);
}
