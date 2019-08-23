#include "caffe/tracker/video.hpp"

#include <string>
#include <vector>

namespace caffe {

using std::string;
using std::vector;

// 视频可视化: 对每一副图片,显示图片和gtbox
template <typename Dtype>
void Video<Dtype>::ShowVideo() const {
  // 视频路径
  const string& video_path = path_;
  // 图片名称集合
  const vector<string>& image_files = all_frames_;
  // 标注ID从0开始
  int annotated_frame_index = 0;
  // 第一帧和结尾帧的ID
  const int start_frame = annotations_[0].frame_num;
  const int end_frame = annotations_[annotations_.size() - 1].frame_num;

  // 遍历所有的帧ID
  for (int image_frame_num = start_frame; image_frame_num <= end_frame; ++image_frame_num) {
    // 加载图片
    const string& image_file = video_path + "/" + image_files[image_frame_num];
    cv::Mat image = cv::imread(image_file);

    // 获取标注的帧ID,必须与image_frame_num才行
    const int annotated_frame_num = annotations_[annotated_frame_index].frame_num;
    // 注意: 帧ID要和标注ID匹配
    if (annotated_frame_num == image_frame_num) {
      // 获取box,然后绘制box
      const BoundingBox<Dtype>& box = annotations_[annotated_frame_index].bbox;
      box.DrawBoundingBox(&image);
      // 下一帧
      if (annotated_frame_index < annotations_.size() - 1) {
        annotated_frame_index++;
      }
    }
    // 显示
    if(!image.data ) {
      LOG(FATAL) << "Could not open or find image: " << image_file;
    } else {
      cv::namedWindow( "VideoSeq", cv::WINDOW_AUTOSIZE );
      cv::imshow( "VideoSeq", image);
      cv::waitKey(1);
    }
  }
}

template <typename Dtype>
void Video<Dtype>::LoadFirstAnnotation(int* first_frame, cv::Mat* image,
                               BoundingBox<Dtype>* box) const {
  LoadAnnotation(0, first_frame, image, box);
}

template <typename Dtype>
void Video<Dtype>::LoadAnnotation(const int annotation_index,
                          int* frame_num,
                          cv::Mat* image,
                          BoundingBox<Dtype>* box) const {
  // 获取标注Frame
  const Frame<Dtype>& annotated_frame = annotations_[annotation_index];
  // 返回帧ID
  *frame_num = annotated_frame.frame_num;
  // 返回帧box
  *box = annotated_frame.bbox;
  // 获取路径
  const string& video_path = path_;
  const vector<string>& image_files = all_frames_;
  if (image_files.empty()) {
    LOG(FATAL) << "Error - no image files for video at path: " << video_path;
    return;
  } else if (*frame_num >= image_files.size()) {
    LOG(FATAL) << "Cannot find frame: " << *frame_num
               << "; only " << image_files.size()
               << " image files were found at " << video_path;
    return;
  }
  // 返回图像
  const string& image_file = video_path + "/" + image_files[*frame_num];
  *image = cv::imread(image_file);
  if (!image->data) {
    LOG(FATAL) << "Could not find file: " << image_file;
  }
}

template <typename Dtype>
bool Video<Dtype>::FindAnnotation(const int frame_num, BoundingBox<Dtype>* box) const {
  // 遍历所有标注
  for (int i = 0; i < annotations_.size(); ++i) {
    const Frame<Dtype>& frame = annotations_[i];
    if (frame.frame_num == frame_num) {
      *box = frame.bbox;
      return true;
    }
  }
  return false;
}

template <typename Dtype>
bool Video<Dtype>::LoadFrame(const int frame_num, const bool draw_bounding_box,
                     const bool load_only_annotation, cv::Mat* image,
                     BoundingBox<Dtype>* box) const {
  // 路径和图像名称
  const string& video_path = path_;
  const vector<string>& image_files = all_frames_;
  // 加载图像
  if (!load_only_annotation) {
    const string& image_file = video_path + "/" + image_files[frame_num];
    *image = cv::imread(image_file);
  }
  // 是否存在标记
  const bool has_annotation = FindAnnotation(frame_num, box);
  // 绘制box
  if (!load_only_annotation && has_annotation && draw_bounding_box) {
    box->DrawBoundingBox(image);
  }
  // 返回值: 是否找到标记
  return has_annotation;
}

INSTANTIATE_CLASS(Video);
}
