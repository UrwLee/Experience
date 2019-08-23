#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>

#include "caffe/tracker/tracker_data_loader.hpp"

namespace caffe {

template<typename Dtype>
TrackerDataLoader<Dtype>::TrackerDataLoader(const TrackerDataLoaderParameter& param): param_(param) {
  // 获取数据集
  // 首先获取图片数据集
  CHECK_GE(param_.image_list_size(), 0);
  CHECK_GE(param_.image_folder_size(), 0);
  CHECK_EQ(param_.image_list_size(),param_.image_folder_size());
  for (int i = 0; i < param_.image_list_size(); ++i) {
    if (i == 0) {
      image_loader_.reset(new ImageLoader<Dtype>(param_.image_list(i),param_.image_folder(i)));
    } else {
      ImageLoader<Dtype> image_loader(param_.image_list(i),param_.image_folder(i));
      image_loader_->merge_from(&image_loader);
    }
  }
  LOG(INFO) << "Found " << image_loader_->get_image_size() << " Images, and "
            << image_loader_->get_anno_size() << " Annotations.";
  // 再获取视频数据集
  if ((param_.vot_type_folder_size() + param_.alov_type_image_folder_size()) == 0) {
    LOG(FATAL) << "Must define at least one video sequence.";
  }
  CHECK_EQ(param_.alov_type_image_folder_size(), param_.alov_type_anno_folder_size());
  // 首先加载VOT类型数据
  if (param_.vot_type_folder_size() > 0) {
    for (int i = 0; i < param_.vot_type_folder_size(); ++i) {
      if (i == 0) {
        video_loader_.reset(new VOTLoader<Dtype>(param_.vot_type_folder(i)));
      } else {
        VOTLoader<Dtype> vot_loader(param_.vot_type_folder(i));
        video_loader_->merge_from(&vot_loader);
      }
    }
  }
  // 再加载ALOV类型数据
  if (param_.vot_type_folder_size() > 0) {
    for (int i = 0; i < param_.alov_type_image_folder_size(); ++i) {
      ALOVLoader<Dtype> alov_loader(param_.alov_type_image_folder(i),
                                    param_.alov_type_anno_folder(i));
      video_loader_->merge_from(&alov_loader);
    }
  } else {
    video_loader_.reset(new ALOVLoader<Dtype>(param_.alov_type_image_folder(0),
                                              param_.alov_type_anno_folder(0)));
    for (int i = 1; i < param_.alov_type_image_folder_size(); ++i) {
      ALOVLoader<Dtype> alov_loader(param_.alov_type_image_folder(i),
                                    param_.alov_type_anno_folder(i));
      video_loader_->merge_from(&alov_loader);
    }
  }
  LOG(INFO) << "Found " << video_loader_->get_size() << " Videos.";
  // 所有数据全部加载完毕
  // 初始化样本生成器
  example_generator_.reset(new ExampleGenerator<Dtype>(param_.lambda_shift(),param_.lambda_scale(),
                          param_.lambda_min_scale(), param_.lambda_max_scale()));
  LOG(INFO) << "The TrackerDataLoader has been initialized.";
}

template<typename Dtype>
void TrackerDataLoader<Dtype>::Load(Dtype* transformed_data, Dtype* transformed_label) {
  // 增广后的图片和boxes
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> targets;
  std::vector<BoundingBox<Dtype> > bboxes_gt_scaled;
  // 获取增广图片和gtboxes
  for (int iter = 0 ; iter < param_.fetch_iters(); ++iter) {
    // 随机选取一张图片
    const std::vector<std::vector<SFrame<Dtype> > >& all_images = image_loader_->get_images();
    const int image_num = rand() % all_images.size();
    const std::vector<SFrame<Dtype> >& image_annotations = all_images[image_num];
    const int annotation_num = rand() % image_annotations.size();
    cv::Mat sel_image;
    BoundingBox<Dtype> sel_bbox;
    image_loader_->LoadAnnotation(image_num, annotation_num, &sel_image, &sel_bbox);
    example_generator_->Reset(sel_bbox,sel_bbox,sel_image,sel_image);
    MakeExamples(param_.generated_examples_per_image(),&images,&targets,&bboxes_gt_scaled);
    // 随机选取一个视频序列
    const std::vector<Video<Dtype> >& videos = video_loader_->get_videos();
    const int video_num = rand() % videos.size();
    const Video<Dtype>& video = videos[video_num];
    const std::vector<Frame<Dtype> >& video_annotations = video.annotations_;
    if (video_annotations.size() < 2) {
      LOG(FATAL) << "Error - video " << video.path_ << " has only " << video_annotations.size() << " annotations";
      return;
    }
    const int annotation_index = rand() % (video_annotations.size() - 1);
    int frame_num_prev;
    cv::Mat image_prev;
    BoundingBox<Dtype> bbox_prev;
    video.LoadAnnotation(annotation_index, &frame_num_prev, &image_prev, &bbox_prev);
    int frame_num_curr;
    cv::Mat image_curr;
    BoundingBox<Dtype> bbox_curr;
    video.LoadAnnotation(annotation_index + 1, &frame_num_curr, &image_curr, &bbox_curr);
    example_generator_->Reset(bbox_prev,bbox_curr,image_prev,image_curr);
    MakeExamples(param_.generated_examples_per_frame(),&images,&targets,&bboxes_gt_scaled);
  }
  // 将获取的图片和gtboxes写入到输出结果中
  CHECK_EQ(images.size(), targets.size());
  CHECK_EQ(images.size(), bboxes_gt_scaled.size());
  CHECK_EQ(images.size(), param_.batch_size());
  // 依次写batch_size次循环
  for (int n = 0; n < param_.batch_size(); ++n) {
    cv::Mat& prev = targets[n];
    cv::Mat& curr = images[n];
    BoundingBox<Dtype>& gtbox = bboxes_gt_scaled[n];
    // resized
    cv::Mat prev_resized, curr_resized;
    cv::resize(prev, prev_resized, Size(param_.resized_width(), param_.resized_height()));
    cv::resize(curr, curr_resized, Size(param_.resized_width(), param_.resized_height()));
    CHECK_EQ(prev_resized.channels(),3);
    CHECK_EQ(curr_resized.channels(),3);
    const int offset = prev_resized.rows * prev_resized.cols;
    const int offset3 = 3 * offset;
    const int half_offs = param_.batch_size() * offset3;
    // 减去均值 & normal
    for (int i = 0; i < prev_resized.rows; ++i) {
      for (int j = 0; j < prev_resized.cols; ++j) {
        Vec3b& rgb_prev = prev_resized.at<Vec3b>(i, j);
        Vec3b& rgb_curr = curr_resized.at<Vec3b>(i, j);
        if (param_.normalize()) {
          // prev
          transformed_data[n * offset3 + i * prev_resized.cols + j] = (rgb_prev[0] - 128)/256.0;
          transformed_data[n * offset3 + offset + i * prev_resized.cols + j] = (rgb_prev[1] - 128)/256.0;
          transformed_data[n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_prev[2] - 128)/256.0;
          // curr
          transformed_data[half_offs + n * offset3 + i * prev_resized.cols + j] = (rgb_curr[0] - 128)/256.0;
          transformed_data[half_offs + n * offset3 + offset + i * prev_resized.cols + j] = (rgb_curr[1] - 128)/256.0;
          transformed_data[half_offs + n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_curr[2] - 128)/256.0;
        } else {
          // we use means to substract
          CHECK_EQ(param_.mean_value_size(), 3) << "Must provide mean_values, and mean_value should have length of 3.";
          // prev
          transformed_data[n * offset3 + i * prev_resized.cols + j] = (rgb_prev[0] - param_.mean_value(0));
          transformed_data[n * offset3 + offset + i * prev_resized.cols + j] = (rgb_prev[1] - param_.mean_value(1));
          transformed_data[n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_prev[2] - param_.mean_value(2));
          // curr
          transformed_data[half_offs + n * offset3 + i * prev_resized.cols + j] = (rgb_curr[0] - param_.mean_value(0));
          transformed_data[half_offs + n * offset3 + offset + i * prev_resized.cols + j] = (rgb_curr[1] - param_.mean_value(1));
          transformed_data[half_offs + n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_curr[2] - param_.mean_value(2));
        }
      }
    }
    // 写gtbox的值
    std::vector<Dtype> bbox_vect;
    gtbox.GetVector(&bbox_vect);
    transformed_label[n*4] = bbox_vect[0];
    transformed_label[n*4+1] = bbox_vect[1];
    transformed_label[n*4+2] = bbox_vect[2];
    transformed_label[n*4+3] = bbox_vect[3];
  }
  // end of load
}

template<typename Dtype>
void TrackerDataLoader<Dtype>::MakeExamples(const int num_generated_examples,
                                            std::vector<cv::Mat>* images,
                                            std::vector<cv::Mat>* targets,
                                            std::vector<BoundingBox<Dtype> >* bboxes_gt_scaled) {
  cv::Mat image;
  cv::Mat target;
  BoundingBox<Dtype> bbox_gt_scaled;
  example_generator_->MakeTrueExample(&image, &target, &bbox_gt_scaled);
  images->push_back(image);
  targets->push_back(target);
  bboxes_gt_scaled->push_back(bbox_gt_scaled);
  example_generator_->MakeTrainingExamples(num_generated_examples - 1, images,
                                         targets, bboxes_gt_scaled);
}

INSTANTIATE_CLASS(TrackerDataLoader);

}  // namespace caffe
