#include "caffe/tracker/alov_loader.hpp"
#include "caffe/tracker/basic.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

using std::string;
using std::vector;
namespace bfs = boost::filesystem;

const bool kDoTest = false;

// 测试集的比例： 20%
const double val_ratio = 0.2;

template <typename Dtype>
ALOVLoader<Dtype>::ALOVLoader(const string& video_folder, const string& annotations_folder)
{
  if (!bfs::is_directory(annotations_folder)) {
    LOG(FATAL) << "Error - " << annotations_folder <<  " is not a valid directory.";
    return;
  }

  // 获取各个子集
  vector<string> categories;
  find_subfolders(annotations_folder, &categories);

  // 子集选取
  const int max_categories = kDoTest ? 3 : categories.size();

  for (int i = 0; i < max_categories; ++i) {
    Category<Dtype> category;
    // 获取子集的完整路径
    const string& category_name = categories[i];
    const string& category_path = annotations_folder + "/" + category_name;
    // 在子集下找到所有后缀为.ann的标注文件
    // 每个ann代表一个视频文件
    const boost::regex annotation_filter(".*\\.ann");
    vector<string> annotation_files;
    find_matching_files(category_path, annotation_filter, &annotation_files);
    // 遍历每个ann标注文件，每个对应一个视频video
    for (int j = 0; j < annotation_files.size(); ++j) {
      // 获得第j个标注文件的路径
      const string& annotation_file = annotation_files[j];
      // 为每个标注文件生成一个video类
      Video<Dtype> video;
      // Get the path to the video image files.
      // video_folder-category_name-annotation_file(除去.ann)构成了视频的完整路径
      const string video_path = video_folder + "/" + category_name + "/" +
          annotation_file.substr(0, annotation_file.length() - 4);
      // 定义视频路径
      video.path_ = video_path;
      // 在视频路径下找到所有的图片
      const boost::regex image_filter(".*\\.jpg");
      find_matching_files(video_path, image_filter, &video.all_frames_);
      // 打开标注文件，读入标注
      const string& annotation_file_path = category_path + "/" + annotation_file;
      FILE* annotation_file_ptr = fopen(annotation_file_path.c_str(), "r");
      int frame_num;
      double Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
      while (true) {
        // 第一个代表帧ID
        const int status = fscanf(annotation_file_ptr, "%d %lf %lf %lf %lf %lf %lf %lf %lf\n",
                     &frame_num, &Ax, &Ay, &Bx, &By, &Cx, &Cy, &Dx, &Dy);
        if (status == EOF) {
          break;
        }
        // 将标注转换为frame标记： frame_num/bbox
        Frame<Dtype> frame;
        frame.frame_num = frame_num - 1;
        BoundingBox<Dtype>& bbox = frame.bbox;
        bbox.x1_ = (Dtype)(std::min(Ax, std::min(Bx, std::min(Cx, Dx))) - 1);
        bbox.y1_ = (Dtype)(std::min(Ay, std::min(By, std::min(Cy, Dy))) - 1);
        bbox.x2_ = (Dtype)(std::max(Ax, std::max(Bx, std::max(Cx, Dx))) - 1);
        bbox.y2_ = (Dtype)(std::max(Ay, std::max(By, std::max(Cy, Dy))) - 1);
        video.annotations_.push_back(frame);
      }
      fclose(annotation_file_ptr);
      this->videos_.push_back(video);
      category.videos.push_back(video);
    }
    categories_.push_back(category);
  }
}

template <typename Dtype>
void ALOVLoader<Dtype>::get_videos(const bool get_train, std::vector<Video<Dtype> >* videos) const {
  for (int category_num = 0; category_num < categories_.size(); ++category_num) {
    // 取出子集
    const Category<Dtype>& category = categories_[category_num];
    int num_videos = category.videos.size();

    const int num_val = static_cast<int>(val_ratio * num_videos);

    const int num_train = num_videos - num_val;

    int start_num;
    int end_num;
    if (get_train) {
      start_num = 0;
      end_num = num_train - 1;
    } else {
      start_num = num_train;
      end_num = num_videos - 1;
    }

    // Add the appropriate videos from this category to the list of videos
    // to return.
    const std::vector<Video<Dtype> >& category_videos = category.videos;
    // 取出的视频列表
    for (int i = start_num; i <= end_num; ++i) {
      const Video<Dtype>& video = category_videos[i];
      videos->push_back(video);
    }
  }
  int num_annotations = 0;
  for (int i = 0; i < videos->size(); ++i) {
    const Video<Dtype>& video = (*videos)[i];
    num_annotations += video.annotations_.size();
  }
  LOG(INFO) << "Total: " << num_annotations << " annotated video frames are fetched.";
}

INSTANTIATE_CLASS(ALOVLoader);
}
