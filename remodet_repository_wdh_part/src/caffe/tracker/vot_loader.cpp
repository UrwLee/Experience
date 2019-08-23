#include "caffe/tracker/vot_loader.hpp"
#include "caffe/tracker/basic.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

using std::string;
using std::vector;
namespace bfs = boost::filesystem;

template <typename Dtype>
VOTLoader<Dtype>::VOTLoader(const std::string& vot_folder)
{
  if (!bfs::is_directory(vot_folder)) {
    LOG(FATAL) << "Error - " << vot_folder <<  " is not a valid directory.";
    return;
  }
  vector<string> videos;
  // 获取所有子目录,每个子目录代表一个视频序列
  find_subfolders(vot_folder, &videos);
  LOG(INFO) << "Found " << videos.size() << " videos.";
  for (int i = 0; i < videos.size(); ++i) {
    const string& video_name = videos[i];
    const string& video_path = vot_folder + "/" + video_name;
    LOG(INFO) << "Loading video: " << video_name;

    // 生成视频对象
    Video<Dtype> video;
    video.path_ = video_path;
    // 遍历该目录下的所有jpg文件
    const boost::regex image_filter(".*\\.jpg");
    find_matching_files(video_path, image_filter, &video.all_frames_);
    // 获得gtbox文件
    const string& bbox_groundtruth_path = video_path + "/groundtruth.txt";
    // 打开该文件
    FILE* bbox_groundtruth_file_ptr = fopen(bbox_groundtruth_path.c_str(), "r");
    // 帧ID=0
    int frame_num = 0;
    double Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
    // 逐行读取
    while (true) {
      // 每行有8个数据
      const int status = fscanf(bbox_groundtruth_file_ptr, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                   &Ax, &Ay, &Bx, &By, &Cx, &Cy, &Dx, &Dy);
      if (status == EOF) {
        break;
      }
      // 定义Frame标记
      Frame<Dtype> frame;
      // 获取帧ID
      frame.frame_num = frame_num++;
      BoundingBox<Dtype>& bbox = frame.bbox;
      bbox.x1_ = (Dtype)(std::min(Ax, std::min(Bx, std::min(Cx, Dx))) - 1);
      bbox.y1_ = (Dtype)(std::min(Ay, std::min(By, std::min(Cy, Dy))) - 1);
      bbox.x2_ = (Dtype)(std::max(Ax, std::max(Bx, std::max(Cx, Dx))) - 1);
      bbox.y2_ = (Dtype)(std::max(Ay, std::max(By, std::max(Cy, Dy))) - 1);
      video.annotations_.push_back(frame);
    }
    fclose(bbox_groundtruth_file_ptr);
    this->videos_.push_back(video);
  }
}

INSTANTIATE_CLASS(VOTLoader);
}
