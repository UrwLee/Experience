#include "caffe/hp/hp_net.hpp"

namespace caffe {

static const float kContextFactor = 1.7;

HPNetWrapper::HPNetWrapper(const std::string& proto, const std::string& model) {
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  net_.reset(new caffe::Net<float>(proto, caffe::TEST));
  net_->CopyTrainedLayersFrom(model);
}

// 提取patch
void HPNetWrapper::getCropPatch(const cv::Mat& image, const BoundingBox<float>& roi, cv::Mat* patch) {
  // 在原始图像中的裁剪位置
  float x1_r, y1_r, w_r, h_r;
  // 在patch中的复制位置
  float x1_a, y1_a, w_a, h_a;
  // 待裁剪的位置
  float x1, y1, w, h;
  // 初始位置
  x1 = roi.x1_ * image.cols;
  y1 = roi.y1_ * image.rows;
  w = roi.get_width() * image.cols;
  h = roi.get_height() * image.rows;
  // (1) 将矩形框变为正方形
  if (w > h) {
    y1 -= (w - h) / 2;
    h = w;
  } else {
    x1 -= (h - w) / 2;
    w = h;
  }
  // (2) 获取裁剪的位置
  x1 = x1 - w / 2 * (kContextFactor - 1);
  y1 = y1 - h / 2 * (kContextFactor - 1);
  w = w * kContextFactor;
  h = h * kContextFactor;
  // (3) 原始图像和patch内的坐标赋初始值．默认都是整个patch全部覆盖
  x1_r = x1;
  y1_r = y1;
  w_r = w;
  h_r = h;
  x1_a = 0;
  y1_a = 0;
  w_a = w;
  h_a = h;
  // (4) 修正位置
  if (x1 < 0) {
    x1_r = 0;   // 从0开始裁剪
    w_r += x1;  // 裁剪的宽度同步减少
    x1_a -= x1; // patch内的坐标增大　
    w_a += x1;  // patch内的宽度同步减少
  }
  if (y1 < 0) { // 同步x1进行操作
    y1_r = 0;
    h_r += y1;
    y1_a -= y1;
    h_a += y1;
  }
  if ((y1 + h) > image.rows) {  // 越过最大值范围
    h_r -= (y1 + h) - image.rows;  // 范围缩短
    h_a = h_r;                     // patch内的范围同步缩短
  }
  if ((x1 + w) > image.cols) { // 同y方向越界操作
    w_r -= (x1 + w) - image.cols;
    w_a = w_r;
  }
  // (5) 裁剪
  cv::Mat crop_patch(w, h, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat area = crop_patch(cv::Rect(x1_a, y1_a, w_a, h_a));
  cv::Rect orig_patch(x1_r, y1_r, w_r, h_r);
  image(orig_patch).copyTo(area);
  // (6) 输出
  *patch = crop_patch;
}

// 加载单张数据
void HPNetWrapper::load(const cv::Mat& image) {
  caffe::Blob<float>* inputBlob = net_->input_blobs()[0];
  int width = inputBlob->width();
  int height = inputBlob->height();
  cv::Mat rsz_image;
  cv::resize(image, rsz_image, cv::Size(width, height), cv::INTER_LINEAR);
  float* data = inputBlob->mutable_cpu_data();
  // 加载数据
  const int offs = height * width;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const cv::Vec3b& rgb = rsz_image.at<cv::Vec3b>(i, j);
      data[           i * width + j] = rgb[0] - 104;
      data[    offs + i * width + j] = rgb[1] - 117;
      data[2 * offs + i * width + j] = rgb[2] - 123;
    }
  }
}

int HPNetWrapper::hpmode(const cv::Mat& image, const BoundingBox<float>& roi, float* score) {
  cv::Mat patch;
  getCropPatch(image, roi, &patch);
  load(patch);
  const std::vector<caffe::Blob<float>* >& outputBlobs = net_->Forward();
  caffe::Blob<float>* outputBlob = outputBlobs[0];
  const float* data = outputBlob->cpu_data();
  int max_id = 0;
  float max_score = 0;
  for (int i = 0; i < outputBlob->count(); ++i) {
    if (data[i] > max_score) {
      max_score = data[i];
      max_id = i;
    }
  }
  *score = max_score;
  return max_id;
}

}
