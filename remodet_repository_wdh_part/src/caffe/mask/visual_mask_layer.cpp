#include <string>
#include <vector>
#include <utility>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <csignal>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/mask/visual_mask_layer.hpp"
#include "caffe/util/myimg_proc.hpp"

namespace caffe {
// used for bbox & mask for different person
static const int COLOR_MAPS[18] = {255,85,0,255,170,130,70,255,0,170,0,130,85,255,0,0,255,85};
// used for keypoints (17 Limbs definition)
static const int LIMB_COCO[34] = {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17};
// used for limbs & keypoints
static const int LIMB_COLOR_MAPS[54] = {255,0,0,255,85,0,255,170,0,255,255,0,170,255,0,85,255,0,0,255,0,0,255,85,0,255,170,0,255, \
                                        255,0,170,255,0,85,255,0,0,255,85,0,255,170,0,255,255,0,255,255,0,170,255,0,85};

template <typename Dtype>
double VisualMaskLayer<Dtype>::get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time,NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

template <typename Dtype>
void VisualMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const VisualMaskParameter &visual_mask_param =
      this->layer_param_.visual_mask_param();
  // thresholod
  kps_threshold_ = visual_mask_param.kps_threshold();
  mask_threshold_ = visual_mask_param.mask_threshold();
  // output saving
  write_frames_ = visual_mask_param.write_frames();
  output_directory_ = visual_mask_param.output_directory();
  // show parts saving
  show_mask_ = visual_mask_param.show_mask();
  show_kps_ = visual_mask_param.show_kps();
  // print score
  print_score_ = visual_mask_param.print_score();
  max_dis_size_ = visual_mask_param.max_dis_size();
}

template <typename Dtype>
void VisualMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // bottom[0]: image
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 3);
  // bottom[1]: ROIs
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 7);
  // bottom[2]: kps or mask
  if (show_mask_ && (!show_kps_)) {
    CHECK_EQ(bottom.size(), 3);
    // [N,1,H,W]
    CHECK_EQ(bottom[2]->num(), bottom[1]->height());
    CHECK_EQ(bottom[2]->channels(), 1);
  } else if ((!show_mask_) && show_kps_) {
    CHECK_EQ(bottom.size(), 3);
    // [1,N,18,3]
    CHECK_EQ(bottom[2]->channels(), bottom[1]->height());
    CHECK_EQ(bottom[2]->height(), 18);
    CHECK_EQ(bottom[2]->width(), 3);
  } else if (show_kps_ && show_mask_) {
    CHECK_EQ(bottom.size(), 4);
    // first: mask
    CHECK_EQ(bottom[2]->num(), bottom[1]->height());
    CHECK_EQ(bottom[2]->channels(), 1);
    // second: kps
    CHECK_EQ(bottom[3]->channels(), bottom[1]->height());
    CHECK_EQ(bottom[3]->height(), 18);
    CHECK_EQ(bottom[3]->width(), 3);
  } else {
    CHECK_EQ(bottom.size(), 2);
  }
  // not used
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void VisualMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // ###########################################################################
  // Get Input images
  const Dtype* image_data = bottom[0]->cpu_data();
  cv::Mat image, dis_image;
  blobTocvImage(image_data, bottom[0]->height(), bottom[0]->width(), bottom[0]->channels(), &image);
  const int width = image.cols;
  const int height = image.rows;
  const int maxLen = (width > height) ? width : height;
  const Dtype ratio = (Dtype)max_dis_size_ / maxLen;
  const int display_width = static_cast<int>(width * ratio);
  const int display_height = static_cast<int>(height * ratio);
  cv::resize(image, dis_image, cv::Size(display_width, display_height), cv::INTER_LINEAR);
  // ###########################################################################
  // Draw
  const Dtype* roi_data = bottom[1]->cpu_data();
  int num = bottom[1]->height();
  for (int n = 0; n < num; ++n) {
    // =========================================================================
    // BBOX
    int r = COLOR_MAPS[3*(n % 6)];
    int g = COLOR_MAPS[3*(n % 6) + 1];
    int b = COLOR_MAPS[3*(n % 6) + 2];
    BoundingBox<Dtype> bbox;
    bbox.x1_ = (int)(roi_data[n*7+3] * display_width);
    bbox.y1_ = (int)(roi_data[n*7+4] * display_height);
    bbox.x2_ = (int)(roi_data[n*7+5] * display_width);
    bbox.y2_ = (int)(roi_data[n*7+6] * display_height);
    // Draw BBOX
    bbox.Draw(r, g, b, &dis_image);
    if (print_score_) {
      char score_str[256];
      snprintf(score_str, 256, "%.3f", roi_data[n*7+2]);
      cv::putText(dis_image, score_str, cv::Point(bbox.x1_ + 3, bbox.y1_ + 3),
          cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);
    }
    // =========================================================================
    // MASK
    if (show_mask_) {
      const Dtype* mask_data = bottom[2]->cpu_data() + bottom[2]->offset(n);
      const int mask_height = bottom[2]->height();
      const int mask_width = bottom[2]->width();
      // 转换为cv
      cv::Mat mask_image = cv::Mat::zeros(mask_height,mask_width,CV_8UC1);
      for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
          mask_image.at<uchar>(y,x) = static_cast<uchar>(mask_data[y*mask_width+x] * Dtype(255));
        }
      }
      // Resize to ROI scale
      cv::Rect roi(bbox.x1_, bbox.y1_, bbox.get_width(), bbox.get_height());
      cv::Mat roi_image = dis_image(roi);
      cv::Mat mask_resized;
      cv::resize(mask_image, mask_resized, cv::Size(roi_image.cols, roi_image.rows), cv::INTER_LINEAR);
      // modify Mask
      Dtype alpha = 0.4;
      for (int y = 0; y < roi_image.rows; ++y) {
        for (int x = 0; x < roi_image.cols; ++x) {
          cv::Vec3b& rgb = roi_image.at<cv::Vec3b>(y, x);
          int mask_val = mask_resized.at<uchar>(y,x);
          if (Dtype(mask_val) / 255. > mask_threshold_) {
            rgb[0] = (1-alpha)*rgb[0] + alpha*b;
            rgb[1] = (1-alpha)*rgb[1] + alpha*g;
            rgb[2] = (1-alpha)*rgb[2] + alpha*r;
          }
        }
      }
    }
    // =========================================================================
    // Keypoints
    if (show_kps_) {
      const Dtype* kps_data = NULL;
      if (show_mask_) {
        kps_data = bottom[3]->cpu_data() + 54 * n;
      } else {
        kps_data = bottom[2]->cpu_data() + 54 * n;
      }
      // draw points
      for (int p = 0; p < 18; ++p) {
        Dtype v = kps_data[3 * p + 2];
        Dtype x = kps_data[3 * p];
        Dtype y = kps_data[3 * p + 1];
        if (v > kps_threshold_) {
          x = x * bbox.get_width() + bbox.x1_;
          y = y * bbox.get_height() + bbox.y1_;
          x = std::min(std::max(x, Dtype(0)), Dtype(dis_image.cols - 1));
          y = std::min(std::max(y, Dtype(0)), Dtype(dis_image.rows - 1));
          cv::Point2f pf;
          pf.x = x;
          pf.y = y;
          cv::circle(dis_image, pf, 2, CV_RGB(LIMB_COLOR_MAPS[3*p],LIMB_COLOR_MAPS[3*p+1],LIMB_COLOR_MAPS[3*p+2]), -1);
        }
      }
      // draw limbs
      for (int l = 0; l < 17; ++l) {
        int pA = LIMB_COCO[2*l];
        int pB = LIMB_COCO[2*l+1];
        Dtype xA = kps_data[3*pA];
        Dtype yA = kps_data[3*pA+1];
        Dtype vA = kps_data[3*pA+2];
        Dtype xB = kps_data[3*pB];
        Dtype yB = kps_data[3*pB+1];
        Dtype vB = kps_data[3*pB+2];
        if (vA > kps_threshold_ && vB > kps_threshold_) {
          xA = xA * bbox.get_width() + bbox.x1_;
          yA = yA * bbox.get_height() + bbox.y1_;
          xA = std::min(std::max(xA, Dtype(0)), Dtype(dis_image.cols - 1));
          yA = std::min(std::max(yA, Dtype(0)), Dtype(dis_image.rows - 1));
          xB = xB * bbox.get_width() + bbox.x1_;
          yB = yB * bbox.get_height() + bbox.y1_;
          xB = std::min(std::max(xB, Dtype(0)), Dtype(dis_image.cols - 1));
          yB = std::min(std::max(yB, Dtype(0)), Dtype(dis_image.rows - 1));
          cv::Point2f pfA, pfB;
          pfA.x = xA;
          pfA.y = yA;
          pfB.x = xB;
          pfB.y = yB;
          cv::line(dis_image,pfA,pfB,CV_RGB(LIMB_COLOR_MAPS[3*l],LIMB_COLOR_MAPS[3*l+1],LIMB_COLOR_MAPS[3*l+2]),2);
        }
      }
    }
  }
  // saving & fps
  static int counter = 1;
  static double last_time = get_wall_time();
  static double this_time = last_time;
  static float fps = 1.0;
  if (write_frames_) {
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(98);
    char fname[256];
    sprintf(fname, "%s/frame%06d.jpg", output_directory_.c_str(), counter);
    cv::imwrite(fname, dis_image, compression_params);
  }
  if (counter % 30 == 0) {
    this_time = get_wall_time();
    fps = (float)30 / (float)(this_time - last_time);
    last_time = this_time;
    std::cout << "Frame ID: " << counter << std::endl;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
              << "FPS: " << fps << std::endl;
  }
  counter++;
  // wait for key-process
  cv::imshow("remo", dis_image);
  if (cv::waitKey(1) == 27) {
    raise(SIGINT);
  }
  // end
  top[0]->mutable_cpu_data()[0] = 0;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(VisualMaskLayer, Forward);
#endif

INSTANTIATE_CLASS(VisualMaskLayer);
REGISTER_LAYER_CLASS(VisualMask);

} // namespace caffe
