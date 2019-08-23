#ifndef __BBOX_HPP_
#define __BBOX_HPP_

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

using namespace std;

class BBox {
  public:
    BBox() {};
    BBox(float x1, float y1, float x2, float y2): x1_(x1), y1_(y1), x2_(x2), y2_(y2) {}
    float width() const { return (x2_ - x1_); }
    float height() const { return (y2_ - y1_); }
    float xmin() const { return x1_; }
    float xmax() const { return x2_; }
    float ymin() const { return y1_; }
    float ymax() const { return y2_; }
    void set_xmin(float x) { x1_ = x; };
    void set_ymin(float x) { y1_ = x; };
    void set_xmax(float x) { x2_ = x; };
    void set_ymax(float x) { y2_ = x; };
    float xcenter() const { return (x1_ + width() / 2.); }
    float ycenter() const { return (y1_ + height() / 2.); }
    float area() const { return width() * height(); }
    void clip() {
      x1_ = min(max(x1_, float(0)), float(1));
      y1_ = min(max(y1_, float(0)), float(1));
      x2_ = min(max(x2_, float(0)), float(1));
      y2_ = min(max(y2_, float(0)), float(1));
    }
    void norm(int cols, int rows) {
       x1_ = x1_ / (float)cols;
       x2_ = x2_ / (float)cols;
       y1_ = y1_ / (float)rows;
       y2_ = y2_ / (float)rows;
    }
    void flip() {
      float t = x1_;
      x1_ = 1.0 - x2_;
      x2_ = 1.0 - t;
    }
    void draw(const int r, const int g, const int b, cv::Mat* image) {
      clip();
      const cv::Point point1(x1_ * image->cols, y1_ * image->rows);
      const cv::Point point2(x2_ * image->cols, y2_ * image->rows);
      const cv::Scalar box_color(b, g, r);
      const int thickness = 2;
      cv::rectangle(*image, point1, point2, box_color, thickness);
    }
    void draw(cv::Mat* image) {
      draw(0,255,0,image);
    }
    float intersection(const BBox& roi) const {
      const float area = std::max((float)0, std::min(x2_, roi.xmax()) - std::max(x1_, roi.xmin())) * std::max((float)0, std::min(y2_, roi.ymax()) - std::max(y1_, roi.ymin()));
      return area;
    }
    float iou(const BBox& roi) const {
      float inter = intersection(roi);
      float area_this = area();
      float area_that = roi.area();
      float t = inter / (area_this + area_that - inter);
      return std::max(t, (float)0);
    }
    float coverage(const BBox& roi) const {
      float inter = intersection(roi);
      float area_this = area();
      float t = inter / area_this;
      t = std::max(t, (float)0);
      return t;
    }
    float coverage_other(BBox& roi) const {
      float inter = intersection(roi);
      float area_that = roi.area();
      float t = inter / area_that;
      t = std::max(t, (float)0);
      return t;
    }
    float project(const BBox& roi, BBox* proj_roi) {
      if (x1_ >= roi.xmax() || x2_ <= roi.xmin() || y1_ >= roi.ymax() || y2_ <= roi.ymin()) {return 0;}
      proj_roi->set_xmin((x1_ - roi.xmin()) / roi.width());
      proj_roi->set_xmax((x2_ - roi.xmin()) / roi.width());
      proj_roi->set_ymin((y1_ - roi.ymin()) / roi.height());
      proj_roi->set_ymax((y2_ - roi.ymin()) / roi.height());
      if (proj_roi->area() <= 0) return 0;
      return coverage(roi);
    }
    float project(const BBox& roi) {
      if (x1_ >= roi.xmax() || x2_ <= roi.xmin() || y1_ >= roi.ymax() || y2_ <= roi.ymin()) {return 0;}
      float t = coverage(roi);
      x1_ = (x1_ - roi.xmin()) / roi.width();
      x2_ = (x2_ - roi.xmin()) / roi.width();
      y1_ = (y1_ - roi.ymin()) / roi.height();
      y2_ = (y2_ - roi.ymin()) / roi.height();
      if (area() <= 0) return 0;
      return t;
    }

  private:
    float x1_, y1_, x2_, y2_;
};

#endif
