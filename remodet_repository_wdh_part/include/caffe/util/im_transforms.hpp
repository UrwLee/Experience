#ifdef USE_OPENCV
#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Generate random number given the probablities for each number.
int roll_weighted_die(const std::vector<float>& probabilities);

template <typename T>
bool is_border(const cv::Mat& edge, T color);

// Auto cropping image.
template <typename T>
cv::Rect CropMask(const cv::Mat& src, T point, int padding = 2);

cv::Mat colorReduce(const cv::Mat& image, int div = 64);

void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut);

void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img);

cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type = cv::BORDER_CONSTANT,
                                  const cv::Scalar pad = cv::Scalar(0, 0, 0),
                                  const int interp_mode = cv::INTER_LINEAR);

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width, const int new_height,
                                   const int interp_mode = cv::INTER_LINEAR);

void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image);

void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox);

cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param);

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param);

cv::Mat ApplyCrop(const cv::Mat& in_img, const RandomCropParameter& param, const Phase& phase, NormalizedBBox* crop_bbox);

cv::Mat ApplyDistorted(const cv::Mat& in_img, const DistoredParameter& param);
void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta);

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper);

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img);

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper);

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img); 

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta);

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob);

cv::Mat DistortImage(const cv::Mat& in_img, const DistortionParameter& param);
cv::Mat ExpandImage(const cv::Mat& in_img, const ExpansionParameter& param,
                    NormalizedBBox* expand_bbox);
cv::Mat ApplyCrop(const cv::Mat& in_img, const NormalizedBBox& crop_bbox);

// 光晕
template <typename Dtype> 
void linSpace(Dtype x1, Dtype x2, int n, float *y);

cv::Mat flare_source(cv::Mat &image, cv::Point center, int radius, float min_alpha, float max_alpha);

void add_sun_process(cv::Mat &image, int no_of_flare_circles, cv::Size imshape,  
        cv::Point center, int radius, float *x, float *y, float min_alpha, float max_alpha);

int rand_int_a2b(int a, int b );

void add_sun_flare_line(cv::Point center, double angle, cv::Size imshape, float *x, float *y);

void add_sun_flare(cv::Mat &image, double angle, cv::Point flare_center, int radius, int no_of_flare_circles, float min_alpha, float max_alpha);

// 模拟黑夜
template <typename Dtype> 
void arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y);

void adjust_gama(float gama, cv::Mat &image); 
void gama_com(float min_gama, float max_gama, float stride_gama, cv::Mat &image); 

// 
}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
#endif  // USE_OPENCV
