#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>

#if CV_VERSION_MAJOR == 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#endif

#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"



namespace caffe {

const float prob_eps = 0.01;

int roll_weighted_die(const vector<float>& probabilities) {
  vector<float> cumulative;
  std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
                   std::back_inserter(cumulative));
  float val;
  caffe_rng_uniform(1, static_cast<float>(0), cumulative.back(), &val);

  // Find the position within the sequence and add 1
  return (std::lower_bound(cumulative.begin(), cumulative.end(), val)
          - cumulative.begin());
}

template <typename T>
bool is_border(const cv::Mat& edge, T color) {
  cv::Mat im = edge.clone().reshape(0, 1);
  bool res = true;
  for (int i = 0; i < im.cols; ++i) {
    res &= (color == im.at<T>(0, i));
  }
  return res;
}

template
bool is_border(const cv::Mat& edge, uchar color);

template <typename T>
cv::Rect CropMask(const cv::Mat& src, T point, int padding) {
  cv::Rect win(0, 0, src.cols, src.rows);

  vector<cv::Rect> edges;
  edges.push_back(cv::Rect(0, 0, src.cols, 1));
  edges.push_back(cv::Rect(src.cols-2, 0, 1, src.rows));
  edges.push_back(cv::Rect(0, src.rows-2, src.cols, 1));
  edges.push_back(cv::Rect(0, 0, 1, src.rows));

  cv::Mat edge;
  int nborder = 0;
  T color = src.at<T>(0, 0);
  for (int i = 0; i < edges.size(); ++i) {
    edge = src(edges[i]);
    nborder += is_border(edge, color);
  }

  if (nborder < 4) {
    return win;
  }

  bool next;
  do {
    edge = src(cv::Rect(win.x, win.height - 2, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.height--;
    }
  } while (next && (win.height > 0));

  do {
    edge = src(cv::Rect(win.width - 2, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.width--;
    }
  } while (next && (win.width > 0));

  do {
    edge = src(cv::Rect(win.x, win.y, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.y++;
      win.height--;
    }
  } while (next && (win.y <= src.rows));

  do {
    edge = src(cv::Rect(win.x, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.x++;
      win.width--;
    }
  } while (next && (win.x <= src.cols));

  // add padding
  if (win.x > padding) {
    win.x -= padding;
  }
  if (win.y > padding) {
    win.y -= padding;
  }
  if ((win.width + win.x + padding) < src.cols) {
    win.width += padding;
  }
  if ((win.height + win.y + padding) < src.rows) {
    win.height += padding;
  }

  return win;
}

template
cv::Rect CropMask(const cv::Mat& src, uchar point, int padding);

cv::Mat colorReduce(const cv::Mat& image, int div) {
  cv::Mat out_img;
  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar* p = lookUpTable.data;
  const int div_2 = div / 2;
  for ( int i = 0; i < 256; ++i ) {
    p[i] = i / div * div + div_2;
  }
  cv::LUT(image, lookUpTable, out_img);
  return out_img;
}

void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut) {
  cv::Mat edgesNeg = edgesIn.clone();
  cv::Scalar val(255, 255, 255);
  cv::floodFill(edgesNeg, cv::Point(0, 0), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(0, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, 0), val);
  cv::bitwise_not(edgesNeg, edgesNeg);
  *filledEdgesOut = (edgesNeg | edgesIn);
  return;
}

void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img) {
  cv::Mat mask, crop_mask;
  if (in_img.channels() > 1) {
    cv::Mat in_img_gray;
    cv::cvtColor(in_img, in_img_gray, CV_BGR2GRAY);
    cv::threshold(in_img_gray, mask, 0, 255,
                  CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  } else {
    cv::threshold(in_img, mask, 0, 255,
                  CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  }
  cv::Rect crop_rect = CropMask(mask, mask.at<uchar>(0, 0), 2);

  if (fill_bg) {
    cv::Mat temp_img = in_img(crop_rect);
    fillEdgeImage(mask, &mask);
    crop_mask = mask(crop_rect).clone();
    *out_img = cv::Mat::zeros(crop_rect.size(), in_img.type());
    temp_img.copyTo(*out_img, crop_mask);
  } else {
    *out_img = in_img(crop_rect).clone();
  }
}

cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type,  const cv::Scalar pad_val,
                                  const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (orig_aspect > new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_height - resSize.height) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, padding,
                       new_height - resSize.height - padding, 0, 0,
                       pad_type, pad_val);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_width - resSize.width) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
                       new_width - resSize.width - padding,
                       pad_type, pad_val);
  }
  return img_resized;
}

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width,
                                   const int new_height,
                                   const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float> (new_width) / new_height;

  if (orig_aspect < new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
  }
  return img_resized;
}

void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image) {
  const int cols = image->cols;
  const int rows = image->rows;

  if (image->channels() == 1) {
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      uchar* ptr = image->ptr<uchar>(j);
      ptr[i]= val[0];
    }
  } else if (image->channels() == 3) {  // color image
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      cv::Vec3b* ptr = image->ptr<cv::Vec3b>(j);
      (ptr[i])[0] = val[0];
      (ptr[i])[1] = val[1];
      (ptr[i])[2] = val[2];
    }
  }
}

void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox) {
  float new_height = param.height();
  float new_width = param.width();
  float orig_aspect = static_cast<float>(old_width) / old_height;
  float new_aspect = new_width / new_height;

  float x_min = bbox->xmin() * old_width;
  float y_min = bbox->ymin() * old_height;
  float x_max = bbox->xmax() * old_width;
  float y_max = bbox->ymax() * old_height;
  float padding;
  switch (param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      x_min = std::max(0.f, x_min * new_width / old_width);
      x_max = std::min(new_width, x_max * new_width / old_width);
      y_min = std::max(0.f, y_min * new_height / old_height);
      y_max = std::min(new_height, y_max * new_height / old_height);
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      if (orig_aspect > new_aspect) {
        padding = (new_height - new_width / orig_aspect) / 2;
        x_min = std::max(0.f, x_min * new_width / old_width);
        x_max = std::min(new_width, x_max * new_width / old_width);
        y_min = y_min * (new_height - 2 * padding) / old_height;
        y_min = padding + std::max(0.f, y_min);
        y_max = y_max * (new_height - 2 * padding) / old_height;
        y_max = padding + std::min(new_height, y_max);
      } else {
        padding = new_width - orig_aspect * new_height / 2;
        x_min = x_min * (new_width - 2 * padding) / old_width;
        x_min = padding + std::max(0.f, x_min);
        x_max = x_max * (new_width - 2*padding) / old_width;
        x_max = padding + std::min(new_width, x_max);
        y_min = std::max(0.f, y_min * new_height / old_height);
        y_max = std::min(new_height, y_max * new_height / old_height);
      }
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      if (orig_aspect < new_aspect) {
        new_height = new_width / orig_aspect;
      } else {
        new_width = orig_aspect * new_height;
      }
      x_min = std::max(0.f, x_min * new_width / old_width);
      x_max = std::min(new_width, x_max * new_width / old_width);
      y_min = std::max(0.f, y_min * new_height / old_height);
      y_max = std::min(new_height, y_max * new_height / old_height);
      break;
    default:
      LOG(FATAL) << "Unknown resize mode.";
  }
  bbox->set_xmin(x_min / new_width);
  bbox->set_ymin(y_min / new_height);
  bbox->set_xmax(x_max / new_width);
  bbox->set_ymax(y_max / new_height);
}

cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param) {
  cv::Mat out_img;

  // Reading parameters
  const int new_height = param.height();
  const int new_width = param.width();

  int pad_mode = cv::BORDER_CONSTANT;
  switch (param.pad_mode()) {
    case ResizeParameter_Pad_mode_CONSTANT:
      break;
    case ResizeParameter_Pad_mode_MIRRORED:
      pad_mode = cv::BORDER_REFLECT101;
      break;
    case ResizeParameter_Pad_mode_REPEAT_NEAREST:
      pad_mode = cv::BORDER_REPLICATE;
      break;
    default:
      LOG(FATAL) << "Unknown pad mode.";
  }

  int interp_mode = cv::INTER_LINEAR;
  int num_interp_mode = param.interp_mode_size();
  if (num_interp_mode > 0) {
    // 多种模式以相同概率出现
    vector<float> probs(num_interp_mode, 1.f / num_interp_mode);
    // 随机出现一种填充方式
    int prob_num = roll_weighted_die(probs);
    // 解析
    switch (param.interp_mode(prob_num)) {
      case ResizeParameter_Interp_mode_AREA:
        interp_mode = cv::INTER_AREA;
        break;
      case ResizeParameter_Interp_mode_CUBIC:
        interp_mode = cv::INTER_CUBIC;
        break;
      case ResizeParameter_Interp_mode_LINEAR:
        interp_mode = cv::INTER_LINEAR;
        break;
      case ResizeParameter_Interp_mode_NEAREST:
        interp_mode = cv::INTER_NEAREST;
        break;
      case ResizeParameter_Interp_mode_LANCZOS4:
        interp_mode = cv::INTER_LANCZOS4;
        break;
      default:
        LOG(FATAL) << "Unknown interp mode.";
    }
  }

  // pad颜色确定，默认是黑色
  cv::Scalar pad_val = cv::Scalar(0, 0, 0);
  const int img_channels = in_img.channels();
  if (param.pad_value_size() > 0) {
    CHECK(param.pad_value_size() == 1 ||
          param.pad_value_size() == img_channels) <<
        "Specify either 1 pad_value or as many as channels: " << img_channels;
    vector<float> pad_values;
    for (int i = 0; i < param.pad_value_size(); ++i) {
      pad_values.push_back(param.pad_value(i));
    }
    if (img_channels > 1 && param.pad_value_size() == 1) {
      // Replicate the pad_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        pad_values.push_back(pad_values[0]);
      }
    }
    pad_val = cv::Scalar(pad_values[0], pad_values[1], pad_values[2]);
  }

  // 完成Resize: 默认使用WARP模式
  switch (param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      cv::resize(in_img, out_img, cv::Size(new_width, new_height), 0, 0,
                 interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      out_img = AspectKeepingResizeAndPad(in_img, new_width, new_height,
                                          pad_mode, pad_val, interp_mode);
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      out_img = AspectKeepingResizeBySmall(in_img, new_width, new_height,
                                           interp_mode);
      break;
    default:
      LOG(INFO) << "Unknown resize mode.";
  }
  return  out_img;
}

cv::Mat ApplyCrop(const cv::Mat& in_img, const NormalizedBBox& crop_bbox){
  cv::Mat out_img;
  // 下面完成裁剪
  int img_width = in_img.cols;
  int img_height = in_img.rows;

  if ((crop_bbox.xmax() - crop_bbox.xmin()) > 1.) {
    LOG(FATAL) << "crop_bbox should be Normalized.";
  }
  if ((crop_bbox.ymax() - crop_bbox.ymin()) > 1.) {
    LOG(FATAL) << "crop_bbox should be Normalized.";
  }

  int w_off_int = (int)(crop_bbox.xmin() * img_width);
  int h_off_int = (int)(crop_bbox.ymin() * img_height);

  float cw = crop_bbox.xmax() - crop_bbox.xmin();
  float ch = crop_bbox.ymax() - crop_bbox.ymin();

  int crop_width_int = (int)(cw * img_width);
  int crop_height_int = (int)(ch * img_height);

  // 定义裁剪ROI
  cv::Rect roi(w_off_int, h_off_int, crop_width_int, crop_height_int);
  out_img = in_img(roi);
  return out_img;
}

cv::Mat ApplyCrop(const cv::Mat& in_img, const RandomCropParameter& param, const Phase& phase, NormalizedBBox* crop_bbox){
  cv::Mat out_img;
  CHECK_GE(param.max_scale(), param.min_scale());
  CHECK_GT(param.min_scale(), 0.);
  CHECK_LE(param.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, param.min_scale(), param.max_scale(), &scale);

  CHECK_GE(param.max_aspect(), param.min_aspect());
  CHECK_GT(param.min_aspect(), 0.);
  CHECK_LT(param.max_aspect(), 10.);
  float aspect_ratio;

  float min_aspect_ratio = std::max<float>(param.min_aspect(),
                                           std::pow(scale, 2.));
  float max_aspect_ratio = std::min<float>(param.max_aspect(),
                                      1. / std::pow(scale, 2.));
  caffe_rng_uniform(1, min_aspect_ratio, max_aspect_ratio, &aspect_ratio);
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);

  // 获取随机裁剪坐标
  float w_off, h_off;
  if (phase == TRAIN) {
    caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
    caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);
  }
  else if (phase == TEST) {
    w_off = (1 - bbox_width) / 2.;
    w_off = (1 - bbox_height) / 2.;
  }
  else {
    LOG(FATAL) << "Illegal phase state.";
  }
  // 获取crop-box
  crop_bbox->set_xmin(w_off);
  crop_bbox->set_ymin(h_off);
  crop_bbox->set_xmax(w_off + bbox_width);
  crop_bbox->set_ymax(h_off + bbox_height);

  // 下面完成裁剪
  int img_width = in_img.cols;
  int img_height = in_img.rows;
  // int img_channels = in_img.channels();

  int w_off_int = (int)(w_off * img_width);
  int h_off_int = (int)(h_off * img_height);
  int crop_width_int = (int)(bbox_width * img_width);
  int crop_height_int = (int)(bbox_height * img_height);

  // 定义裁剪ROI
  cv::Rect roi(w_off_int, h_off_int, crop_width_int, crop_height_int);
  out_img = in_img(roi);
  return out_img;
}

cv::Mat ApplyDistorted(const cv::Mat& in_img, const DistoredParameter& param) {
  cv::Mat out_img;
  // convert BGR -> HSV
  const float hue = param.hue();
  const float sat = param.sat();
  const float val = param.val();

  const float sat_inv = 1. / sat;
  const float val_inv = 1. / val;
  const float hue_inv = -hue;

  CHECK_GE(sat, 1.) << "the sat param shoule be greater than 1.";
  CHECK_GE(val, 1.) << "the val param shoule be greater than 1.";
  CHECK_GE(hue, 0.) << "the hue param shoule be greater than 0.";
  CHECK_LT(hue, 1.) << "the hue param should be less than 1.";

  // choose a ramdom param
  float sat_scale, val_scale, hue_add;
  caffe_rng_uniform(1, sat_inv, sat, &sat_scale);
  caffe_rng_uniform(1, val_inv, val, &val_scale);
  caffe_rng_uniform(1, hue_inv, hue, &hue_add);

  cv::Mat hsv_img;
  // BGR2HSV: H:0-180, S/V:0-255
  // BGR2HSV_FULL: H:0-255, S/V:0-255
  cv::cvtColor(in_img, hsv_img, CV_BGR2HSV);

  // scale for sat and val
  vector<cv::Mat> channels;
  cv::split(hsv_img, channels);
  cv::Mat hue_channel = channels.at(0);
  cv::Mat sat_channel = channels.at(1);
  cv::Mat val_channel = channels.at(2);
  // sat scale: 0-255
  for (int i = 0; i < sat_channel.rows; ++i){
    const uchar* src_ptr = channels.at(1).ptr<uchar>(i);
    uchar* dst_ptr  = sat_channel.ptr<uchar>(i);
    for (int j = 0; j < sat_channel.cols; ++j) {
      dst_ptr[j] = cv::saturate_cast<uchar>(src_ptr[j]*sat_scale);
    }
  }
  // val scale: 0-255
  for (int i = 0; i < val_channel.rows; ++i){
    const uchar* src_ptr = channels.at(2).ptr<uchar>(i);
    uchar* dst_ptr  = val_channel.ptr<uchar>(i);
    for (int j = 0; j < val_channel.cols; ++j) {
      dst_ptr[j] = cv::saturate_cast<uchar>(src_ptr[j]*val_scale);
    }
  }
  //add hue
  //0-255:FULL
  // int hue_add_int = (int)(hue_add * 256);
  // for (int i = 0; i < hue_channel.rows; ++i){
  //   const uchar* src_ptr = channels.at(0).ptr<uchar>(i);
  //   uchar* dst_ptr  = hue_channel.ptr<uchar>(i);
  //   for (int j = 0; j < hue_channel.cols; ++j) {
  //     dst_ptr[j] = saturate_cast<uchar>(src_ptr[j] + hue_add_int);
  //   }
  // }
  // H:0-180
  for (int i = 0; i < hue_channel.rows; ++i) {
    const uchar* src_ptr = channels.at(0).ptr<uchar>(i);
    uchar* dst_ptr = hue_channel.ptr<uchar>(i);
    for (int j = 0; j < hue_channel.cols; ++j) {
      float norm_hue = (float)src_ptr[j] / 180.0 + hue_add;
      if (norm_hue > 1.) norm_hue -= 1.;
      if (norm_hue < 0.) norm_hue += 1.;
      dst_ptr[j] = (int)(norm_hue * 180);
    }
  }
  //merge of HSV channels
  vector<cv::Mat> transformed_channels;
  transformed_channels.push_back(hue_channel);
  transformed_channels.push_back(sat_channel);
  transformed_channels.push_back(val_channel);
  cv::merge(transformed_channels,hsv_img);
  cv::cvtColor(hsv_img, out_img, CV_HSV2BGR);
  return out_img;
}

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param) {
  cv::Mat out_img;

  // 灰度化
  if (param.decolorize()) {
    cv::Mat grayscale_img;
    cv::cvtColor(in_img, grayscale_img, CV_BGR2GRAY);
    cv::cvtColor(grayscale_img, out_img,  CV_GRAY2BGR);
  } else {
    out_img = in_img;
  }

  // 高斯模糊化， 7x7-kernel， 1.5-delta_sigma
  if (param.gauss_blur()) {
    cv::GaussianBlur(out_img, out_img, cv::Size(7, 7), 1.5);
  }

  // 直方图： 对亮度分量进行归一化，增加对比度
  if (param.hist_eq()) {
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      cv::equalizeHist(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Mat temp_img;
      cv::equalizeHist(out_img, temp_img);
      out_img = temp_img;
    }
  }

  // 限制对比度的自适应局部直方图归一化：CLAHE方法
  if (param.clahe()) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      clahe->apply(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
      clahe->setClipLimit(4);
      cv::Mat temp_img;
      clahe->apply(out_img, temp_img);
      out_img = temp_img;
    }
  }

  // jpg图像压缩
  if (param.jpeg() > 0) {
    vector<uchar> buf;
    vector<int> params;
    params.push_back(CV_IMWRITE_JPEG_QUALITY);
    params.push_back(param.jpeg());
    cv::imencode(".jpg", out_img, buf, params);
    out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
  }

  // 图像腐蚀
  if (param.erode()) {
    cv::Mat element = cv::getStructuringElement(
        2, cv::Size(3, 3), cv::Point(1, 1));
    cv::erode(out_img, out_img, element);
  }

  // 色彩退化：从8位颜色降低为4位
  if (param.posterize()) {
    cv::Mat tmp_img;
    tmp_img = colorReduce(out_img);
    out_img = tmp_img;
  }

  // 反色处理
  if (param.inverse()) {
    cv::Mat tmp_img;
    cv::bitwise_not(out_img, tmp_img);
    out_img = tmp_img;
  }

  // 加噪
  vector<uchar> noise_values;
  // 椒盐噪声参数
  if (param.saltpepper_param().value_size() > 0) {
    CHECK(param.saltpepper_param().value_size() == 1
          || param.saltpepper_param().value_size() == out_img.channels())
        << "Specify either 1 pad_value or as many as channels: "
        << out_img.channels();

    for (int i = 0; i < param.saltpepper_param().value_size(); i++) {
      noise_values.push_back(uchar(param.saltpepper_param().value(i)));
    }
    if (out_img.channels()  > 1
        && param.saltpepper_param().value_size() == 1) {
      // Replicate the pad_value for simplicity
      for (int c = 1; c < out_img.channels(); ++c) {
        noise_values.push_back(uchar(noise_values[0]));
      }
    }
  }
  // 加入椒盐噪声
  if (param.saltpepper()) {
    const int noise_pixels_num =
        floor(param.saltpepper_param().fraction()
              * out_img.cols * out_img.rows);
    constantNoise(noise_pixels_num, noise_values, &out_img);
  }

  // 转换为HSV图像
  if (param.convert_to_hsv()) {
    cv::Mat hsv_image;
    cv::cvtColor(out_img, hsv_image, CV_BGR2HSV);
    out_img = hsv_image;
  }

  // 转换为LAB图像
  if (param.convert_to_lab()) {
    cv::Mat lab_image;
    out_img.convertTo(lab_image, CV_32F);
    lab_image *= 1.0 / 255;
    cv::cvtColor(lab_image, out_img, CV_BGR2Lab);
  }
  return  out_img;
}

void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < brightness_prob) {
    CHECK_GE(brightness_delta, 0) << "brightness_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -brightness_delta, brightness_delta, &delta);
    AdjustBrightness(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    in_img.convertTo(*out_img, -1, 1, delta);
  } else {
    *out_img = in_img;
  }
}

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < contrast_prob) {
    CHECK_GE(upper, lower) << "contrast upper must be >= lower.";
    CHECK_GE(lower, 0) << "contrast lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustContrast(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img) {
  if (fabs(delta - 1.f) > 1e-3) {
    in_img.convertTo(*out_img, -1, delta, 0);
  } else {
    *out_img = in_img;
  }
}

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < saturation_prob) {
    CHECK_GE(upper, lower) << "saturation upper must be >= lower.";
    CHECK_GE(lower, 0) << "saturation lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustSaturation(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta - 1.f) != 1e-3) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the saturation.
    channels[1].convertTo(channels[1], -1, delta, 0);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < hue_prob) {
    CHECK_GE(hue_delta, 0) << "hue_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -hue_delta, hue_delta, &delta);
    AdjustHue(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the hue.
    channels[0].convertTo(channels[0], -1, 1, delta);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < random_order_prob) {
    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);
    CHECK_EQ(channels.size(), 3);

    // Shuffle the channels.
    std::random_shuffle(channels.begin(), channels.end());
    cv::merge(channels, *out_img);
  } else {
    *out_img = in_img;
  }
}

/*
  wdh 
  添加光晕
*/
// ===================================================
// func: 模拟 python linspce
template <typename Dtype> 
void linSpace(Dtype x1, Dtype x2, int n, float *y){
    int i = 0;
    Dtype d = (x2 - x1) / (n - 1);
    for(i=0; i<n; i++){
        y[i] = x1 + i*d;
    }
}
template
void linSpace(int x1, int x2, int n, float *y);

template
void linSpace(double x1, double x2, int n, float *y);

// func: 主阳光光晕设置, 
/*
  center: sun 位置
  radius: sun 光圈半径
  min_alpha: 最小sun 透明度
  max_alpha: 
*/
cv::Mat flare_source(cv::Mat &image, cv::Point center, int radius, float min_alpha, float max_alpha){
    cv::Mat overlay, output;
    image.copyTo(overlay);
    image.copyTo(output);
    int num_times = (int)(radius / 2.f);
    float alpha = 0.f;
    int rad = 0;
    for(int num=0; num < num_times; ++num){
        alpha = num * ((max_alpha - min_alpha) / (num_times - 1.f) );
        rad = 1.f + num * ((radius - 1.f) /( num_times - 1.f));
        cv::circle(overlay, center, (int)rad, cv::Scalar(240, 250, 255), 2);
        float alp = std::pow(alpha * (num_times - num - 1.f)/num, 3);
        cv::addWeighted(overlay, alp, output, 1 -alp, 0, output);
    }
    return output;
}

// func: 产生int a~b 随机数
int rand_int_a2b(int a, int b ){
  CHECK_GT(b, a);
  return caffe_rng_rand() % (b-a+1) + a;
}

// func: 产生 sun 和 around 光圈
/*
  no_of_flare_circles: around 光晕个数

*/
void add_sun_process(cv::Mat &image, int no_of_flare_circles, cv::Size imshape,  
        cv::Point center, int radius, float *x, float *y, float min_alpha, float max_alpha){
    cv::Mat overlay, output;
    image.copyTo(overlay);
    image.copyTo(output);
    
    int num_times = (int)(radius / 2.f);
    float alpha = 0.f;
    int rad = 0;
    
    for(int i=0; i<no_of_flare_circles; ++i){ // 
        float alpha;
        caffe_rng_uniform(1, 0.05f, 0.2f, &alpha);
        int r = rand_int_a2b(0, 9); // 这里是 数组x-1的长度 有问题需要改, rand() % (b-a+1)+ a 
        int rad = rand_int_a2b(1, (int)(imshape.height/10.f + 1));
        // 不同颜色转换 

        int color_r = rand_int_a2b(190, 240);
        int color_g = rand_int_a2b(200, 250);
        int color_b = rand_int_a2b(205, 255);

        cv::circle(overlay, cv::Point((int)(x[r]),(int)(y[r])), rad, cv::Scalar(color_r, color_g, color_b), -1 ); 
        cv::addWeighted(overlay, alpha, output, 1-alpha, 0, output);
    }
    output = flare_source(output, center, radius, min_alpha, max_alpha);
    image = output; 
}

void add_sun_flare_line(cv::Point center, double angle, cv::Size imshape, float *x, float *y){
    int num = 0;
    float rand_x[10], rand_y[10];
    linSpace<int>(0, imshape.height, 10, rand_x);
    for(int i=0; i<10; ++i){
        rand_y[i] = tan(angle) * (rand_x[i] - center.x) + center.y;
        x[i] = rand_x[i];
        y[i] = 2.f * center.y - rand_y[i];
    }
}

void add_sun_flare(cv::Mat &image, double angle, cv::Point flare_center, int radius, int no_of_flare_circles, float min_alpha, float max_alpha){
   
    cv::Size imshape = image.size();
    double angle_t = 0;
    if(angle == -1){
        caffe_rng_uniform(1, 0.0, 2*3.1415, &angle_t);
    }
    else {
        angle_t = angle;
    }
    cv::Point flare_center_t;

    if(flare_center.x == -1 && flare_center.y == -1){ // func: 获得随机光晕坐标

        flare_center_t.x = rand_int_a2b(0, imshape.width);
        flare_center_t.y = rand_int_a2b(0, (int)(imshape.height/2.f));  // 设置只在图片上半边出现 光晕
    }
    else{
      flare_center_t.x = flare_center.x;
      flare_center_t.y = flare_center.y;
    }
    float x[10];
    float y[10];
    add_sun_flare_line(flare_center_t, angle_t, imshape, x, y);
    add_sun_process(image, no_of_flare_circles, imshape, flare_center_t, radius, x, y,  min_alpha,  max_alpha);   
}

/*
  e.g.
   add_sun_flare(out_img, -1, cv::Point(-1,-1), 200, 8, 0.05f, 0.2f); 
   图片, 光晕角度(-1随机), sun 位置(-1随机),  光晕个数, sun 最小(float类型)

*/
// ======================================================


// func: 模拟黑夜场景, 
/*
 gama_com(
    float min_gama,  
    float max_gama,
    float stride_gama,  // 不进, 在min max之间 随机选择一个参数
    cv::Mat &image)
*/
// ======================================================

template <typename Dtype>
void arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y){
    // check x1 > x2
    int num = (int)((x1 - x2)/stride);
    for(int i=0; i<num; ++i){
        y[i] = x2 + i*stride;
    }
}

template
void arange(float x2, float x1, float stride, float *y);


void adjust_gama(float gama, cv::Mat &image){
    int table[256];
    float ivgama = 1.0/gama;
    cv::Mat lut(1, 256, CV_8U);
    cv::Mat out;
    unsigned char *p = lut.data;
    for (int i=0; i<256; ++i){
        table[i] = pow((i / 255.0), ivgama) * 255;
        p[i] = table[i];
    }
    cv::LUT(image, lut, image);
}

void gama_com(float min_gama, float max_gama, float stride_gama, cv::Mat &image){
    int num = (int)((max_gama - min_gama) / stride_gama);  
    float list_gama[num];
    arange(min_gama, max_gama, stride_gama, list_gama); 
    
    int random_num = caffe_rng_rand() % num;
    CHECK_LT(random_num, num);
    if (list_gama[random_num] <= 0) list_gama[random_num] = 0.01f;        
    float gama = list_gama[random_num];
    adjust_gama(gama, image);
}
/*
e.g. 
    黑夜效果 0.16 开始, 小于0.16 太黑..  , 最好在尺寸较大图片上进行增广
    gama_com(0.3f, 0.5f, 0.01f, out_img);

*/
// ======================================================


 
cv::Mat DistortImage(const cv::Mat& in_img, const DistortionParameter& param) {
  cv::Mat out_img = in_img; 
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob > 0.5) {
    // Do random sun flare
    // add_sun_flare(out_img, -1, cv::Point(-1, -1), 200, 8, 0.f, 0.5f); 
    // gama_com(0.3f, 0.5f, 0.01f, out_img);

    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  } else {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  }

  return out_img;
}

cv::Mat ExpandImage(const cv::Mat& in_img, const ExpansionParameter& param,
                    NormalizedBBox* expand_bbox) {
  const float expand_prob = param.prob();

  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob > expand_prob) {
    *expand_bbox = UnitBBox();
    return (in_img);
  }

  const float max_expand_ratio = param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    *expand_bbox = UnitBBox();
    return (in_img);
  }

  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);

  const int img_height = in_img.rows;
  const int img_width = in_img.cols;

  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);

  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);

  h_off = floor(h_off);
  w_off = floor(w_off);

  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  cv::Mat expand_img;
  expand_img.create(height, width, in_img.type());
  expand_img.setTo(cv::Scalar(0));
  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  in_img.copyTo(expand_img(bbox_roi));

  return expand_img;
}
}  // namespace caffe

#endif  // USE_OPENCV
