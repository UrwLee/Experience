#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <algorithm>

#include "caffe/mask/seg_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/im_transforms.hpp"

using namespace std;

namespace caffe {

template<typename Dtype>
SegDataTransformer<Dtype>::SegDataTransformer(
  const SegDataTransformationParameter& param, Phase phase)
  : param_(param), phase_(phase) {
  // check if we want to use mean_file
  srand( (unsigned)time( NULL ) );
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
                                          "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
                                           "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void SegDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    Blob<Dtype>* transformed_blob,
    bool preserve_pixel_vals) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  const int crop_size = param_.crop_size();
  const Dtype scale = preserve_pixel_vals ? 1 : param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file && !preserve_pixel_vals) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values && !preserve_pixel_vals) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  if (!preserve_pixel_vals) {
    randomDistortion(&cv_cropped_img);

    Dtype DarkProb = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX);
    if (DarkProb < param_.dark_prop()) {
      gama_com(param_.dark_gamma_min(), param_.dark_gamma_max(), 0.01, cv_cropped_img);
    }
    Dtype RAND_AUG = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX); //[0,1]
    if (param_.augmention() && RAND_AUG > 0.5) {
      std::vector<Dtype> probability;

      //10种增广，每种增广的分配一个probablity
      for (int i = 0; i < 6 ; i++) {
        Dtype RAND_PROB = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX); //[0,1]
        probability.push_back(RAND_PROB);
      }

      cv_cropped_img = ApplyAugmentation(cv_cropped_img, probability);

      CHECK_EQ(cv_cropped_img.rows, height) << "image must equals to blob";
      CHECK_EQ(cv_cropped_img.cols, width) << "image must equals to blob";
    }
  }
  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file && !preserve_pixel_vals) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values && !preserve_pixel_vals) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SegDataTransformer<Dtype>::randomDistortion(cv::Mat* image) {
  cv::Mat img = image->clone();
  CHECK_EQ(img.channels(), 3);
  *image = DistortImage(img, param_.dis_param());
}

template <typename Dtype>
void SegDataTransformer<Dtype>::arange(Dtype x2, Dtype x1, Dtype stride, Dtype *y) {
  // check x1 > x2
  int num = (int)((x1 - x2) / stride);
  for (int i = 0; i < num; ++i) {
    y[i] = x2 + i * stride;
  }
}

template <typename Dtype>
void SegDataTransformer<Dtype>::adjust_gama(Dtype gama, cv::Mat &image) {
  int table[256];
  Dtype ivgama = 1.0 / gama;
  cv::Mat lut(1, 256, CV_8U);
  cv::Mat out;
  unsigned char *p = lut.data;
  for (int i = 0; i < 256; ++i) {
    table[i] = pow((i / 255.0), ivgama) * 255;
    p[i] = table[i];
  }
  cv::LUT(image, lut, image);
}

template <typename Dtype>
void SegDataTransformer<Dtype>::gama_com(Dtype min_gama, Dtype max_gama, Dtype stride_gama, cv::Mat &image) {
  int num = (int)((max_gama - min_gama) / stride_gama);
  Dtype list_gama[num];
  arange(min_gama, max_gama, stride_gama, list_gama);
  int random_num = caffe_rng_rand() % num;
  CHECK_LT(random_num, num);
  if (list_gama[random_num] <= 0) list_gama[random_num] = 0.01f;
  Dtype gama = list_gama[random_num];
  adjust_gama(gama, image);
}

template<typename Dtype>
cv::Mat SegDataTransformer<Dtype>::ApplyAugmentation(const cv::Mat& in_img, std::vector<Dtype> prob) {
  cv::Mat out_img = in_img.clone();

  //最大图像质量
  const int quality = param_.quality();
  //噪声点数目
  const int noise_num = param_.noise_num();
  CHECK_LE(quality, 100.0) << "img quality cannot larger than 100";

  // 灰度化
  if (prob[0] > 0.8) {
    cv::Mat grayscale_img;
    cv::cvtColor(out_img, grayscale_img, CV_BGR2GRAY);
    cv::cvtColor(grayscale_img, out_img,  CV_GRAY2BGR);
  }

  //随机三种高斯模糊变化
  if (prob[1] > 0.5) {
    Dtype prob = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX); //[0,1]
    int kernel ;
    if (prob > 0) {kernel = 1;}
    if (prob > 0.3333) {kernel = 3;}
    if (prob > 0.6666) {kernel = 5;}
    cv::GaussianBlur(out_img, out_img, cv::Size(kernel, kernel), 1.5);
  }

  // jpg图像压缩
  if (prob[2] > 0.5) {
    Dtype prob = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX); //[0,1]
    vector<uchar> buf;
    vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back((quality - 30) * prob + 30);
    cv::imencode(".jpg", out_img, buf, params);
    out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
  }

  // 图像腐蚀
  if (prob[3] > 0.99) {
    cv::Mat element = cv::getStructuringElement(
                        2, cv::Size(3, 3), cv::Point(1, 1));
    cv::erode(out_img, out_img, element);
  }

  // 图像膨胀
  if (prob[4] > 0.99) {
    cv::Mat element = cv::getStructuringElement(
                        2, cv::Size(3, 3), cv::Point(1, 1));
    cv::dilate(out_img, out_img, element);
  }

  // 加噪
  if (prob[5] > 0.7) {
    for (int k = 0; k < noise_num ; k++) //将图像中num个像素随机
    {
      Dtype RAND = static_cast<Dtype>(rand()) / static_cast<Dtype>(RAND_MAX); //[0,1]
      int i = std::rand() % out_img.cols;
      int j = std::rand() % out_img.rows;

      if ( RAND < 0.3333) {
        //将图像颜色随机改变
        out_img.at<cv::Vec3b>(j, i)[0] = 255;
        out_img.at<cv::Vec3b>(j, i)[1] = 255;
        out_img.at<cv::Vec3b>(j, i)[2] = 255;
      } else if (RAND < 0.6666) {
        out_img.at<cv::Vec3b>(j, i)[0] = 0;
        out_img.at<cv::Vec3b>(j, i)[1] = 0;
        out_img.at<cv::Vec3b>(j, i)[2] = 0;
      } else {
        out_img.at<cv::Vec3b>(j, i)[0] = rand() % 255;
        out_img.at<cv::Vec3b>(j, i)[1] = rand() % 255;
        out_img.at<cv::Vec3b>(j, i)[2] = rand() % 255;
      }
    }
  }

  return  out_img;
}


template <typename Dtype>
void SegDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
                          (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int SegDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
    static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

#endif  // USE_OPENCV

INSTANTIATE_CLASS(SegDataTransformer);
}