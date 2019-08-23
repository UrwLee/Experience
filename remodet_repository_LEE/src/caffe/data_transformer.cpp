#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include "caffe/util/im_transforms.hpp"
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
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

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
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

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  //最后数据转换, 获取输入数据
  const string& data = datum.data();
  // 获取输入尺寸
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  //获取裁剪尺寸和增益参数
  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  // 随机镜像
  *do_mirror = param_.mirror() && Rand(2);
  // 均值
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  // 处理均值
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  // 默认为输入高度
  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  // 如果需要裁剪,则先裁剪
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // 随机裁剪
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  // 根据裁剪结果设置crop-box
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // 输入数据指针
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          // 输出数据指针
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          // 输出数据指针
          top_index = (c * height + h) * width + w;
        }
        // 使用uint8类型
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
        // 使用float数据类型
          datum_element = datum.float_data(data_index);
        }
        // 减掉均值,乘以增益
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_data, &crop_bbox, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, crop_bbox, do_mirror);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  /**
   * 得到裁剪尺寸
   * 获取输入尺寸:原图片随机裁剪后得到的尺寸
   */
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // 获取输出尺寸:resize尺寸,300x300
  // channels=3
  // num = 1
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  /**
   * 怎么保证: 输入尺寸始终大于等于resize尺寸呢?
   * 输入尺寸: 原图按照随机数裁剪得到的,其尺寸并不大
   * resize尺寸: 300x300
   */
  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, crop_bbox, do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_blob, &crop_bbox, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    bool* do_mirror) {
  // 获取数据,准备转换
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;
  // 将图片数据按照变换参数进行转换,尺寸变换参数写入crop_box,也就是裁剪的区域
  // 默认此处应该是不裁剪的
  // 问题: 采样器获得datum的尺寸是随机的,怎么保证与transformed_blob相同呢?
  // Transform只是做了个裁剪,感觉中间缺少了一步:
  // 将采样器获得的图片进行resize成300x300,然后再进行Transform
  // do_mirror是随机产生的
  // 步骤应该是:
  /**
   * 1. 采样器随机获取一副裁剪的图片crop_bbox,标注列表进行修正;
   * 2. 将采样器获得的图片按照resize参数进行resize,标注列表再次进行修正;
   * 3. 完成Transform过程: 减去均值,mirror/HSV等等, 如果存在mirror,box也需要进行修正
   */
  Transform(datum, transformed_blob, &crop_bbox, do_mirror);

  // Transform annotation.
  TransformAnnotation(anno_datum, crop_bbox, *do_mirror,
                      transformed_anno_group_all);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_group_all,
            &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) {
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum& anno_datum,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // 对所有的标定box全部进行转换
    // 遍历所有类的boxes列表
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      // 该类的boxes
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      // 变换后的boxes信息
      AnnotationGroup transformed_anno_group;
      // 处理每个box
      bool has_valid_annotation = false;
      // 遍历这个类下的每条box记录
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        // 获取该条记录
        const Annotation& anno = anno_group.annotation(a);
        // 获取其box
        const NormalizedBBox& bbox = anno.bbox();
        // 是否满足发布条件???
        // 两种准则:中心和交集
        // 中心:目标box的中心要落在裁剪范围内才满足发布条件
        // 交集:目标box落在裁剪范围内的百分比要达到要求才能发布
        // 不满足发布条件的目标box等同于被裁剪掉
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
          continue;
        }
        // 通过发布条件的box,开始进行范围修正
        // Adjust bounding box annotation.
        NormalizedBBox proj_bbox;
        /**
         * ProjectBBox:完成标定box在新的图片范围crop_box的重新标定
         * 结果存放到proj_bbox
         * 只要有交集,返回结果为True
         */
        if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
          // 存在交集
          has_valid_annotation = true;
          // 在输出组中创建一条box记录
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          // 设置其实例ID
          transformed_anno->set_instance_id(anno.instance_id());
          // 获取其数据指针
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          // 复制
          transformed_bbox->CopyFrom(proj_bbox);
          // 如果需要镜像,则修改x参数
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
        }
      }
      // Save for output.
      // 只要该组下有一条记录,则将该组增加到最终的标注输出中
      if (has_valid_annotation) {
        // 设置组编号,其实就是类号
        transformed_anno_group.set_group_label(anno_group.group_label());
        // 将该组增加到最终输出结果中
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }//所有的类都遍历结束
  } else {
    // 暂时不支持其他类型的标注
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum& datum,
                                       const NormalizedBBox& bbox,
                                       Datum* crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    cv::Mat crop_img;
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
    crop_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  const int w_off = static_cast<int>(scaled_bbox.xmin());
  const int h_off = static_cast<int>(scaled_bbox.ymin());
  const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

  // Crop the image using bbox.
  crop_datum->set_channels(datum_channels);
  crop_datum->set_height(height);
  crop_datum->set_width(width);
  crop_datum->set_label(datum.label());
  crop_datum->clear_data();
  crop_datum->clear_float_data();
  crop_datum->set_encoded(false);
  const int crop_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(crop_datum_size, ' ');
  for (int h = h_off; h < h_off + height; ++h) {
    for (int w = w_off; w < w_off + width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
        buffer[crop_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  crop_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum) {
  // 源数据裁剪
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  // 将所有满足条件的标注全部进行转换
  TransformAnnotation(anno_datum, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group());
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformCvbyCBox(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       const NormalizedBBox& crop_bbox,
                                       bool* do_mirror) {
  // 获取原图尺寸
  const int img_channels = cv_img.channels();
  // int img_height = cv_img.rows;
  // int img_width = cv_img.cols;

  // 获取转换结果尺寸
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  // 断言
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte.";

  // 获取scale
  const Dtype scale = param_.scale();
  // 随机镜像
  *do_mirror = param_.mirror() && Rand(2);

  // const int crop_size = param_.crop_size();
  // 均值
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  CHECK(!(has_mean_file == true && has_mean_values == true)) <<
                        "the mean file and mean values could not be given in the same time.";
  // 定义裁剪/失真/噪声/resize图像
  // 首先裁剪
  // 然后随机失真
  // 然后随机加噪声
  // 最后进行resize
  // resize之后的图像进行[-平均值][*scale][mirror]操作!
  cv::Mat cv_cropped_image, cv_distorted_image, cv_noised_image, cv_resized_image;

  CropImage(cv_img,crop_bbox,&cv_cropped_image);

  if (param_.has_distored_param()) {
    cv_distorted_image = ApplyDistorted(cv_cropped_image, param_.distored_param());
  }
  else cv_distorted_image = cv_cropped_image;

  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_distorted_image, param_.noise_param());
  }
  else cv_noised_image = cv_distorted_image;

  if (param_.has_resize_param()) {
    cv_resized_image = ApplyResize(cv_noised_image, param_.resize_param());
  }
  else cv_resized_image = cv_noised_image;

  CHECK_GT(img_channels, 0);
  // 减去均值
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(cv_resized_image.channels(), data_mean_.channels());
    CHECK_EQ(cv_resized_image.rows, data_mean_.height());
    CHECK_EQ(cv_resized_image.cols, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == cv_resized_image.channels()) <<
        "Specify either 1 mean_value or as many as channels: " << cv_resized_image.channels();
    if (cv_resized_image.channels() > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < cv_resized_image.channels(); ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  // 检查图片的数据
  CHECK(cv_resized_image.data);

  int resized_height = cv_resized_image.rows;
  int resized_width = cv_resized_image.cols;
  int resized_channels = cv_resized_image.channels();

  CHECK_EQ(resized_width, width);
  CHECK_EQ(resized_height, height);
  CHECK_EQ(resized_channels, channels);

  // 复制数据
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_resized_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * height + h_idx_real) * width + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
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
  void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                         Blob<Dtype>* transformed_blob,
                                         NormalizedBBox* crop_bbox,
                                         bool* do_mirror) {
    // 获取原图尺寸
    const int img_channels = cv_img.channels();
    // int img_height = cv_img.rows;
    // int img_width = cv_img.cols;

    // 获取转换结果尺寸
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    // 断言
    CHECK_EQ(channels, img_channels);
    CHECK_GE(num, 1);
    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte.";

    // 获取scale
    const Dtype scale = param_.scale();
    // 随机镜像
    *do_mirror = param_.mirror() && Rand(2);

    // const int crop_size = param_.crop_size();
    // 均值
    const bool has_mean_file = param_.has_mean_file();
    const bool has_mean_values = mean_values_.size() > 0;
    CHECK(!(has_mean_file == true && has_mean_values == true)) <<
                          "the mean file and mean values could not be given in the same time.";
    // 定义裁剪/失真/噪声/resize图像
    // 首先裁剪
    // 然后随机失真
    // 然后随机加噪声
    // 最后进行resize
    // resize之后的图像进行[-平均值][*scale][mirror]操作!
    cv::Mat cv_cropped_image, cv_distorted_image, cv_noised_image, cv_resized_image;

    // CropImage(cv_img,crop_bbox,&cv_cropped_image);
    if (param_.has_crop_param()) {
      cv_cropped_image = ApplyCrop(cv_img, param_.crop_param(), phase_, crop_bbox);
    }
    else cv_cropped_image = cv_img;

    if (param_.has_distored_param()) {
      cv_distorted_image = ApplyDistorted(cv_cropped_image, param_.distored_param());
    }
    else cv_distorted_image = cv_cropped_image;

    if (param_.has_noise_param()) {
      cv_noised_image = ApplyNoise(cv_distorted_image, param_.noise_param());
    }
    else cv_noised_image = cv_distorted_image;

    if (param_.has_resize_param()) {
      cv_resized_image = ApplyResize(cv_noised_image, param_.resize_param());
    }
    else cv_resized_image = cv_noised_image;

    CHECK_GT(img_channels, 0);

    // 减去均值
    Dtype* mean = NULL;
    if (has_mean_file) {
      CHECK_EQ(cv_resized_image.channels(), data_mean_.channels());
      CHECK_EQ(cv_resized_image.rows, data_mean_.height());
      CHECK_EQ(cv_resized_image.cols, data_mean_.width());
      mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
      CHECK(mean_values_.size() == 1 || mean_values_.size() == cv_resized_image.channels()) <<
          "Specify either 1 mean_value or as many as channels: " << cv_resized_image.channels();
      if (cv_resized_image.channels() > 1 && mean_values_.size() == 1) {
        // Replicate the mean_value for simplicity
        for (int c = 1; c < cv_resized_image.channels(); ++c) {
          mean_values_.push_back(mean_values_[0]);
        }
      }
    }

  // 检查图片的数据
  CHECK(cv_resized_image.data);

  int resized_height = cv_resized_image.rows;
  int resized_width = cv_resized_image.cols;
  int resized_channels = cv_resized_image.channels();

  CHECK_EQ(resized_width, width);
  CHECK_EQ(resized_height, height);
  CHECK_EQ(resized_channels, channels);

  // 复制数据
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_resized_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * height + h_idx_real) * width + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
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
void DataTransformer<Dtype>::NormTransform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       const bool normalize,
                                       const vector<Dtype>& mean_value) {
  const int img_channels = cv_img.channels();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();
  CHECK_EQ(channels, img_channels);
  CHECK_EQ(img_channels, mean_value.size());
  CHECK_GE(num, 1);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte.";

  CHECK(param_.has_resize_param());
  cv::Mat resized_image;
  resized_image = ApplyResize(cv_img, param_.resize_param());

  CHECK(resized_image.data);

  int resized_height = resized_image.rows;
  int resized_width = resized_image.cols;
  int resized_channels = resized_image.channels();

  CHECK_EQ(resized_width, width);
  CHECK_EQ(resized_height, height);
  CHECK_EQ(resized_channels, channels);

  // 复制数据
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = resized_image.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (normalize) {
          transformed_data[top_index] = pixel / (Dtype)256 - (Dtype)0.5;
        } else {
          transformed_data[top_index] = pixel - mean_value[c];
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype* data, cv::Mat* cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
        "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        if (has_mean_file) {
          ptr[img_idx++] = static_cast<uchar>(data[idx] / scale + mean[idx]);
        } else {
          if (has_mean_values) {
            ptr[img_idx++] =
                static_cast<uchar>(data[idx] / scale + mean_values_[c]);
          } else {
            ptr[img_idx++] = static_cast<uchar>(data[idx] / scale);
          }
        }
      }
    }
  }

  if (param_.has_resize_param()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param());
  } else {
    *cv_img = orig_img;
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Blob<Dtype>* blob,
                                          vector<cv::Mat>* cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype* image_data = blob->cpu_data();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(cv_img, transformed_blob, &crop_bbox, &do_mirror);
}

template <typename Dtype>
void DataTransformer<Dtype>::EasyTransform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       bool* do_mirror) {
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // 断言
  CHECK_EQ(channels, img_channels);
  CHECK_EQ(height, img_height);
  CHECK_EQ(width, img_width);
  CHECK_EQ(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte.";

  // 获取scale
  const Dtype scale = param_.scale();
  // 随机镜像
  *do_mirror = param_.mirror() && Rand(2);

  // 均值
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  CHECK(!(has_mean_file == true && has_mean_values == true)) <<
      "the mean file and mean values could not be given in the same time.";

  // 减去均值
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  // 检查图片的数据
  CHECK(cv_img.data);
  // 复制数据
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * height + h_idx_real) * width + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
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
void DataTransformer<Dtype>::CropImage(const cv::Mat& img,
                                       const NormalizedBBox& bbox,
                                       cv::Mat* crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
                data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
                           input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }

  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.has_resize_param()) {
    if (param_.resize_param().has_height() &&
        param_.resize_param().height() > 0) {
      datum_height = param_.resize_param().height();
    }
    if (param_.resize_param().has_width() &&
        param_.resize_param().width() > 0) {
      datum_width = param_.resize_param().width();
    }
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h: datum_height;
  shape[3] = (crop_w)? crop_w: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template <typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.has_resize_param()) {
    if (param_.resize_param().has_height() &&
        param_.resize_param().height() > 0) {
      img_height = param_.resize_param().height();
    }
    if (param_.resize_param().has_width() &&
        param_.resize_param().width() > 0) {
      img_width = param_.resize_param().width();
    }
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;
  return shape;
}

template <typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
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
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

//--------------------------------------------------------------------------
template <typename Dtype>
bool DataTransformer<Dtype>::TransformAnnotationOfPerson(
    const vector<PersonBBox>& item_anno_person,
    const std::map<string,int>& part_name_label,
    const NormalizedBBox& crop_bbox, const bool& do_mirror,
    vector<AnnotationPart>* transformed_item_anno_part) {
  transformed_item_anno_part->clear();
  int body_count = 0;
  int head_count = 0;
  int torso_count = 0;
  int arm_count = 0;
  int leg_count = 0;
  int hand_count = 0;
  int foot_count = 0;
  if (part_name_label.find("body") == part_name_label.end()) {
    LOG(FATAL) << "body-part is not found.";
  }
  if (part_name_label.find("head") == part_name_label.end()) {
    LOG(FATAL) << "head-part is not found.";
  }
  if (part_name_label.find("torso") == part_name_label.end()) {
    LOG(FATAL) << "torso-part is not found.";
  }
  if (part_name_label.find("arm") == part_name_label.end()) {
    LOG(FATAL) << "arm-part is not found.";
  }
  if (part_name_label.find("leg") == part_name_label.end()) {
    LOG(FATAL) << "leg-part is not found.";
  }
  if (part_name_label.find("hand") == part_name_label.end()) {
    LOG(FATAL) << "hand-part is not found.";
  }
  if (part_name_label.find("foot") == part_name_label.end()) {
    LOG(FATAL) << "foot-part is not found.";
  }
  int label_body = part_name_label.find("body")->second;
  int label_head = part_name_label.find("head")->second;
  int label_torso = part_name_label.find("torso")->second;
  int label_arm = part_name_label.find("arm")->second;
  int label_leg = part_name_label.find("leg")->second;
  int label_hand = part_name_label.find("hand")->second;
  int label_foot = part_name_label.find("foot")->second;
  AnnotationPart anno_body;
  anno_body.set_part_name("body");
  anno_body.set_part_label(label_body);
  AnnotationPart anno_head;
  anno_head.set_part_name("head");
  anno_head.set_part_label(label_head);
  AnnotationPart anno_torso;
  anno_torso.set_part_name("torso");
  anno_torso.set_part_label(label_torso);
  AnnotationPart anno_arm;
  anno_arm.set_part_name("arm");
  anno_arm.set_part_label(label_arm);
  AnnotationPart anno_leg;
  anno_leg.set_part_name("leg");
  anno_leg.set_part_label(label_leg);
  AnnotationPart anno_hand;
  anno_hand.set_part_name("hand");
  anno_hand.set_part_label(label_hand);
  AnnotationPart anno_foot;
  anno_foot.set_part_name("foot");
  anno_foot.set_part_label(label_foot);
  // 遍历所有人物对象
  for (int i = 0; i < item_anno_person.size(); ++i) {
    // 获取该人物对象
    const PersonBBox& person = item_anno_person[i];
    // 检查body是否存在
    CHECK(person.has_body()) << "the body of person must be given.";
    // 转换后的bbox
    NormalizedBBox proj_bbox;
    /**
     * head
     */
    bool has_head = false;
    if (person.has_head()) {
      const NormalizedBBox& bbox = person.head();
      if (param_.has_emit_constraint() &&
        MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
        if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
          has_head = true;
          // 在head输出列表中创建一条标注记录
          Annotation_P* annopart_head =
              anno_head.add_annotation();
          // 设置其bodyID
          annopart_head->set_body_id(i);
          // 设置其实例ID
          annopart_head->set_instance_id(head_count++);
          // 设置其bbox数值
          NormalizedBBox* transformed_bbox = annopart_head->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          transformed_bbox->set_label(label_head);
          transformed_bbox->set_score(1.0);
          transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
          transformed_bbox->set_dir(bbox.dir());
          transformed_bbox->set_pose(0);
          transformed_bbox->set_truncated(0);
          transformed_bbox->set_main(0);
          // 裁剪不影响DIR,如果需要处理,使用emitType=1进行设置
          // 不满足,设为-1,表示不作为TP
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
        }
      }
    }
    /**
     * torso
     */
     bool has_torso = false;
     if (person.has_torso()) {
       const NormalizedBBox& bbox = person.torso();
       if (param_.has_emit_constraint() &&
         MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
         if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
           has_torso = true;
           // 在torso输出列表中创建一条标注记录
           Annotation_P* annopart_torso =
               anno_torso.add_annotation();
           // 设置其bodyID
           annopart_torso->set_body_id(i);
           // 设置其实例ID
           annopart_torso->set_instance_id(torso_count++);
           // 设置其bbox数值
           NormalizedBBox* transformed_bbox = annopart_torso->mutable_bbox();
           transformed_bbox->CopyFrom(proj_bbox);
           transformed_bbox->set_label(label_torso);
           transformed_bbox->set_score(1.0);
           transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
           transformed_bbox->set_dir(bbox.dir());
           transformed_bbox->set_pose(0);
           transformed_bbox->set_truncated(0);
           transformed_bbox->set_main(0);
           // 裁剪不影响DIR,如果需要处理,使用emitType=1进行设置
           // 不满足,设为-1,表示不作为TP
           if (do_mirror) {
             Dtype temp = transformed_bbox->xmin();
             transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
             transformed_bbox->set_xmax(1 - temp);
           }
         }
       }
     }
     /**
      * arms
      */
     bool has_arm = false;
     if (person.arm_size() > 0){
       CHECK_LE(person.arm_size(), 2) << "Not more than 2 arms for a person.";
      //  遍历所有arm
       for (int j = 0; j < person.arm_size(); ++j){
        //  获取该arm
         const NormalizedBBox& bbox = person.arm(j);
        //  满足发布条件
         if (param_.has_emit_constraint() &&
           MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
           if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
            //  存在arm
             has_arm = true;
             // 在arm输出列表中创建一条标注记录
             Annotation_P* annopart_arm =
                 anno_arm.add_annotation();
             // 设置其bodyID
             annopart_arm->set_body_id(i);
             // 设置其实例ID
             annopart_arm->set_instance_id(arm_count++);
             // 设置其bbox数值
             NormalizedBBox* transformed_bbox = annopart_arm->mutable_bbox();
             transformed_bbox->CopyFrom(proj_bbox);
             transformed_bbox->set_label(label_arm);
             transformed_bbox->set_score(1.0);
             transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
             transformed_bbox->set_pose(0);
             transformed_bbox->set_dir(0);
             transformed_bbox->set_truncated(0);
             transformed_bbox->set_main(0);
             // 镜像
             if (do_mirror) {
               Dtype temp = transformed_bbox->xmin();
               transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
               transformed_bbox->set_xmax(1 - temp);
             }
           }
         }
       }
     }
    /**
     * legs
     */
     bool has_leg = false;
     if (person.leg_size() > 0){
       CHECK_LE(person.leg_size(), 2) << "Not more than 2 legs for a person.";
       for (int k = 0; k < person.leg_size(); ++k){
        //  获取该arm
         const NormalizedBBox& bbox = person.leg(k);
        //  满足发布条件
         if (param_.has_emit_constraint() &&
           MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
           if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
            //  存在arm
             has_leg = true;
             // 在leg输出列表中创建一条标注记录
             Annotation_P* annopart_leg =
                 anno_leg.add_annotation();
             // 设置其bodyID
             annopart_leg->set_body_id(i);
             // 设置其实例ID
             annopart_leg->set_instance_id(leg_count++);
             // 设置其bbox数值
             NormalizedBBox* transformed_bbox = annopart_leg->mutable_bbox();
             transformed_bbox->CopyFrom(proj_bbox);
             transformed_bbox->set_label(label_leg);
             transformed_bbox->set_score(1.0);
             transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
             transformed_bbox->set_pose(0);
             transformed_bbox->set_dir(0);
             transformed_bbox->set_truncated(0);
             transformed_bbox->set_main(0);
             if (do_mirror) {
               Dtype temp = transformed_bbox->xmin();
               transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
               transformed_bbox->set_xmax(1 - temp);
             }
           }
         }
       }
     }
     /**
      * hand
      */
      bool has_hand = false;
      if (person.hand_size() > 0){
        CHECK_LE(person.hand_size(), 2) << "Not more than 2 hands for a person.";
        for (int m = 0; m < person.hand_size(); ++m){
         //  获取该arm
          const NormalizedBBox& bbox = person.hand(m);
         //  满足发布条件
          if (param_.has_emit_constraint() &&
            MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
            if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
              has_hand = true;
              // 在hand输出列表中创建一条标注记录
              Annotation_P* annopart_hand =
                  anno_hand.add_annotation();
              // 设置其bodyID
              annopart_hand->set_body_id(i);
              // 设置其实例ID
              annopart_hand->set_instance_id(hand_count++);
              // 设置其bbox数值
              NormalizedBBox* transformed_bbox = annopart_hand->mutable_bbox();
              transformed_bbox->CopyFrom(proj_bbox);
              transformed_bbox->set_label(label_hand);
              transformed_bbox->set_score(1.0);
              transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
              transformed_bbox->set_pose(0);
              transformed_bbox->set_dir(0);
              transformed_bbox->set_truncated(0);
              transformed_bbox->set_main(0);
              if (do_mirror) {
                Dtype temp = transformed_bbox->xmin();
                transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
                transformed_bbox->set_xmax(1 - temp);
              }
            }
          }
        }
      }

      /**
       * foot
       */
       bool has_foot = false;
       if (person.foot_size() > 0){
         CHECK_LE(person.foot_size(), 2) << "Not more than 2 foots for a person.";
         for (int n = 0; n < person.foot_size(); ++n){
           const NormalizedBBox& bbox = person.foot(n);
          //  满足发布条件
           if (param_.has_emit_constraint() &&
             MeetEmitConstraint(crop_bbox, bbox, param_.emit_constraint())) {
             if (ProjectBBox(crop_bbox, bbox, &proj_bbox)) {
              //  存在arm
               has_foot = true;
               // 在foot输出列表中创建一条标注记录
               Annotation_P* annopart_foot =
                   anno_foot.add_annotation();
               // 设置其bodyID
               annopart_foot->set_body_id(i);
               // 设置其实例ID
               annopart_foot->set_instance_id(foot_count++);
               // 设置其bbox数值
               NormalizedBBox* transformed_bbox = annopart_foot->mutable_bbox();
               transformed_bbox->CopyFrom(proj_bbox);
               transformed_bbox->set_label(label_foot);
               transformed_bbox->set_score(1.0);
               transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
               transformed_bbox->set_pose(0);
               transformed_bbox->set_dir(0);
               transformed_bbox->set_truncated(0);
               transformed_bbox->set_main(0);
               if (do_mirror) {
                 Dtype temp = transformed_bbox->xmin();
                 transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
                 transformed_bbox->set_xmax(1 - temp);
               }
             }
           }
         }
       }
       /**
        * body
        */
      bool has_body = has_head || has_torso || has_arm || has_leg || has_hand || has_foot;
      const NormalizedBBox& tp_bbox = person.body();
      if (param_.has_emit_constraint() &&
        MeetEmitConstraint(crop_bbox, tp_bbox, param_.emit_constraint())){
        has_body = true;
      }
      if (has_body) {
        if (ProjectBBox(crop_bbox, tp_bbox, &proj_bbox)) {
          //记录body
          Annotation_P* annopart_body =
              anno_body.add_annotation();
          // 设置其bodyID
          annopart_body->set_body_id(i);
          // 设置其实例ID
          annopart_body->set_instance_id(body_count++);
          // 设置其bbox数值
          NormalizedBBox* transformed_bbox = annopart_body->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          transformed_bbox->set_label(label_body);
          transformed_bbox->set_score(1.0);
          transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
          transformed_bbox->set_truncated(tp_bbox.truncated());
          transformed_bbox->set_main(tp_bbox.main());
          transformed_bbox->set_dir(0);
          // 设置pose:满足pose的发布规则,保留其pose-label,否则设置为0
          // 务必使用min_overlap规则
          // 如果不满足,则设置为0 -> ignored
          if (param_.has_emit_constraint() &&
            MeetEmitConstraint(crop_bbox, tp_bbox, param_.emit_constraint(), 2)) {
            transformed_bbox->set_pose(tp_bbox.pose());
          }
          else {
            transformed_bbox->set_pose(0);
          }
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
        } else {
          LOG(FATAL) << "The annotation has encountered mistakes. The body is NULL while some parts are found.";
        }
     }
  }
  // 写入最终的vector之中
  transformed_item_anno_part->push_back(anno_body);
  transformed_item_anno_part->push_back(anno_head);
  transformed_item_anno_part->push_back(anno_torso);
  transformed_item_anno_part->push_back(anno_arm);
  transformed_item_anno_part->push_back(anno_leg);
  transformed_item_anno_part->push_back(anno_hand);
  transformed_item_anno_part->push_back(anno_foot);
  //print INFO of this image
  LOG(INFO) << "The image after random-crop has "
            << body_count << " persons, " << head_count
            << " heads, " << torso_count << " torsos, "
            << arm_count << " arms, " << leg_count
            << " legs, " << hand_count << " hands, "
            << foot_count << "foots.";
  return true;
}
//--------------------------------------------------------------------------
template <typename Dtype>
bool DataTransformer<Dtype>::TransformAnnotationOfPerson(
    const vector<PersonBBox>& item_anno_person,
    const NormalizedBBox& crop_bbox, const bool& do_mirror,
    vector<AnnotationPart>* transformed_item_anno_part) {
  transformed_item_anno_part->clear();

  AnnotationPart anno_body;
  anno_body.set_part_name("body");
  anno_body.set_part_label(1);

  // 遍历所有人物对象
  for (int i = 0; i < item_anno_person.size(); ++i) {
    // 获取该人物对象
    const PersonBBox& person = item_anno_person[i];
    // 检查body是否存在
    CHECK(person.has_body()) << "the body of person must be given.";
    // 转换后的bbox
    NormalizedBBox proj_bbox;
    const NormalizedBBox& body = person.body();
    // print the INFO of body
    // LOG(INFO) << "Before Anno-Trans: [ " << body.xmin() << " " << body.ymin()
    //           << " " << body.xmax() << " " << body.ymax() << " ]";
    if (param_.has_emit_constraint() &&
        MeetEmitConstraint(crop_bbox, body, param_.emit_constraint()) &&
        ProjectBBox(crop_bbox, body, &proj_bbox)){
      //记录body
      Annotation_P* annopart_body =
          anno_body.add_annotation();
      // 设置其bodyID
      annopart_body->set_body_id(i);
      // 设置其实例ID
      annopart_body->set_instance_id(i);
      // 设置其bbox数值
      NormalizedBBox* transformed_bbox = annopart_body->mutable_bbox();
      transformed_bbox->CopyFrom(proj_bbox);
      transformed_bbox->set_label(1);
      transformed_bbox->set_score(1.0);
      transformed_bbox->set_size((proj_bbox.xmax()-proj_bbox.xmin()) * (proj_bbox.ymax()-proj_bbox.ymin()));
      if (do_mirror) {
        Dtype temp = transformed_bbox->xmin();
        transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
        transformed_bbox->set_xmax(1 - temp);
      }
      // print the INFO
      // LOG(INFO) << "After Anno-Trans: [ " << transformed_bbox->xmin() << " "
      //           << transformed_bbox->ymin() << " " << transformed_bbox->xmax()
      //           << " " << transformed_bbox->ymax() << " ]";
   } else {
     DLOG(INFO) << "A person has been rejected after random-crop.";
   }
  }
  // 写入最终的vector之中
  transformed_item_anno_part->push_back(anno_body);
  //print INFO of this image
  // LOG(INFO) << anno_body.annotation_size() << " persons are left after random-crop.";
  return true;
}

template <typename Dtype>
bool DataTransformer<Dtype>::TransformAnnotationOfPerson(
    const vector<PersonBBox>& item_anno_person,
    vector<AnnotationPart>* transformed_item_anno_part) {
  transformed_item_anno_part->clear();

  AnnotationPart anno_body;
  anno_body.set_part_name("body");
  anno_body.set_part_label(1);

  // 遍历所有人物对象
  for (int i = 0; i < item_anno_person.size(); ++i) {
    // 获取该人物对象
    const PersonBBox& person = item_anno_person[i];
    // 检查body是否存在
    CHECK(person.has_body()) << "the body of person must be given.";
    const NormalizedBBox& body = person.body();
    //记录body
    Annotation_P* annopart_body =
        anno_body.add_annotation();
    // 设置其bodyID
    annopart_body->set_body_id(i);
    // 设置其实例ID
    annopart_body->set_instance_id(i);
    // 设置其bbox数值
    NormalizedBBox* transformed_bbox = annopart_body->mutable_bbox();
    transformed_bbox->CopyFrom(body);
    transformed_bbox->set_label(1);
    transformed_bbox->set_score(1.0);
    transformed_bbox->set_size((body.xmax()-body.xmin()) * (body.ymax()-body.ymin()));
  }
  transformed_item_anno_part->push_back(anno_body);
  return true;
}

template <typename Dtype>
bool DataTransformer<Dtype>::TransformAnnotationOfPerson(
    const vector<PersonBBox>& item_anno_person,
    vector<AnnotationGroup>* transformed_item_anno) {
  transformed_item_anno->clear();

  int person_label = 1;

  AnnotationGroup anno_body;
  anno_body.set_group_label(person_label);

  // 遍历所有人物对象
  for (int i = 0; i < item_anno_person.size(); ++i) {
    // 获取该人物对象
    const PersonBBox& person = item_anno_person[i];
    // 检查body是否存在
    CHECK(person.has_body()) << "the body of person must be given.";
    const NormalizedBBox& body = person.body();
    //记录body
    Annotation* anno =
        anno_body.add_annotation();
    // 设置ID
    anno->set_instance_id(i);
    // 设置其bbox数值
    NormalizedBBox* transformed_bbox = anno->mutable_bbox();
    transformed_bbox->CopyFrom(body);
    transformed_bbox->set_label(person_label);
    transformed_bbox->set_difficult(body.difficult());
  }
  transformed_item_anno->push_back(anno_body);
  return true;
}

template <typename Dtype>
bool DataTransformer<Dtype>::CropBoxSampler(
         const vector<AnnotationPart> &item_anno_part,
         NormalizedBBox *crop_bbox){
  const RandomCropParameter& crop_param = param_.crop_param();
  // CHECK(crop_param.has_min_scale());
  // CHECK(crop_param.has_max_scale());
  // CHECK(crop_param.has_min_aspect());
  // CHECK(crop_param.has_max_aspect());
  float min_scale = crop_param.min_scale();
  float max_scale = crop_param.max_scale();
  float min_aspect = crop_param.min_aspect();
  float max_aspect = crop_param.max_aspect();
  int max_sample_size = crop_param.max_sample_size();

  // LOG(INFO) << "min_scale: " << min_scale;
  // LOG(INFO) << "max_scale: " << max_scale;
  // LOG(INFO) << "min_aspect: " << min_aspect;
  // LOG(INFO) << "max_aspect: " << max_aspect;
  // LOG(INFO) << "max_sample_size: " << max_sample_size;

  float scale, aspect_ratio;
  float min_aspect_ratio,max_aspect_ratio;
  float bbox_width, bbox_height;
  float w_off, h_off;
  for (int i = 0; i < max_sample_size; i++) {
    caffe_rng_uniform(1,min_scale,max_scale,&scale);
    min_aspect_ratio = std::max<float>(min_aspect,
                                      std::pow(scale, 2.));
    max_aspect_ratio = std::min<float>(max_aspect,
                                  1. / std::pow(scale, 2.));
    caffe_rng_uniform(1, min_aspect_ratio, max_aspect_ratio, &aspect_ratio);
    bbox_width = scale * sqrt(aspect_ratio);
    bbox_height = scale / sqrt(aspect_ratio);

    // LOG(INFO) << "bbox_height: " << bbox_height;
    // LOG(INFO) << "bbox_width: " << bbox_width;

    if (phase_ == TRAIN) {
      caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
      caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);
    } else if (phase_ == TEST) {
      w_off = (1 - bbox_width) / 2.;
      w_off = (1 - bbox_height) / 2.;
    } else {
      LOG(FATAL) << "Illegal phase state.";
    }
    crop_bbox->set_xmin(w_off);
    crop_bbox->set_ymin(h_off);
    crop_bbox->set_xmax(w_off + bbox_width);
    crop_bbox->set_ymax(h_off + bbox_height);

    // LOG(INFO) << "xmin: " << crop_bbox->xmin();
    // LOG(INFO) << "ymin: " << crop_bbox->ymin();
    // LOG(INFO) << "xmax: " << crop_bbox->xmax();
    // LOG(INFO) << "ymax: " << crop_bbox->ymax();

    // 遍历所有的标注结果，如果某个gtbox能够发布，则直接退出
    for (int j = 0; j < item_anno_part.size(); ++j) {
      const AnnotationPart& annopart = item_anno_part[j];
      for (int k = 0; k < annopart.annotation_size(); ++k) {
        const Annotation_P& anp = annopart.annotation(k);
        const NormalizedBBox& gtbox = anp.bbox();
        if (param_.has_emit_constraint() &&
                  MeetEmitConstraint(*crop_bbox,gtbox,
                  param_.emit_constraint())) {
            // LOG(INFO) << "[Sampler Success]";
            return true;
        }
      }
    }
  }
  // 采样最大次数，仍然失败，返回False
  // LOG(INFO) << "[Sampler Failed]";
  return false;
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const vector<AnnotationPart> &item_anno_part,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    const float boxsize_thre,
    vector<AnnotationPart>* transformed_item_anno_part) {
    transformed_item_anno_part->clear();

    for (int i = 0; i < item_anno_part.size(); ++i) {
      // 创建一个类标记组
      const AnnotationPart &ap = item_anno_part[i];
      AnnotationPart transformed_ap;
      transformed_ap.set_part_name(ap.part_name());
      transformed_ap.set_part_label(ap.part_label());
      // 遍历该类下所有实例
      for (int j = 0; j < ap.annotation_size(); ++j) {
        const Annotation_P& apart = ap.annotation(j);
        NormalizedBBox proj_bbox;
        const NormalizedBBox& gtbox = apart.bbox();
        // LOG(INFO) << "Before Anno-Trans: [ " << gtbox.xmin() << " " << gtbox.ymin()
        //           << " " << gtbox.xmax() << " " << gtbox.ymax() << " ]";
        if (param_.has_emit_constraint() &&
            MeetEmitConstraint(crop_bbox, gtbox, param_.emit_constraint()) &&
            ProjectBBox(crop_bbox, gtbox, &proj_bbox)) {
            if (BBoxSize(proj_bbox) > boxsize_thre) {
              // 创建标注
              Annotation_P* part = transformed_ap.add_annotation();
              part->set_body_id(apart.body_id());
              part->set_instance_id(apart.instance_id());
              NormalizedBBox* transformed_bbox = part->mutable_bbox();
              transformed_bbox->CopyFrom(proj_bbox);
              transformed_bbox->set_label(gtbox.label());
              transformed_bbox->set_difficult(gtbox.difficult());
              if (do_mirror) {
                Dtype temp = transformed_bbox->xmin();
                transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
                transformed_bbox->set_xmax(1 - temp);
              }
            } else {
              DLOG(INFO) << "[Note] A person is rejected due to small size after random-crop.";
            }
            // LOG(INFO) << "After Anno-Trans: [ " << transformed_bbox->xmin() << " "
            //           << transformed_bbox->ymin() << " " << transformed_bbox->xmax()
            //           << " " << transformed_bbox->ymax() << " ]";
        } else {
            DLOG(INFO) << "[Note] A person has been rejected for emit_constraints after random-crop.";
        }
      }
      // LOG(INFO) << "[ANNO-CROP] -> " << transformed_ap.annotation_size()
      //           << "persons have been found after random-crop.";
      transformed_item_anno_part->push_back(transformed_ap);
    }
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const vector<AnnotationGroup>& anno_group,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    const float boxsize_thre,
    vector<AnnotationGroup>* transformed_anno_group) {
    transformed_anno_group->clear();

    if (phase_ == TRAIN) {
      CHECK(param_.has_emit_constraint())
          << "The emit_constraint must be specified in"
          << " TransformAnnotation functions during TRAIN.";
    }

    for (int i = 0; i < anno_group.size(); ++i) {
      // 遍历各个类
      const AnnotationGroup &anno = anno_group[i];
      const int label = anno.group_label();
      AnnotationGroup transformed_anno;
      transformed_anno.set_group_label(label);
      int obj_count = 0;
      // 遍历该类下所有实例
      for (int j = 0; j < anno.annotation_size(); ++j) {
        const Annotation& an = anno.annotation(j);
        NormalizedBBox proj_bbox;
        const NormalizedBBox& gtbox = an.bbox();
        // TRAIN：满足发布条件
        // TEST：直接转换
        if ((param_.has_emit_constraint() &&
            MeetEmitConstraint(crop_bbox, gtbox, param_.emit_constraint()))
            || (phase_ == TEST)) {
            // 必须是有效的box
            if (ProjectBBox(crop_bbox, gtbox, &proj_bbox)){
              // 太小的box直接滤掉
              if (BBoxSize(proj_bbox) > boxsize_thre) {
                // 创建标注
                Annotation* obj = transformed_anno.add_annotation();
                obj->set_instance_id(obj_count++);
                NormalizedBBox* transformed_bbox = obj->mutable_bbox();
                transformed_bbox->CopyFrom(proj_bbox);
                transformed_bbox->set_difficult(gtbox.difficult());
                transformed_bbox->set_label(label);
                if (do_mirror) {
                  Dtype temp = transformed_bbox->xmin();
                  transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
                  transformed_bbox->set_xmax(1 - temp);
                }
              } else {
                DLOG(INFO) << "[Note] A person is rejected due to small size after random-crop.";
              }
          }
        } else {
          DLOG(INFO) << "[Note] A person has been rejected for emit_constraints after random-crop during TRAIN.";
        }
      }
      transformed_anno_group->push_back(transformed_anno);
    }
}

INSTANTIATE_CLASS(DataTransformer);
}  // namespace caffe
