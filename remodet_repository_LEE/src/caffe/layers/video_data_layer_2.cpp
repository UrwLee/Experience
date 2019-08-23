#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdint.h>
#include <algorithm>
#include <csignal>
#include <map>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_data_layer_2.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/util/myimg_proc.hpp"

namespace caffe {

template <typename Dtype>
VideoData2Layer<Dtype>::VideoData2Layer(const LayerParameter& param)
  : BasePrefetchingData2Layer<Dtype>(param) {
}

template <typename Dtype>
VideoData2Layer<Dtype>::~VideoData2Layer() {
  this->StopInternalThread();
  if (cap_.isOpened()) {
    cap_.release();
  }
}

template <typename Dtype>
void VideoData2Layer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // get the batch Size
  const int batch_size = this->layer_param_.data_param().batch_size();
  // get the video param
  const VideoDataParameter& video_data_param =
      this->layer_param_.video_data_param();
  // get the video type : webcam or video
  video_type_ = video_data_param.video_type();

  /**
  * define the size of webcam
  */
 if (video_type_ == VideoDataParameter_VideoType_WEBCAM){
   webcam_width_ = video_data_param.webcam_width();
   webcam_height_ = video_data_param.webcam_height();
   crop_width_ = video_data_param.crop_width();
   crop_height_ = video_data_param.crop_height();

   CHECK_EQ(webcam_height_, crop_height_) << "the webcam height and crop height should be equal.";
   CHECK_GE(webcam_width_, crop_width_) << "the webcam width should be larger or equal to the crop width.";

   contrast_scale_ = video_data_param.contrast_scale();
   exposure_ = video_data_param.exposure();
   medianblur_ksize_ = video_data_param.medianblur_ksize();
 }
  // Read an image, and use it to initialize the top blob.
  // top[0] -> resized transformed_data_
  // top[1] -> orig image
  /**
   * 尝试读入视频流
   */
  cv::Mat cv_img;
  cv::Mat src_img;
  if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
    const int device_id = video_data_param.device_id();
    if (!cap_.open(device_id)) {
      LOG(FATAL) << "Failed to open webcam: " << device_id;
    }
    // set the cam resolution
    cap_.set(CV_CAP_PROP_FRAME_WIDTH,webcam_width_);
    cap_.set(CV_CAP_PROP_FRAME_HEIGHT,webcam_height_);

    cap_ >> src_img;
    // crop
    int cv_width_min = webcam_width_ / 2 - crop_width_ / 2;
    int cv_width_max = cv_width_min + crop_width_;
    int cv_height_min = 0;
    int cv_height_max = crop_height_;

    cv_img = src_img(cv::Range(cv_height_min,cv_height_max), cv::Range(cv_width_min,cv_width_max));

  } else if (video_type_ == VideoDataParameter_VideoType_VIDEO) {
    CHECK(video_data_param.has_video_file()) << "Must provide video file!";
    const string& video_file = video_data_param.video_file();
    if (!cap_.open(video_file)) {
      LOG(FATAL) << "Failed to open video: " << video_file;
    }
    total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    processed_frames_ = 0;
    // Read image to infer shape.
    cap_ >> cv_img;
    // Set index back to the first frame.
    cap_.set(CV_CAP_PROP_POS_FRAMES, 0);
  } else {
    LOG(FATAL) << "Unknow video type!";
  }
  /**
   * 读入是否成功?
   */
  CHECK(cv_img.data) << "Could not load image!";
  // 初始化输出shape0
  top0_shape_ = this->data_transformer_->InferBlobShape(cv_img);
  // 对变换器的输出shape进行设置
  this->transformed_data_.Reshape(top0_shape_);
  top0_shape_[0] = batch_size;
  // 对top[0]进行设置
  top[0]->Reshape(top0_shape_);
  /**
   * 对prefetch_队列进行shape
   */
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top0_shape_);
  }
  LOG(INFO) << "output data [0] size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  /**
   * top[1] -> orig data_
   */
  const int img_height = cv_img.rows;
  const int img_width  = cv_img.cols;
  const int img_channels = cv_img.channels();
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK_GT(img_channels, 0);

  vector<int> shape(4);
  shape[0] = batch_size;
  shape[1] = img_channels;
  shape[2] = img_height;
  shape[3] = img_width;
  top1_shape_ = shape;
  top[1]->Reshape(top1_shape_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].orig_data_.Reshape(top1_shape_);
  }
  LOG(INFO) << "output data [1] size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[2]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void VideoData2Layer<Dtype>::load_batch_orig(Batch_Orig<Dtype>* batch_orig) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch_orig->data_.count());
  CHECK(batch_orig->orig_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  top0_shape_[0] = 1;
  this->transformed_data_.Reshape(top0_shape_);
  // Reshape batch according to the batch_size.
  top0_shape_[0] = batch_size;
  batch_orig->data_.Reshape(top0_shape_);
  //reshape top[1]
  batch_orig->orig_data_.Reshape(top1_shape_);

  Dtype* top0_data = batch_orig->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = batch_orig->label_.mutable_cpu_data();
  }

  //load_batch
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    cv::Mat cv_img;
    cv::Mat src_img;
    cv::Mat temp_img;
    if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
      cap_ >> src_img;
      // 裁剪
      int cv_width_min = webcam_width_ / 2 - crop_width_ / 2;
      int cv_width_max = cv_width_min + crop_width_;
      int cv_height_min = 0;
      int cv_height_max = crop_height_;
      cv_img = src_img(cv::Range(cv_height_min,cv_height_max), cv::Range(cv_width_min,cv_width_max));
      //增强
      cv_img.convertTo(cv_img,-1,contrast_scale_,exposure_);
      if(medianblur_ksize_ >= 3){
        cv::medianBlur(cv_img,temp_img,medianblur_ksize_);
        sharp_2D(temp_img,cv_img);
      }
      // // 图像增强，测试
      // denoised_img.convertTo(denoised_img,-1,alpha_contrast,beta_exposure);
      // grayworld_awb_single(denoised_img,cv_img);
      // dynamic_awb(denoised_img,cv_img,0.1);
    } else if (video_type_ == VideoDataParameter_VideoType_VIDEO) {
      if (processed_frames_ >= total_frames_) {
        LOG(INFO) << "Finished processing video.";
        raise(SIGINT);
      }
      ++processed_frames_;
      cap_ >> cv_img;
    } else {
      LOG(FATAL) << "Unknown video type.";
    }
    CHECK(cv_img.data) << "Could not load image!";
    read_time += timer.MicroSeconds();
    /**
     * the img is loaded
     */
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset0 = batch_orig->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top0_data + offset0);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    //load top[1]
    //---------------------------------------------------------
    int offset1 = batch_orig->orig_data_.offset(item_id);
    //指向batch中待保存的位置
    Dtype* top1_data = batch_orig->orig_data_.mutable_cpu_data() + offset1;
    int top1_index;
    for(int h = 0; h < cv_img.rows; ++h){
      // 指向该行的数据指针
      const uchar* ptr = cv_img.ptr<uchar>(h);
      // 该行数据的索引
      // 按照cv的数据格式遍历
      // h-w-c
      int img_index = 0;
      for(int w = 0; w < cv_img.cols; ++w){
        for(int c = 0; c < cv_img.channels(); ++c){
          top1_index = (c * cv_img.rows + h) * cv_img.cols + w;
          // uchar数据保存为Dtype
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          // 写入到batch
          top1_data[top1_index] = pixel;
        }
      }
    }
    //----------------------------------------------------------
    if (this->output_labels_) {
      top_label[item_id] = 0;
    }
    trans_time += timer.MicroSeconds();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoData2Layer);
REGISTER_LAYER_CLASS(VideoData2);

}  // namespace caffe
#endif  // USE_OPENCV
