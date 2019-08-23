#include "caffe/tracker/fe_roi_maker.hpp"

namespace caffe {

template <typename Dtype>
FERoiMaker<Dtype>::FERoiMaker(const std::string& network_proto,
                              const std::string& caffe_model,
                              const int gpu_id,
                              const std::string& features,
                              const int resized_width,
                              const int resized_height) {
  // Get Net
  LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  net_.reset(new Net<Dtype>(network_proto, caffe::TEST));
  if (caffe_model != "NONE") {
    net_->CopyTrainedLayersFrom(caffe_model);
  } else {
    LOG(FATAL) << "Must define a pre-trained model.";
  }
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  input_width_ = input_layer->width();
  input_height_ = input_layer->height();
  LOG(INFO) << "Network requires input size: (width, height) "
            << input_width_ << ", " << input_height_;
  CHECK_EQ(input_layer->channels(), 3) << "Input layer should have 3 channels.";
  // Get Layer
  LayerParameter layer_param;
  layer_param.set_name("roi_resize_layer");
  layer_param.set_type("RoiResize");
  RoiResizeParameter* roi_resize_param = layer_param.mutable_roi_resize_param();
  roi_resize_param->set_target_spatial_width(resized_width);
  roi_resize_param->set_target_spatial_height(resized_height);
  roi_resize_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  vector<Blob<Dtype>*> resize_bottom_vec;
  vector<Blob<Dtype>*> resize_top_vec;
  Blob<Dtype> bot_0(1,1,12,12);
  Blob<Dtype> bot_1(1,1,1,4);
  Blob<Dtype> top_0(1,1,resized_height,resized_width);
  resize_bottom_vec.push_back(&bot_0);
  resize_bottom_vec.push_back(&bot_1);
  resize_top_vec.push_back(&top_0);
  roi_resize_layer_->SetUp(resize_bottom_vec, resize_top_vec);
  // features
  features_ = features;
}

template <typename Dtype>
void FERoiMaker<Dtype>::getFeatures(const std::string& feature, const Blob<Dtype>* map) {
  const boost::shared_ptr<Blob<Dtype> > feature_blob = net_->blob_by_name(feature.c_str());
  map = feature_blob.get();
}

template <typename Dtype>
void FERoiMaker<Dtype>::load(const cv::Mat& image) {
  Blob<Dtype>* input_blob = net_->input_blobs()[0];
  CHECK_EQ(image.channels(), 3);
  int w = image.cols;
  int h = image.rows;
  CHECK_GT(w,0);
  CHECK_GT(h,0);
  if (!image.data) {
    LOG(FATAL) << "Image data open failed.";
  }
  Dtype scalar = sqrt(512*288/w/h);
  cv::Mat resized_image;
  cv::resize(image,resized_image,cv::Size(),scalar,scalar,CV_INTER_LINEAR);
  input_blob->Reshape(1, 3, resized_image.rows, resized_image.cols);
  net_->Reshape();
  Dtype* transformed_data = input_blob->mutable_cpu_data();
  const int offset = resized_image.rows * resized_image.cols;
  bool normalize = false;
  for (int i = 0; i < resized_image.rows; ++i) {
    for (int j = 0; j < resized_image.cols; ++j) {
      cv::Vec3b& rgb = resized_image.at<cv::Vec3b>(i, j);
      if (normalize) {
        transformed_data[i * resized_image.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[offset + i * resized_image.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2 * offset + i * resized_image.cols + j] = (rgb[2] - 128)/256.0;
      } else {
        transformed_data[i * resized_image.cols + j] = rgb[0] - 104;
        transformed_data[offset + i * resized_image.cols + j] = rgb[1] - 117;;
        transformed_data[2 * offset + i * resized_image.cols + j] = rgb[2] - 123;;
      }
    }
  }
}

template <typename Dtype>
void FERoiMaker<Dtype>::roi_resize(const Blob<Dtype>* feature, const BoundingBox<Dtype>& bbox,
                                   Blob<Dtype>* resize_fmap) {
  // get bbox
  Blob<Dtype> bbox_blob(1,1,1,4);
  Dtype* bbox_data = bbox_blob.mutable_cpu_data();
  bbox_data[0] = bbox.x1_;
  bbox_data[1] = bbox.y1_;
  bbox_data[2] = bbox.x2_;
  bbox_data[3] = bbox.y2_;
  vector<Blob<Dtype>*> resize_bottom_vec;
  vector<Blob<Dtype>*> resize_top_vec;
  Blob<Dtype> fmap;
  fmap.ReshapeLike(*feature);
  fmap.ShareData(*feature);
  resize_bottom_vec.push_back(&fmap);
  resize_bottom_vec.push_back(&bbox_blob);
  resize_top_vec.push_back(resize_fmap);
  roi_resize_layer_->Reshape(resize_bottom_vec, resize_top_vec);
  roi_resize_layer_->Forward(resize_bottom_vec, resize_top_vec);
}

template <typename Dtype>
void FERoiMaker<Dtype>::get_features(const cv::Mat& image, const BoundingBox<Dtype>& bbox,
                                     Blob<Dtype>* resized_fmap) {
  // Load
  load(image);
  // Forward
  net_->Forward();
  // Get features
  const Blob<Dtype>* feature_blob = NULL;
  getFeatures(features_, feature_blob);
  // roi resize
  roi_resize(feature_blob,bbox,resized_fmap);
  // done.
}

template <typename Dtype>
void FERoiMaker<Dtype>::get_fmap(const cv::Mat& image, const BoundingBox<Dtype>& bbox,
                                 Blob<Dtype>* resized_fmap, Blob<Dtype>* fmap) {
   // Load
   load(image);
   // Forward
   net_->Forward();
   // Get features
   getFeatures(features_, fmap);
   // roi resize
   roi_resize(fmap,bbox,resized_fmap);
}

template <typename Dtype>
void FERoiMaker<Dtype>::get_features(const cv::Mat& image_prev,
                                     const cv::Mat& image_curr,
                                     const BoundingBox<Dtype>& bbox_prev,
                                     const BoundingBox<Dtype>& bbox_curr,
                                     Blob<Dtype>* resized_prev,
                                     Blob<Dtype>* resized_curr) {
 get_features(image_prev, bbox_prev, resized_prev);
 get_features(image_curr, bbox_curr, resized_curr);
}

template <typename Dtype>
void FERoiMaker<Dtype>::get_features(const cv::Mat& image_prev,
                                     const cv::Mat& image_curr,
                                     const BoundingBox<Dtype>& bbox_prev,
                                     const BoundingBox<Dtype>& bbox_curr,
                                     Blob<Dtype>* resized_unified_map) {
  Blob<Dtype> resized_prev;
  Blob<Dtype> resized_curr;
  get_features(image_prev, bbox_prev, &resized_prev);
  get_features(image_curr, bbox_curr, &resized_curr);
  CHECK_EQ(resized_prev.num(), resized_curr.num());
  CHECK_EQ(resized_prev.channels(), resized_curr.channels());
  CHECK_EQ(resized_prev.height(), resized_curr.height());
  CHECK_EQ(resized_prev.width(), resized_curr.width());
  resized_unified_map->Reshape(1,2*resized_prev.channels(),resized_prev.height(),resized_prev.width());
  caffe_copy(resized_prev.count(),resized_prev.cpu_data(),resized_unified_map->mutable_cpu_data());
  caffe_copy(resized_curr.count(),resized_curr.cpu_data(),
             resized_unified_map->mutable_cpu_data()+resized_prev.count());
}

INSTANTIATE_CLASS(FERoiMaker);

}
