#include "caffe/tracker/regressor.hpp"

namespace caffe {

using caffe::Blob;
using caffe::Net;
using std::string;

template <typename Dtype>
Regressor<Dtype>::Regressor(const string& network_proto,
                            const string& caffe_model,
                            const int gpu_id,
                            const int num_inputs)
  : num_inputs_(num_inputs), caffe_model_(caffe_model), modified_params_(false) {
  SetupNetwork(network_proto, caffe_model, gpu_id);
}

template <typename Dtype>
Regressor<Dtype>::Regressor(const string& network_proto,
                            const string& caffe_model,
                            const int gpu_id)
  : num_inputs_(1), caffe_model_(caffe_model), modified_params_(false) {
  SetupNetwork(network_proto, caffe_model, gpu_id);
}

template <typename Dtype>
Regressor<Dtype>::Regressor(const boost::shared_ptr<caffe::Net<Dtype> >& net,
                            const int gpu_id) {
#ifdef CPU_ONLY
  LOG(INFO) << "Using CPU mode in Caffe.";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  this->net_ = net;
  CHECK_EQ(this->net_->num_outputs(), 1) << "Network should have exactly one output.";
  Blob<Dtype>* input_layer = this->net_->input_blobs()[0];
  LOG(INFO) << "Network requires input size: (width, height) "
            << input_layer->width() << ", " << input_layer->height();
  num_channels_ = input_layer->channels();
  CHECK_EQ(num_channels_, 3) << "Input layer should have 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

template <typename Dtype>
void Regressor<Dtype>::SetupNetwork(const string& network_proto,
                                   const string& caffe_model,
                                   const int gpu_id) {
#ifdef CPU_ONLY
  LOG(INFO) << "Using CPU mode in Caffe.";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  LOG(INFO) << "Using GPU mode in Caffe with device: " << gpu_id;
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

  // setup network
  this->net_.reset(new Net<Dtype>(network_proto, caffe::TEST));

  if (caffe_model != "NONE") {
    this->net_->CopyTrainedLayersFrom(caffe_model_);
  } else {
    LOG(INFO) << "Not initializing network from pre-trained model.";
  }

  CHECK_EQ(this->net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<Dtype>* input_layer = this->net_->input_blobs()[0];
  LOG(INFO) << "Network requires input size: (width, height) "
            << input_layer->width() << ", " << input_layer->height();

  num_channels_ = input_layer->channels();
  CHECK_EQ(num_channels_, 3) << "Input layer should have 3 channels.";

  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

template <typename Dtype>
void Regressor<Dtype>::Init() {
  if (modified_params_ ) {
    LOG(INFO) << "Reloading params.";
    this->net_->CopyTrainedLayersFrom(caffe_model_);
    modified_params_ = false;
  }
}

template <typename Dtype>
void Regressor<Dtype>::Regress(const cv::Mat& image, const cv::Mat& target,
                               BoundingBox<Dtype>* bbox) {
  assert(this->net_->phase() == caffe::TEST);
  std::vector<Dtype> estimation;
  Estimate(image, target, &estimation);
  *bbox = BoundingBox<Dtype>(estimation);
}

template <typename Dtype>
void Regressor<Dtype>::Regress(const std::vector<cv::Mat>& images,
                               const std::vector<cv::Mat>& targets,
                               std::vector<BoundingBox<Dtype> >* bboxes) {
  assert(this->net_->phase() == caffe::TEST);
  CHECK_EQ(images.size(),targets.size());
  std::vector<Dtype> estimation;
  Estimate(images, targets, &estimation);
  CHECK_EQ(4*images.size(), estimation.size());
  bboxes->clear();
  for (int i = 0; i < images.size(); ++i) {
    std::vector<Dtype> res;
    res.push_back(estimation[i*4]);
    res.push_back(estimation[i*4+1]);
    res.push_back(estimation[i*4+2]);
    res.push_back(estimation[i*4+3]);
    bboxes->push_back(BoundingBox<Dtype>(res));
  }
}

template <typename Dtype>
void Regressor<Dtype>::Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<Dtype>* output) {
  assert(this->net_->phase() == caffe::TEST);
  Blob<Dtype>* input_blob = this->net_->input_blobs()[0];
  input_blob->Reshape(2, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  this->net_->Reshape();

  Dtype* transformed_data = input_blob->mutable_cpu_data();

  // resize/mean处理后，送入网络
  cv::Mat image_resized, target_resized;
  cv::resize(image, image_resized, cv::Size(input_geometry_.width, input_geometry_.height));
  cv::resize(target, target_resized, cv::Size(input_geometry_.width, input_geometry_.height));
  CHECK_EQ(image_resized.channels(),3);
  CHECK_EQ(target_resized.channels(),3);
  const int offset = image_resized.rows * image_resized.cols;
  const int offset3 = 3 * offset;
  bool normalize = false;
  for (int i = 0; i < image_resized.rows; ++i) {
    for (int j = 0; j < image_resized.cols; ++j) {
      cv::Vec3b& rgb_prev = target_resized.at<cv::Vec3b>(i, j);
      cv::Vec3b& rgb_curr = image_resized.at<cv::Vec3b>(i, j);
      if (normalize) {
        // prev
        transformed_data[i * image_resized.cols + j] = (rgb_prev[0] - 128)/256.0;
        transformed_data[offset + i * image_resized.cols + j] = (rgb_prev[1] - 128)/256.0;
        transformed_data[2 * offset + i * image_resized.cols + j] = (rgb_prev[2] - 128)/256.0;
        // curr
        transformed_data[offset3 + i * image_resized.cols + j] = (rgb_curr[0] - 128)/256.0;
        transformed_data[offset3 + offset + i * image_resized.cols + j] = (rgb_curr[1] - 128)/256.0;
        transformed_data[offset3 + 2 * offset + i * image_resized.cols + j] = (rgb_curr[2] - 128)/256.0;
      } else {
        // prev
        transformed_data[i * image_resized.cols + j] = rgb_prev[0] - 104;
        transformed_data[offset + i * image_resized.cols + j] = rgb_prev[1] - 117;
        transformed_data[2 * offset + i * image_resized.cols + j] = rgb_prev[2] - 123;
        // curr
        transformed_data[offset3 + i * image_resized.cols + j] = rgb_curr[0] - 104;
        transformed_data[offset3 + offset + i * image_resized.cols + j] = rgb_curr[1] - 117;
        transformed_data[offset3 + 2 * offset + i * image_resized.cols + j] = rgb_curr[2] - 123;
      }
    }
  }
  // 使用前向计算，得到输出
  this->net_->Forward();
  // 获取结果
  GetOutput(output);
}

template <typename Dtype>
void Regressor<Dtype>::ReshapeImageInputs(const int num_images) {
  Blob<Dtype>* input_target = this->net_->input_blobs()[0];
  input_target->Reshape(2*num_images, num_channels_,
                       input_geometry_.height, input_geometry_.width);
}

template <typename Dtype>
void Regressor<Dtype>::GetFeatures(const string& feature_name, std::vector<Dtype>* output) const {
  const boost::shared_ptr<Blob<Dtype> > layer = this->net_->blob_by_name(feature_name.c_str());
  const Dtype* begin = layer->cpu_data();
  const Dtype* end = begin + layer->count();
  *output = std::vector<Dtype>(begin, end);
}

template <typename Dtype>
void Regressor<Dtype>::Estimate(const std::vector<cv::Mat>& images,
                                const std::vector<cv::Mat>& targets,
                                std::vector<Dtype>* output) {
  assert(this->net_->phase() == caffe::TEST);
  CHECK_EQ(images.size(), targets.size());
  ReshapeImageInputs(images.size());

  Blob<Dtype>* input_blob = this->net_->input_blobs()[0];
  Dtype* transformed_data = input_blob->mutable_cpu_data();

  // 导入数据
  for (int n = 0; n < images.size(); ++n) {
    const cv::Mat& prev = targets[n];
    const cv::Mat& curr = images[n];
    // resized
    cv::Mat prev_resized, curr_resized;
    cv::resize(prev, prev_resized, cv::Size(input_geometry_.width, input_geometry_.height));
    cv::resize(curr, curr_resized, cv::Size(input_geometry_.width, input_geometry_.height));
    CHECK_EQ(prev_resized.channels(),3);
    CHECK_EQ(curr_resized.channels(),3);
    const int offset = prev_resized.rows * prev_resized.cols;
    const int offset3 = 3 * offset;
    const int half_offs = images.size() * offset3;
    const bool normalize = false;
    for (int i = 0; i < prev_resized.rows; ++i) {
      for (int j = 0; j < prev_resized.cols; ++j) {
        cv::Vec3b& rgb_prev = prev_resized.at<cv::Vec3b>(i, j);
        cv::Vec3b& rgb_curr = curr_resized.at<cv::Vec3b>(i, j);
        if (normalize) {
          // prev
          transformed_data[n * offset3 + i * prev_resized.cols + j] = (rgb_prev[0] - 128)/256.0;
          transformed_data[n * offset3 + offset + i * prev_resized.cols + j] = (rgb_prev[1] - 128)/256.0;
          transformed_data[n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_prev[2] - 128)/256.0;
          // curr
          transformed_data[half_offs + n * offset3 + i * prev_resized.cols + j] = (rgb_curr[0] - 128)/256.0;
          transformed_data[half_offs + n * offset3 + offset + i * prev_resized.cols + j] = (rgb_curr[1] - 128)/256.0;
          transformed_data[half_offs + n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_curr[2] - 128)/256.0;
        } else {
          // prev
          transformed_data[n * offset3 + i * prev_resized.cols + j] = (rgb_prev[0] - 104);
          transformed_data[n * offset3 + offset + i * prev_resized.cols + j] = (rgb_prev[1] - 117);
          transformed_data[n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_prev[2] - 123);
          // curr
          transformed_data[half_offs + n * offset3 + i * prev_resized.cols + j] = (rgb_curr[0] - 104);
          transformed_data[half_offs + n * offset3 + offset + i * prev_resized.cols + j] = (rgb_curr[1] - 117);
          transformed_data[half_offs + n * offset3 + 2 * offset + i * prev_resized.cols + j] = (rgb_curr[2] - 123);
        }
      }
    }
  }
  this->net_->Reshape();
  this->net_->Forward();
  GetOutput(output);
}

template <typename Dtype>
void Regressor<Dtype>::GetOutput(std::vector<Dtype>* output) {
  GetFeatures("fc8", output);
}

INSTANTIATE_CLASS(Regressor);

}
