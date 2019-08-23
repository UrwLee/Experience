#include "caffe/tracker/fregressor.hpp"

namespace caffe {

using caffe::Blob;
using caffe::Net;
using std::string;

template <typename Dtype>
FRegressor<Dtype>::FRegressor(const int gpu_id,
                              const std::string& tracker_network_proto,
                              const std::string& tracker_caffe_model,
                              const std::string& res_features,
                              const int num_inputs)
  : num_inputs_(num_inputs), features_(res_features) {
  SetupNetwork(tracker_network_proto, tracker_caffe_model, gpu_id);
}

template <typename Dtype>
FRegressor<Dtype>::FRegressor(const int gpu_id,
                            const std::string& tracker_network_proto,
                            const std::string& tracker_caffe_model,
                            const std::string& res_features)
  : num_inputs_(1), features_(res_features) {
  SetupNetwork(tracker_network_proto, tracker_caffe_model, gpu_id);
}

template <typename Dtype>
FRegressor<Dtype>::FRegressor(const int gpu_id,
                              const boost::shared_ptr<caffe::Net<Dtype> >& net,
                              const std::string& res_features)
  : num_inputs_(1), features_(res_features) {
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
  LOG(INFO) << "Network requires input size: (channels, height, width) "
            << input_layer->channels() << ", " << input_layer->height() << ", " << input_layer->width();
  input_channels_ = input_layer->channels();
  input_height_ = input_layer->height();
  input_width_ = input_layer->width();
}

template <typename Dtype>
void FRegressor<Dtype>::SetupNetwork(const string& network_proto,
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
    this->net_->CopyTrainedLayersFrom(caffe_model);
  } else {
    LOG(INFO) << "Not initializing network from pre-trained model.";
  }
  CHECK_EQ(this->net_->num_outputs(), 1) << "Network should have exactly one output.";
  Blob<Dtype>* input_layer = this->net_->input_blobs()[0];
  LOG(INFO) << "Network requires input size: (channels, height, width) "
            << input_layer->channels() << ", " << input_layer->height() << ", " << input_layer->width();
  input_channels_ = input_layer->channels();
  input_height_ = input_layer->height();
  input_width_ = input_layer->width();
}

template <typename Dtype>
void FRegressor<Dtype>::Regress(const Blob<Dtype>& curr, const Blob<Dtype>& prev,
                                BoundingBox<Dtype>* bbox) {
  assert(this->net_->phase() == caffe::TEST);
  std::vector<Dtype> estimation;
  Estimate(curr, prev, &estimation);
  *bbox = BoundingBox<Dtype>(estimation);
}

template <typename Dtype>
void FRegressor<Dtype>::Regress(const std::vector<boost::shared_ptr<Blob<Dtype> > >& curr,
                                const std::vector<boost::shared_ptr<Blob<Dtype> > >& prev,
                                std::vector<BoundingBox<Dtype> >* bboxes) {
  assert(this->net_->phase() == caffe::TEST);
  CHECK_EQ(curr.size(),prev.size());
  std::vector<Dtype> estimation;
  Estimate(curr, prev, &estimation);
  CHECK_EQ(4*curr.size(), estimation.size());
  bboxes->clear();
  for (int i = 0; i < curr.size(); ++i) {
    std::vector<Dtype> res;
    res.push_back(estimation[i*4]);
    res.push_back(estimation[i*4+1]);
    res.push_back(estimation[i*4+2]);
    res.push_back(estimation[i*4+3]);
    BoundingBox<Dtype> rbox(res);
    bboxes->push_back(rbox);
  }
}

template <typename Dtype>
void FRegressor<Dtype>::Estimate(const Blob<Dtype>& curr, const Blob<Dtype>& prev, std::vector<Dtype>* output) {
  assert(this->net_->phase() == caffe::TEST);
  Blob<Dtype>* input_blob = this->net_->input_blobs()[0];
  input_blob->Reshape(2, input_channels_, input_height_, input_width_);
  this->net_->Reshape();
  Dtype* transformed_data = input_blob->mutable_cpu_data();
  // CHECK
  CHECK_EQ(curr.width(), input_width_);
  CHECK_EQ(curr.height(), input_height_);
  CHECK_EQ(curr.channels(), input_channels_);
  CHECK_EQ(curr.num(), 1);
  CHECK_EQ(prev.width(), input_width_);
  CHECK_EQ(prev.height(), input_height_);
  CHECK_EQ(prev.channels(), input_channels_);
  CHECK_EQ(prev.num(), 1);
  // data transfer
  caffe_copy(prev.count(),prev.cpu_data(),transformed_data);
  caffe_copy(curr.count(),curr.cpu_data(),transformed_data+prev.count());
  this->net_->Forward();
  GetOutput(output);
}

template <typename Dtype>
void FRegressor<Dtype>::ReshapeNumInputs(const int num_imputs) {
  Blob<Dtype>* input_target = this->net_->input_blobs()[0];
  input_target->Reshape(2*num_imputs, input_channels_, input_height_, input_width_);
}

template <typename Dtype>
void FRegressor<Dtype>::GetFeatures(const string& feature_name, std::vector<Dtype>* output) {
  const boost::shared_ptr<Blob<Dtype> > layer = this->net_->blob_by_name(feature_name.c_str());
  const Dtype* begin = layer->cpu_data();
  const Dtype* end = begin + layer->count();
  *output = std::vector<Dtype>(begin, end);
}

template <typename Dtype>
void FRegressor<Dtype>::Estimate(const std::vector<boost::shared_ptr<Blob<Dtype> > >& curr,
                                 const std::vector<boost::shared_ptr<Blob<Dtype> > >& prev,
                                 std::vector<Dtype>* output) {
  assert(this->net_->phase() == caffe::TEST);
  CHECK_EQ(curr.size(), prev.size());
  ReshapeNumInputs(curr.size());

  Blob<Dtype>* input_blob = this->net_->input_blobs()[0];
  Dtype* transformed_data = input_blob->mutable_cpu_data();

  // 导入数据
  int half_offset = curr.size() * curr[0].get()->count();
  for (int i = 0; i < curr.size(); ++i) {
    Blob<Dtype>* curr_f = curr[i].get();
    Blob<Dtype>* prev_f = prev[i].get();
    CHECK_EQ(curr_f->width(), input_width_);
    CHECK_EQ(curr_f->height(), input_height_);
    CHECK_EQ(curr_f->channels(), input_channels_);
    CHECK_EQ(curr_f->num(), 1);
    CHECK_EQ(prev_f->width(), input_width_);
    CHECK_EQ(prev_f->height(), input_height_);
    CHECK_EQ(prev_f->channels(), input_channels_);
    CHECK_EQ(prev_f->num(), 1);
    caffe_copy(prev_f->count(),prev_f->cpu_data(),transformed_data+i*prev_f->count());
    caffe_copy(curr_f->count(),curr_f->cpu_data(),transformed_data+i*prev_f->count()+half_offset);
  }
  this->net_->Reshape();
  this->net_->Forward();
  GetOutput(output);
}

template <typename Dtype>
void FRegressor<Dtype>::GetOutput(std::vector<Dtype>* output) {
  GetFeatures(features_, output);
}

INSTANTIATE_CLASS(FRegressor);

}
