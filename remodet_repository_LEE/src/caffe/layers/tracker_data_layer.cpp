#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/tracker_data_layer.hpp"

namespace caffe {

template <typename Dtype>
TrackerDataLayer<Dtype>::~TrackerDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TrackerDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  tracker_data_loader_.reset(new TrackerDataLoader<Dtype>(loader_param_));
  const int batch_size = loader_param_.batch_size();
  const int height = loader_param_.resized_height();
  const int width = loader_param_.resized_width();
  // for data
  top[0]->Reshape(2*batch_size, 3, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(2*batch_size, 3, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  // labels
  top[1]->Reshape(batch_size, 4, 1, 1);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(batch_size, 4, 1, 1);
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();
}

template <typename Dtype>
void TrackerDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  const int batch_size = loader_param_.batch_size();
  const int height = loader_param_.resized_height();
  const int width = loader_param_.resized_width();
  batch->data_.Reshape(2*batch_size, 3, height, width);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  batch->label_.Reshape(batch_size, 4, 1, 1);
  Dtype* top_label = batch->label_.mutable_cpu_data();
  // 调用TrackerDataLoader的Load方法即可生成一个batch
  tracker_data_loader_->Load(top_data, top_label);
}

INSTANTIATE_CLASS(TrackerDataLayer);
REGISTER_LAYER_CLASS(TrackerData);
}  // namespace caffe
