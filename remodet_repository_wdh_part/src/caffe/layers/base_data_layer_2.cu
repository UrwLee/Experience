#include <vector>

#include "caffe/layers/base_data_layer_2.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingData2Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch_Orig<Dtype>* batch_orig = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch_orig->data_);
  top[1]->ReshapeLike(batch_orig->orig_data_);
  // Copy the data
  caffe_copy(batch_orig->data_.count(), batch_orig->data_.gpu_data(),
             top[0]->mutable_gpu_data());
  caffe_copy(batch_orig->orig_data_.count(), batch_orig->orig_data_.gpu_data(),
             top[1]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch_orig->label_);
    // Copy the labels.
    caffe_copy(batch_orig->label_.count(), batch_orig->label_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch_orig);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingData2Layer);

}  // namespace caffe
