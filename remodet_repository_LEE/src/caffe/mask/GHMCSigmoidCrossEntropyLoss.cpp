#include <vector>
#include <math.h>
#include <cfloat>
#include "caffe/mask/GHMCSigmoidCrossEntropyLoss.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void GHMCSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const GhmcLossParameter& param = this->layer_param_.ghmc_loss_param();
  // bins
  m_ = param.m();
  //weight
  weight = param.weight();
  // mmt
  alpha_ = param.alpha();
  weight_type = param.weight_type();
  use_group = param.use_group();
  diff_thred = param.diff_thred();
  power = param.power();
  //k1 = param.k1();
  //k2 = param.k2();
  //b1 = param.b1();
  //b2 = param.b2();
  CHECK_GT(m_, 0) << "m must be larger than zero";
  CHECK_GE(alpha_, 0) << "alpha must be >= 0";
  CHECK_LT(alpha_, 1) << "alpha must be < 1";

  // running (samples number in one bins)
  r_num_ = new float[m_];
  memset(r_num_, 0, m_ * sizeof(float));
}

template <typename Dtype>
void GHMCSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // grident norm |p-p*|
  gradNorm_.ReshapeLike(*bottom[0]);
  // 最终elemtwise的权值
  beta_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GHMCSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* pred = sigmoid_top_vec_[0]->cpu_data();

  Dtype* gradNorm_data = gradNorm_.mutable_cpu_data();
  caffe_sub(count, pred, target, gradNorm_data);
  caffe_abs(count, gradNorm_data, gradNorm_data);

  Dtype* beta_data = beta_.mutable_cpu_data();
  caffe_set(count, Dtype(0), beta_data);
  //LOG(INFO)<<m_;
  float epsin = 1.0 / m_;
  //compute the r_num
  int *num_in_bin = new int[m_];
  memset(num_in_bin, 0, m_ * sizeof(int));

  for (int k = 0; k < count; k++) {
    //LOG(INFO)<<"raw_diff:"<<gradNorm_data[k];
    for (int i = 0; i < m_; i++) {
      float min_g = i * epsin;
      float max_g = (i + 1) * epsin;
      // Don't calculate ignore label
      if ( gradNorm_data[k] < max_g && gradNorm_data[k] >= min_g) {
        num_in_bin[i] += 1;
        //record the index of r_num
        beta_data[k] = i;
        break;
      }
    }
  }

  int valid = 0;
  for (int i = 0; i < m_; i++) {
    if (num_in_bin[i] > 0) {
      r_num_[i] = alpha_ * r_num_[i] + (1 - alpha_) * num_in_bin[i];
      valid++;
    }
  }

  delete[] num_in_bin;
  //compute beta and loss,   beta = N / GD(g)

  Dtype loss = 0;
  /*
  if (valid > 0) {
    for (int k = 0; k < count; ++k) {
      int id = beta_data[k];
      if(weight_type == "index"){
        //LOG(INFO)<<"index";
        beta_data[k] =pow((count * 1.0 / r_num_[id]),weight) / valid;}
      if(weight_type == "multiply"){
        //LOG(INFO)<<"multiply";
        beta_data[k] = weight*(count * 1.0 / r_num_[id]) / valid;}
      loss -= (input_data[k] * (target[k] - (input_data[k] >= 0)) -
               log(1 + exp(input_data[k] - 2 * input_data[k] * (input_data[k] >= 0)))) * beta_data[k];
    }
  }
  */
  if(use_group){
    if (valid > 0) {
      for (int k = 0; k < count; ++k) {
        int id = beta_data[k];
        LOG(INFO)<<power;
        //LOG(INFO)<<(count * 1.0 / r_num_[id]) / valid;
        if((r_num_[id]/(count * 1.0))<=diff_thred){
	  float k_ = pow(diff_thred,1.0-power);
          //LOG(INFO)<<r_num_[id] /(count * 1.0);
          //LOG(INFO)<<r_num_[id] /count * 1.0<<"-----";
          beta_data[k] =1/(k_*pow(r_num_[id] /(count * 1.0),power)+FLT_MIN) / valid;}
        else{  
          float k_ = pow(1.0-diff_thred,1.0-power);
          beta_data[k] =1.0/(1.0-k_* pow(1.0-r_num_[id] / (count * 1.0),power)+FLT_MIN) / valid;}
        loss -= (input_data[k] * (target[k] - (input_data[k] >= 0)) -
                 log(1 + exp(input_data[k] - 2 * input_data[k] * (input_data[k] >= 0)))) * beta_data[k];
      }
    }
  }

  else{
    //LOG(INFO)<<"1111";
    if (valid > 0) {
      for (int k = 0; k < count; ++k) {
        int id = beta_data[k];
        beta_data[k] = (count * 1.0 / r_num_[id]) / valid;
        LOG(INFO)<<"GHMC_DIFF:"<<beta_data[k] * gradNorm_data[k];
	LOG(INFO)<<"RAW_DIFF:"<<gradNorm_data[k];
        loss -= (input_data[k] * (target[k] - (input_data[k] >= 0)) -
                 log(1 + exp(input_data[k] - 2 * input_data[k] * (input_data[k] >= 0)))) * beta_data[k];
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void GHMCSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    const Dtype* beta_data = beta_.cpu_data();
    caffe_mul(count, beta_data, bottom_diff, bottom_diff);

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
// STUB_GPU(GHMCSigmoidCrossEntropyLossLayer);
STUB_GPU_BACKWARD(GHMCSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(GHMCSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(GHMCSigmoidCrossEntropyLoss);

}  // namespace caffe
