#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/mask/prior_box_fixedsizetopflags_layer.hpp"

namespace caffe {

template <typename Dtype>
void PriorBoxFixedSizeTopFlagsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const PriorBoxParameter &prior_box_param =
      this->layer_param_.prior_box_param();
  if ((prior_box_param.min_size_size() > 0) && (prior_box_param.pro_width_size() > 0)) {
    LOG(FATAL) << "min_size and pro_width could not be provided at the same time.";
  } else if (prior_box_param.min_size_size() == 0 && prior_box_param.pro_width_size() == 0) {
    LOG(FATAL) << "Must provide min_size or pro_width.";
  }
    stride_ = prior_box_param.stride();//default=1
  // use size and aspect_ratio to define prior-boxes
  if (prior_box_param.min_size_size() > 0) {
    // min_sizes
    min_size_.clear();
    for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
      CHECK_GT(prior_box_param.min_size(i), 0) << "min_size must be positive.";
      min_size_.push_back(prior_box_param.min_size(i));
    }
    // flip the ar
    flip_ = prior_box_param.flip();


    // aspect_ratios
    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.);
    for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
      float ar = prior_box_param.aspect_ratio(i);
      bool already_exist = false;
      for (int j = 0; j < aspect_ratios_.size(); ++j) {
        if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
          already_exist = true;
          break;
        }
      }
      if (!already_exist) {
        aspect_ratios_.push_back(ar);
        if (flip_) {
          aspect_ratios_.push_back(1. / ar);
        }
      }
    }
    // max_scale
    num_priors_ = aspect_ratios_.size() * min_size_.size();
    if (prior_box_param.max_size_size() > 0) {
      CHECK_EQ(prior_box_param.max_size_size(), min_size_.size())
        << "max_sizes and min_sizes must have the same length.";
      max_size_.clear();
      for (int i = 0; i < prior_box_param.max_size_size(); ++i) {
        CHECK_GT(prior_box_param.max_size(i), min_size_[i]) << "max_size must be greater than min_size.";
        max_size_.push_back(prior_box_param.max_size(i));
      }
      num_priors_ += max_size_.size();
    }
  } else if (prior_box_param.pro_width_size() > 0) {
    CHECK_EQ(prior_box_param.pro_width_size(), prior_box_param.pro_height_size())
      << "pro_width and pro_height must have the same length.";
    pro_widths_.clear();
    pro_heights_.clear();
    for (int i = 0; i < prior_box_param.pro_width_size(); ++i) {
     // CHECK_GT(prior_box_param.pro_width(i),0) << "pro_width must be positive.";
     // CHECK_LE(prior_box_param.pro_width(i),1) << "pro_width must be less than 1.";
     // LOG(INFO)<<prior_box_param.pro_width(i)<<"************"<<prior_box_param.pro_width_size()<<prior_box_param.pro_height_size();
      pro_widths_.push_back(prior_box_param.pro_width(i));
    }
    for (int i = 0; i < prior_box_param.pro_height_size(); ++i) {
     // CHECK_GT(prior_box_param.pro_height(i),0) << "pro_height must be positive.";
     // CHECK_LE(prior_box_param.pro_height(i),1) << "pro_height must be less than 1.";
      pro_heights_.push_back(prior_box_param.pro_height(i));
    }
    num_priors_ = pro_widths_.size();
  } else {
    LOG(FATAL) << "Error: min_sizes / pro_widths are not provided.";
  }

  // output prior-boxes need to be clipped?
  clip_ = prior_box_param.clip();
  // get the boxes code variances
  if (prior_box_param.variance_size() > 1) {
    // Must and only provide 4 variance.
    CHECK_EQ(prior_box_param.variance_size(), 4);
    for (int i = 0; i < prior_box_param.variance_size(); ++i) {
      CHECK_GT(prior_box_param.variance(i), 0);
      variance_.push_back(prior_box_param.variance(i));
    }
  } else if (prior_box_param.variance_size() == 1) {
    CHECK_GT(prior_box_param.variance(0), 0);
    variance_.push_back(prior_box_param.variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }
}

template <typename Dtype>
void PriorBoxFixedSizeTopFlagsLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  // bottom[0] -> the feature map
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);

  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  top_shape[2] = ((layer_width-1)/stride_+1) * ((layer_height-1)/stride_ +1)* num_priors_ * 4;
  // LOG(INFO)<<layer_width<<"~~"<<layer_width/stride_<<"~~~";
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape);
  
  vector<int> top1_shape(2, 1);
  top1_shape[0] = 1;
  top1_shape[1] = ((layer_width-1)/stride_+1) * ((layer_height-1)/stride_ +1)* num_priors_;
  top[1]->Reshape(top1_shape);
   
}

template <typename Dtype>
void PriorBoxFixedSizeTopFlagsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  // get the feature map size
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  // get the input image size
  const int img_width = bottom[1]->width();
  const int img_height = bottom[1]->height();
  // get the step size = imgSize/featureSize
  const float step_x = static_cast<float>(img_width) / layer_width;
  const float step_y = static_cast<float>(img_height) / layer_height;
  // get the output data
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *top_data1 = top[1]->mutable_cpu_data();
  // channel dim
  int dim = ((layer_width-1)/stride_+1) * ((layer_height-1)/stride_ +1)* num_priors_ * 4;
  // LOG(INFO)<<dim<<"!!!!!!!!!!!!!";
  int idx = 0;
  int idx1 = 0;
  // now we cal. each locations [output locations]
  for (int h = 0; h < layer_height; h=h+stride_) {
    for (int w = 0; w < layer_width; w=w+stride_) {
      // LOG(INFO)<<"##########"<<h<<"$$"<<w<<"@@"<<layer_height<<"@@"<<layer_width;
      float center_x = (w + 0.5) * step_x;
      float center_y = (h + 0.5) * step_y;
      float box_width, box_height;
      // use min_size_ to define prior-boxes
      if (min_size_.size() > 0) {
        // 使用所有的scale提出boxes
        for (int i = 0; i < min_size_.size(); ++i) {
          // 1
          box_width = box_height = min_size_[i];
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
          // sqrt(min*max)
          if (max_size_.size() > 0) {
            box_width = box_height = sqrt(min_size_[i] * max_size_[i]);
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
          // aspect_ratios
          for (int r = 0; r < aspect_ratios_.size(); ++r) {
            float ar = aspect_ratios_[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size_[i] * sqrt(ar);
            box_height = min_size_[i] / sqrt(ar);
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
        }
      } else if (pro_widths_.size() > 0) {
          CHECK_EQ(pro_widths_.size(),pro_heights_.size());
          for (int i = 0; i < pro_widths_.size(); ++i) {
            box_width = pro_widths_[i];
            box_height = pro_heights_[i];
            top_data[idx++] = center_x / img_width  - box_width/ 2./img_width;
            top_data[idx++] = center_y / img_height - box_height/ 2./img_height;
            top_data[idx++] = center_x / img_width  + box_width/ 2/img_width;
            top_data[idx++] = center_y / img_height + box_height/ 2./img_height;
      			if(box_width<=img_width&&box_height<=img_height){
      				top_data1[idx1++] = 1;
      			}else{
      				top_data1[idx1++] = 0;
      			}
            // LOG(INFO)<<img_width<<"*"<<img_height;
          }
      } else {
        LOG(FATAL) << "Error: min_sizes / pro_widths are not provided.";
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }

  // output variances
  top_data += top[0]->offset(0, 1);
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; h=h+stride_) {
      for (int w = 0; w < layer_width; w=w+stride_) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PriorBoxFixedSizeTopFlagsLayer);
REGISTER_LAYER_CLASS(PriorBoxFixedSizeTopFlags);

} // namespace caffe
