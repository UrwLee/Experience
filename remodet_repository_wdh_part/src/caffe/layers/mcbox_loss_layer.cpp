#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/layers/mcbox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void McBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 使用损失层的设置函数
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // bottom [0] -> loc
  // bottom [1] -> conf (1+classes)
  // bottom [2] -> gt
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
  // 获取multibox的损失参数
  const McBoxLossParameter& mcbox_loss_param =
      this->layer_param_.mcbox_loss_param();
  // 定义样本数
  num_ = bottom[0]->num();
  // 定义gt数
  num_gt_ = bottom[2]->height();
  num_classes_ = mcbox_loss_param.num_classes();
  CHECK_GE(num_classes_, 1);
  background_label_id_ = mcbox_loss_param.background_label_id();
  // overlaps for initial loc-error
  overlap_threshold_ = mcbox_loss_param.overlap_threshold();
  // use prior for matching
  use_prior_for_matching_ = mcbox_loss_param.use_prior_for_matching();
  // use prior for box initial
  use_prior_for_init_ = mcbox_loss_param.use_prior_for_init();
  // use diff for training
  use_difficult_gt_ = mcbox_loss_param.use_difficult_gt();
  // rescore
  rescore_ = mcbox_loss_param.rescore();
  //code_loc_type
  code_loc_type_ = mcbox_loss_param.code_loc_type();
  // iters
  iters_ = mcbox_loss_param.iters();
  // iters_thre
  iter_using_bgboxes_ = mcbox_loss_param.iter_using_bgboxes();
  // background_box_loc_scale_
  background_box_loc_scale_ = mcbox_loss_param.background_box_loc_scale();
  // loss param
  object_scale_ = mcbox_loss_param.object_scale();
  noobject_scale_ = mcbox_loss_param.noobject_scale();
  class_scale_ = mcbox_loss_param.class_scale();
  loc_scale_ = mcbox_loss_param.loc_scale();

  // clip the priors to 1.
  clip_ = mcbox_loss_param.clip();

  // prior-boxes
  prior_width_.clear();
  prior_height_.clear();
  // load unit box
  prior_width_.push_back(Dtype(0.95));
  prior_height_.push_back(Dtype(0.95));
  if (mcbox_loss_param.boxsize_size() > 0 && mcbox_loss_param.pwidth_size() > 0) {
    LOG(FATAL) << "boxsize and pwidth/height could not be provided at the same time.";
  } else if (mcbox_loss_param.boxsize_size() == 0 && mcbox_loss_param.pwidth_size() == 0) {
    LOG(FATAL) << "Must provide boxsize or pwidth.";
  }

  if (mcbox_loss_param.boxsize_size() > 0) {
    CHECK_GT(mcbox_loss_param.aspect_ratio_size(), 0);
    for (int i = 0; i < mcbox_loss_param.boxsize_size(); ++i) {
      CHECK_GT(mcbox_loss_param.boxsize(i), 0);
      CHECK_LT(mcbox_loss_param.boxsize(i), 1);
      Dtype base_size = mcbox_loss_param.boxsize(i);
      for (int j = 0; j < mcbox_loss_param.aspect_ratio_size(); ++j) {
        Dtype ratio = mcbox_loss_param.aspect_ratio(j);
        Dtype w = base_size * sqrt(ratio);
        if (clip_) w = std::min(w, Dtype(1));
        Dtype h = base_size / sqrt(ratio);
        if (clip_) h = std::min(h, Dtype(1));
        prior_width_.push_back(w);
        prior_height_.push_back(h);
      }
    }
    num_priors_ = prior_width_.size();
  } else {
    CHECK_EQ(mcbox_loss_param.pwidth_size(), mcbox_loss_param.pheight_size());
    for (int i = 0; i < mcbox_loss_param.pwidth_size(); ++i) {
      CHECK_GT(mcbox_loss_param.pwidth(i), 0);
      CHECK_LE(mcbox_loss_param.pwidth(i), 1);
      CHECK_GT(mcbox_loss_param.pheight(i), 0);
      CHECK_LE(mcbox_loss_param.pheight(i), 1);
      prior_width_.push_back(mcbox_loss_param.pwidth(i));
      prior_height_.push_back(mcbox_loss_param.pheight(i));
    }
    num_priors_ = prior_width_.size();
  }

  // loss type
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void McBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //add loss
  vector<int> top_shape(4,1);
  // top_shape[3] = 3;
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->num();
  num_gt_ = bottom[2]->height();
  // [n,h,w,num_priors_*4]
  // [n,h,w,num_priors_*(num_classes_+1)]
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->width(), num_priors_ * 4);
  CHECK_EQ(bottom[1]->width(), num_priors_ * (num_classes_ + 1));

  // diff_
  loc_diff_.ReshapeLike(*bottom[0]);
  conf_diff_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void McBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* gt_data = bottom[2]->cpu_data();
  num_ = bottom[0]->num();
  num_gt_ = bottom[2]->height();
  CHECK_EQ(num_, bottom[1]->num());
  /**************************************************************************#
  获取整个batch的GT-boxes
  #***************************************************************************/
  // int -> 样本号
  // vector<> -> 每个样本对应的gt-boxes列表
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);

  int layer_height = bottom[0]->channels();
  int layer_width = bottom[0]->height();
  CHECK_EQ(bottom[1]->height(), layer_width);
  CHECK_EQ(bottom[1]->channels(), layer_height);

  int outsize_loc = layer_width*layer_height*num_priors_*4;
  int outsize_conf = layer_width*layer_height*num_priors_*(1+num_classes_);
  /**************************************************************************#
  Step-0: Logistic and Softmax
  #***************************************************************************/
  Dtype* mutable_conf_data = bottom[1]->mutable_cpu_data();
  // 处理object分类 以及 softmax
  for (int item_id = 0; item_id < num_; ++item_id) {
    for (int i = 0; i < layer_height; ++i) {
      for (int j = 0; j < layer_width; ++j) {
        for (int n = 0; n < num_priors_; ++n) {
          int idx_conf = item_id * outsize_conf + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * (num_classes_ + 1);
          // object分类：logistic
          mutable_conf_data[idx_conf] = logistic(conf_data[idx_conf]);
          // conf分类：softmax
          Softmax(conf_data + idx_conf + 1, num_classes_, mutable_conf_data + idx_conf + 1);
        }
      }
    }
  }

  // 保存误差
  Dtype* loc_diff = loc_diff_.mutable_cpu_data();
  Dtype* conf_diff = conf_diff_.mutable_cpu_data();
  caffe_set(loc_diff_.count(), Dtype(0), loc_diff);
  caffe_set(conf_diff_.count(), Dtype(0), conf_diff);

  Dtype loss = 0;
  Dtype loc_loss = 0;
  Dtype conf_loss = 0;
  Dtype softmax_loss = 0.0;

  match_count_ = 0;
  int conf_count = 0;
  for (int item_id = 0; item_id < num_; ++item_id) {
    /**************************************************************************#
    Step-1: 所有boxes初始化
    #***************************************************************************/
    for (int i = 0; i < layer_height; ++i) {
      for (int j = 0; j < layer_width; ++j) {
        for (int n = 0; n < num_priors_; ++n) {
          int idx_loc = item_id * outsize_loc + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * 4;
          int idx_conf = item_id * outsize_conf + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * (num_classes_ + 1);
          NormalizedBBox pred_bbox, corner_pred_box;
          if (use_prior_for_init_) {
            // we use anchor for initial
            Dtype xcenter = (Dtype(j) + 0.5) / layer_width;
            Dtype ycenter = (Dtype(i) + 0.5) / layer_height;
            pred_bbox.set_xmin(xcenter);
            pred_bbox.set_ymin(ycenter);
            pred_bbox.set_xmax(prior_width_[n]);
            pred_bbox.set_ymax(prior_height_[n]);
          } else {
            pred_bbox = get_NormalizedBBoxbyLoc(
                        loc_data,prior_width_,prior_height_,
                        n,idx_loc,j,i,layer_width,layer_height,code_loc_type_);
          }
          CenterToCorner(pred_bbox, &corner_pred_box);
          // 遍历所有gt，查找最大iou
          Dtype best_iou = 0;
          // int best_label = -1;
          for (map<int, vector<NormalizedBBox> >::iterator it = all_gt_bboxes.begin();
              it != all_gt_bboxes.end(); ++it) {
            int image_id = it->first;
            if (item_id != image_id) continue;
            if (it->second.size() == 0) continue;
            for (int k = 0; k < it->second.size(); ++k) {
              const NormalizedBBox& gtbox = it->second[k];
              float iou = JaccardOverlap(corner_pred_box, gtbox);
              if (iou > best_iou){
                best_iou = iou;
              }
            }
          }
          // 如果最大iou低于限定值，认为是背景，按照背景返回误差
          if (best_iou < overlap_threshold_) {
            conf_diff[idx_conf] =
              noobject_scale_ * (conf_data[idx_conf] - 0) * logistic_gradient(conf_data[idx_conf]);
            conf_count ++;
          }
          // LOC回归误差： 早期，趋向于0，后期，不反悔误差
          // 在训练早期阶段，默认将所有box的估计值都逼近anchor，即LOC回归器输出全部为(0,0,0,0)
          // 在后期，该功能被禁止
          if (!use_prior_for_init_) {
            if (iters_ < iter_using_bgboxes_) {
              NormalizedBBox anchor;
              anchor.set_xmin((Dtype(j) + 0.5) / layer_width);
              anchor.set_ymin((Dtype(i) + 0.5) / layer_height);
              anchor.set_xmax(prior_width_[n]);
              anchor.set_ymax(prior_height_[n]);
              Backward_mcbox(anchor,loc_data,idx_loc,
                              prior_width_,prior_height_,n,j,i,
                              layer_width,layer_height,
                              background_box_loc_scale_,
                              loc_diff,code_loc_type_);
            }
          }
        }
      }
    }
    /**************************************************************************#
    Step-2： gt中心匹配
    #***************************************************************************/
    for (map<int, vector<NormalizedBBox> >::iterator it = all_gt_bboxes.begin();
        it != all_gt_bboxes.end(); ++it) {
      int image_id = it->first;
      if (image_id != item_id) continue;
      if (it->second.size() == 0) continue;
      for (int k = 0; k < it->second.size(); ++k) {
        // gtbox: corner
        const NormalizedBBox& gtbox = it->second[k];
        // convert to center-coding
        NormalizedBBox gt_center;
        CornerToCenter(gtbox,&gt_center);
        float best_iou = 0;
        int best_loc_idx = 0;
        int best_conf_idx = 0;
        int best_n = 0;
        // get the (i,j) for matching
        int j = (int)(gt_center.xmin() * layer_width);
        int i = (int)(gt_center.ymin() * layer_height);
        // shift the (i,j) point for matching
        NormalizedBBox gt_shift;
        gt_shift.CopyFrom(gt_center);
        gt_shift.set_xmin(Dtype(0));
        gt_shift.set_ymin(Dtype(0));
        for (int n = 0; n < num_priors_; ++n) {
          int idx_loc = item_id * outsize_loc + (i * layer_width * num_priors_
                        + j * num_priors_ + n) * 4;
          // get pred-boxes (center mode)
          NormalizedBBox pred_bbox = get_NormalizedBBoxbyLoc(
                      loc_data,prior_width_,prior_height_,
                      n,idx_loc,j,i,layer_width,layer_height,code_loc_type_);
          // use prior for matching: use prior-box instead
          if (use_prior_for_matching_) {
            pred_bbox.set_xmax(prior_width_[n]);
            pred_bbox.set_ymax(prior_height_[n]);
          }
          pred_bbox.set_xmin(Dtype(0));
          pred_bbox.set_ymin(Dtype(0));
          NormalizedBBox corner_pred, corner_gt;
          CenterToCorner(pred_bbox,&corner_pred);
          CenterToCorner(gt_shift,&corner_gt);
          float iou = JaccardOverlap(corner_pred,corner_gt);
          if (iou > best_iou) {
            best_iou = iou;
            best_n = n;
            best_loc_idx = idx_loc;
            best_conf_idx = item_id * outsize_conf + (i * layer_width * num_priors_
                          + j * num_priors_ + n) * (num_classes_ + 1);
          }
        }
        // 获得了匹配box
        // 首先计算实际估计的pred与gt的iou
        NormalizedBBox match_pred_bbox = get_NormalizedBBoxbyLoc(
                    loc_data,prior_width_,prior_height_,
                    best_n,best_loc_idx,j,i,layer_width,layer_height,code_loc_type_);
        NormalizedBBox corner_match_pred_bbox;
        CenterToCorner(match_pred_bbox,&corner_match_pred_bbox);
        // 计算IOU
        float iou = JaccardOverlap(corner_match_pred_bbox, gtbox);
        // 计算loc返回误差
        Backward_mcbox(gt_center,loc_data,best_loc_idx,
                       prior_width_,prior_height_,best_n,
                       j,i,layer_width,layer_height,
                       loc_scale_, loc_diff,code_loc_type_);
        // 计算object分类返回误差
        if (rescore_) {
          conf_diff[best_conf_idx] =
            object_scale_ * (conf_data[best_conf_idx] - iou) * logistic_gradient(conf_data[best_conf_idx]);
            conf_count ++;
          // LOG(INFO) << "conf:" << conf_diff[best_conf_idx];
        } else {
          conf_diff[best_conf_idx] =
            object_scale_ * (conf_data[best_conf_idx] - 1.) * logistic_gradient(conf_data[best_conf_idx]);
        }
        // 计算softmax分类返回误差
        int gt_label = gtbox.label();
        for (int c = 1; c < num_classes_ + 1; ++c) {
          conf_diff[best_conf_idx + c] =
            class_scale_ * (conf_data[best_conf_idx + c] - ((c == gt_label)? 1 : 0));
            softmax_loss = softmax_loss + conf_diff[best_conf_idx + c] * conf_diff[best_conf_idx + c];
        }
        // match_count ++
        match_count_ ++;
      }
    }
  }
  /**************************************************************************#
  Step-Final： Loss，整个batch上的L2损失
  #***************************************************************************/
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype loc_sumsq = loc_diff_.sumsq_data();
  Dtype conf_sumsq = conf_diff_.sumsq_data();
  // loss = loc_sumsq + conf_sumsq;
  // 平均损失
  loc_loss = ( match_count_ == 0 ) ? 0: loc_sumsq / match_count_ ;
  conf_loss = (conf_sumsq - softmax_loss) / (num_*layer_width*layer_height*num_priors_);
  softmax_loss = ( match_count_ == 0 ) ? 0: softmax_loss / match_count_ / num_classes_;
  // conf_loss = conf_sumsq / num_*layer_width*layer_height*num_priors_;
  loss = loc_loss + conf_loss + softmax_loss;
  top_data[0] = loss;
  // LOG(INFO) << "conf_count:" << conf_count;
  // LOG(INFO) << "conf len:" << conf_diff_.count();
  // LOG(INFO) << "match count" << match_count_;
  // LOG(INFO) << "priors:" << num_*layer_width*layer_height*num_priors_;
  // LOG(INFO) << "loc_loss" << loc_loss;
  // LOG(INFO) << "conf_loss" << conf_loss;
  // LOG(INFO) << "softmax_loss" << softmax_loss;
  // LOG(INFO) << "loss" << loss;
  // top_data[1] = loc_loss;
  // top_data[2] = conf_loss;
  iters_++;
}

template <typename Dtype>
void McBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to ground truth.";
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_,
          num_priors_*bottom[0]->shape(1)*bottom[0]->shape(2),
          match_count_);
      const Dtype alpha = top[0]->cpu_diff()[0] / normalizer;
      const Dtype* diff_data = NULL;
      if (i == 0) {
        diff_data = loc_diff_.cpu_data();
      } else {
        diff_data = conf_diff_.cpu_data();
      }
      caffe_cpu_axpby(
          bottom[i]->count(),
          alpha,
          diff_data,
          Dtype(0),
          bottom[i]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_CLASS(McBoxLossLayer);
REGISTER_LAYER_CLASS(McBoxLoss);

}  // namespace caffe
