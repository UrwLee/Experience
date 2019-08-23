#include <algorithm>
#include <csignal>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "boost/iterator/counting_iterator.hpp"
#include "caffe/mask/bbox_func.hpp"

namespace caffe {

template <typename Dtype>
bool BBoxAscend(const LabeledBBox<Dtype> &bbox1, const LabeledBBox<Dtype> &bbox2) {
  return bbox1.score < bbox2.score;
}

template bool BBoxAscend(const LabeledBBox<float> &bbox1, const LabeledBBox<float> &bbox2);
template bool BBoxAscend(const LabeledBBox<double> &bbox1, const LabeledBBox<double> &bbox2);

template <typename Dtype>
bool BBoxDescend(const LabeledBBox<Dtype> &bbox1, const LabeledBBox<Dtype> &bbox2) {
  return bbox1.score > bbox2.score;
}

template bool BBoxDescend(const LabeledBBox<float> &bbox1, const LabeledBBox<float> &bbox2);
template bool BBoxDescend(const LabeledBBox<double> &bbox1, const LabeledBBox<double> &bbox2);

template <typename Dtype>
bool VectorDescend(const std::vector<Dtype>& vs1, const std::vector<Dtype>& vs2) {
    return vs1[0] > vs2[0];
}
template bool VectorDescend(const std::vector<float>& vs1, const std::vector<float>& vs2);
template bool VectorDescend(const std::vector<double>& vs1, const std::vector<double>& vs2);

template <typename Dtype>
bool VectorAescend(const std::vector<Dtype>& vs1, const std::vector<Dtype>& vs2) {
    return vs1[0] < vs2[0];
}
template bool VectorAescend(const std::vector<float>& vs1, const std::vector<float>& vs2);
template bool VectorAescend(const std::vector<double>& vs1, const std::vector<double>& vs2);

template <typename T, typename Dtype>
bool PairAscend(const pair<Dtype, T> &pair1, const pair<Dtype, T> &pair2) {
  return pair1.first < pair2.first;
}
template bool PairAscend(const pair<float, int> &pair1, const pair<float, int> &pair2);
template bool PairAscend(const pair<float, pair<int, int> > &pair1, const pair<float, pair<int, int> > &pair2);
template bool PairAscend(const pair<double, int> &pair1, const pair<double, int> &pair2);
template bool PairAscend(const pair<double, pair<int, int> > &pair1, const pair<double, pair<int, int> > &pair2);

template <typename T, typename Dtype>
bool PairDescend(const pair<Dtype, T> &pair1, const pair<Dtype, T> &pair2) {
  return pair1.first > pair2.first;
}
template bool PairDescend(const pair<float, int> &pair1, const pair<float, int> &pair2);
template bool PairDescend(const pair<float, pair<int, int> > &pair1, const pair<float, pair<int, int> > &pair2);
template bool PairDescend(const pair<double, int> &pair1, const pair<double, int> &pair2);
template bool PairDescend(const pair<double, pair<int, int> > &pair1, const pair<double, pair<int, int> > &pair2);

template <typename Dtype>
void EncodeBBox_Corner(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* encode_bbox) {
  if (encode_variance_in_target) {
    encode_bbox->x1_ = bbox.x1_ - prior_bbox.x1_;
    encode_bbox->y1_ = bbox.y1_ - prior_bbox.y1_;
    encode_bbox->x2_ = bbox.x2_ - prior_bbox.x2_;
    encode_bbox->y2_ = bbox.y2_ - prior_bbox.y2_;
  } else {
    CHECK_EQ(prior_variance.size(), 4);
    for (int i = 0; i < prior_variance.size(); ++i) {
      CHECK_GT(prior_variance[i], 0);
    }
    encode_bbox->x1_ = (bbox.x1_ - prior_bbox.x1_) / prior_variance[0];
    encode_bbox->y1_ = (bbox.y1_ - prior_bbox.y1_) / prior_variance[1];
    encode_bbox->x2_ = (bbox.x2_ - prior_bbox.x2_) / prior_variance[2];
    encode_bbox->y2_ = (bbox.y2_ - prior_bbox.y2_) / prior_variance[3];
  }
}

template void EncodeBBox_Corner(const BoundingBox<float>& prior_bbox, const vector<float>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<float>& bbox,
                                BoundingBox<float>* encode_bbox);
template void EncodeBBox_Corner(const BoundingBox<double>& prior_bbox, const vector<double>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<double>& bbox,
                                BoundingBox<double>* encode_bbox);

template <typename Dtype>
void EncodeBBox_Center(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* encode_bbox) {
   Dtype prior_width = prior_bbox.get_width();
   CHECK_GT(prior_width, 0);
   Dtype prior_height = prior_bbox.get_height();
   CHECK_GT(prior_height, 0);
   Dtype prior_center_x = prior_bbox.get_center_x();
   Dtype prior_center_y = prior_bbox.get_center_y();
   Dtype bbox_width = bbox.get_width();
   CHECK_GT(bbox_width, 0);
   Dtype bbox_height = bbox.get_height();
   CHECK_GT(bbox_height, 0);
   Dtype bbox_center_x = bbox.get_center_x();
   Dtype bbox_center_y = bbox.get_center_y();
   if (encode_variance_in_target) {
     encode_bbox->x1_ = (bbox_center_x - prior_center_x) / prior_width;
     encode_bbox->y1_ = (bbox_center_y - prior_center_y) / prior_height;
     encode_bbox->x2_ = log(bbox_width / prior_width);
     encode_bbox->y2_ = log(bbox_height / prior_height);
   } else {
     encode_bbox->x1_ = (bbox_center_x - prior_center_x) / prior_width / prior_variance[0];
     encode_bbox->y1_ = (bbox_center_y - prior_center_y) / prior_height / prior_variance[1];
     encode_bbox->x2_ = log(bbox_width / prior_width) / prior_variance[2];
     encode_bbox->y2_ = log(bbox_height / prior_height) / prior_variance[3];
   }
}

template void EncodeBBox_Center(const BoundingBox<float>& prior_bbox, const vector<float>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<float>& bbox,
                                BoundingBox<float>* encode_bbox);
template void EncodeBBox_Center(const BoundingBox<double>& prior_bbox, const vector<double>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<double>& bbox,
                                BoundingBox<double>* encode_bbox);

template <typename Dtype>
void DecodeBBox_Corner(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* decode_bbox) {
  if (encode_variance_in_target) {
    decode_bbox->x1_ = prior_bbox.x1_ + bbox.x1_;
    decode_bbox->y1_ = prior_bbox.y1_ + bbox.y1_;
    decode_bbox->x2_ = prior_bbox.x2_ + bbox.x2_;
    decode_bbox->y2_ = prior_bbox.y2_ + bbox.y2_;
  } else {
    decode_bbox->x1_ = prior_bbox.x1_ + prior_variance[0] * bbox.x1_;
    decode_bbox->y1_ = prior_bbox.y1_ + prior_variance[1] * bbox.y1_;
    decode_bbox->x2_ = prior_bbox.x2_ + prior_variance[2] * bbox.x2_;
    decode_bbox->y2_ = prior_bbox.y2_ + prior_variance[3] * bbox.y2_;
  }
}

template void DecodeBBox_Corner(const BoundingBox<float>& prior_bbox, const vector<float>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<float>& bbox,
                                BoundingBox<float>* decode_bbox);
template void DecodeBBox_Corner(const BoundingBox<double>& prior_bbox, const vector<double>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<double>& bbox,
                                BoundingBox<double>* decode_bbox);

template <typename Dtype>
void DecodeBBox_Center(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* decode_bbox) {
   Dtype prior_width = prior_bbox.get_width();
   CHECK_GT(prior_width, 0);
   Dtype prior_height = prior_bbox.get_height();
   CHECK_GT(prior_height, 0);
   Dtype prior_center_x = prior_bbox.get_center_x();
   Dtype prior_center_y = prior_bbox.get_center_y();
   Dtype decode_bbox_center_x, decode_bbox_center_y;
   Dtype decode_bbox_width, decode_bbox_height;
   if (encode_variance_in_target) {
     decode_bbox_center_x = bbox.x1_ * prior_width + prior_center_x;
     decode_bbox_center_y = bbox.y1_ * prior_height + prior_center_y;
     decode_bbox_width = exp(bbox.x2_) * prior_width;
     decode_bbox_height = exp(bbox.y2_) * prior_height;
   } else {
     decode_bbox_center_x = bbox.x1_ * prior_width * prior_variance[0] + prior_center_x;
     decode_bbox_center_y = bbox.y1_ * prior_height * prior_variance[1] + prior_center_y;
     decode_bbox_width = exp(bbox.x2_ * prior_variance[2]) * prior_width;
     decode_bbox_height = exp(bbox.y2_ * prior_variance[3]) * prior_height;
   }
   decode_bbox->x1_ = decode_bbox_center_x - decode_bbox_width / 2.;
   decode_bbox->y1_ = decode_bbox_center_y - decode_bbox_height / 2.;
   decode_bbox->x2_ = decode_bbox_center_x + decode_bbox_width / 2.;
   decode_bbox->y2_ = decode_bbox_center_y + decode_bbox_height / 2.;
}

template void DecodeBBox_Center(const BoundingBox<float>& prior_bbox, const vector<float>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<float>& bbox,
                                BoundingBox<float>* decode_bbox);
template void DecodeBBox_Center(const BoundingBox<double>& prior_bbox, const vector<double>& prior_variance,
                                const bool encode_variance_in_target, const BoundingBox<double>& bbox,
                                BoundingBox<double>* decode_bbox);

template <typename Dtype>
LabeledBBox<Dtype> LabeledBBox_Copy(const LabeledBBox<Dtype>& bbox) {
  LabeledBBox<Dtype> bbox_bak;
  bbox_bak.bindex = bbox.bindex;
  bbox_bak.cid = bbox.cid;
  bbox_bak.pid = bbox.pid;
  bbox_bak.is_diff = bbox.is_diff;
  bbox_bak.iscrowd = bbox.iscrowd;
  bbox_bak.score = bbox.score;
  bbox_bak.bbox.x1_ = bbox.bbox.x1_;
  bbox_bak.bbox.y1_ = bbox.bbox.y1_;
  bbox_bak.bbox.x2_ = bbox.bbox.x2_;
  bbox_bak.bbox.y2_ = bbox.bbox.y2_;
  return bbox_bak;
}

template LabeledBBox<float> LabeledBBox_Copy(const LabeledBBox<float>& bbox);
template LabeledBBox<double> LabeledBBox_Copy(const LabeledBBox<double>& bbox);

template <typename Dtype>
void DecodeBBoxes_Corner(const vector<LabeledBBox<Dtype> >& prior_bboxes,
                         const vector<vector<Dtype> >& prior_variances,
                         const bool variance_encoded_in_target,
                         const vector<LabeledBBox<Dtype> >& bboxes,
                         vector<LabeledBBox<Dtype> >* decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    LabeledBBox<Dtype> decode_bbox = LabeledBBox_Copy(bboxes[i]);
    DecodeBBox_Corner(prior_bboxes[i].bbox, prior_variances[i],variance_encoded_in_target,
                      bboxes[i].bbox, &decode_bbox.bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

template void DecodeBBoxes_Corner(const vector<LabeledBBox<float> >& prior_bboxes,
                                  const vector<vector<float> >& prior_variances,
                                  const bool variance_encoded_in_target,
                                  const vector<LabeledBBox<float> >& bboxes,
                                  vector<LabeledBBox<float> >* decode_bboxes);
template void DecodeBBoxes_Corner(const vector<LabeledBBox<double> >& prior_bboxes,
                                  const vector<vector<double> >& prior_variances,
                                  const bool variance_encoded_in_target,
                                  const vector<LabeledBBox<double> >& bboxes,
                                  vector<LabeledBBox<double> >* decode_bboxes);

template <typename Dtype>
void DecodeBBoxes_Center(const vector<LabeledBBox<Dtype> >& prior_bboxes,
                         const vector<vector<Dtype> >& prior_variances,
                         const bool variance_encoded_in_target,
                         const vector<LabeledBBox<Dtype> >& bboxes,
                         vector<LabeledBBox<Dtype> >* decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    LabeledBBox<Dtype> decode_bbox = LabeledBBox_Copy(bboxes[i]);
    DecodeBBox_Center(prior_bboxes[i].bbox, prior_variances[i],variance_encoded_in_target,
                      bboxes[i].bbox, &decode_bbox.bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

template void DecodeBBoxes_Center(const vector<LabeledBBox<float> >& prior_bboxes,
                                  const vector<vector<float> >& prior_variances,
                                  const bool variance_encoded_in_target,
                                  const vector<LabeledBBox<float> >& bboxes,
                                  vector<LabeledBBox<float> >* decode_bboxes);
template void DecodeBBoxes_Center(const vector<LabeledBBox<double> >& prior_bboxes,
                                  const vector<vector<double> >& prior_variances,
                                  const bool variance_encoded_in_target,
                                  const vector<LabeledBBox<double> >& bboxes,
                                  vector<LabeledBBox<double> >* decode_bboxes);

template <typename Dtype>
void DecodeBBoxes(const vector<vector<LabeledBBox<Dtype> > >& all_loc_preds,
                     const vector<LabeledBBox<Dtype> >& prior_bboxes,
                     const vector<vector<Dtype> >& prior_variances,
                     const int num, const int code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<LabeledBBox<Dtype> > >* all_decode_bboxes) {
   CHECK_EQ(all_loc_preds.size(), num);
   all_decode_bboxes->clear();
   all_decode_bboxes->resize(num);
   for (int i = 0; i < num; ++i) { //遍历所有样本
     vector<LabeledBBox<Dtype> >& decode_bboxes = (*all_decode_bboxes)[i]; // all_decode_bboxes 形式[num][num_priors]
     const vector<LabeledBBox<Dtype> >& label_loc_preds = all_loc_preds[i];
     // center
     if (code_type == 0) {
       DecodeBBoxes_Center(prior_bboxes,prior_variances,variance_encoded_in_target,
                           label_loc_preds,&decode_bboxes);
     } else {
       DecodeBBoxes_Corner(prior_bboxes,prior_variances,variance_encoded_in_target,
                           label_loc_preds,&decode_bboxes);
     }
   }
}

template void DecodeBBoxes(const vector<vector<LabeledBBox<float> > >& all_loc_preds,
                           const vector<LabeledBBox<float> >& prior_bboxes,
                           const vector<vector<float> >& prior_variances,
                           const int num, const int code_type,
                           const bool variance_encoded_in_target,
                           vector<vector<LabeledBBox<float> > >* all_decode_bboxes);
template void DecodeBBoxes(const vector<vector<LabeledBBox<double> > >& all_loc_preds,
                          const vector<LabeledBBox<double> >& prior_bboxes,
                          const vector<vector<double> >& prior_variances,
                          const int num, const int code_type,
                          const bool variance_encoded_in_target,
                          vector<vector<LabeledBBox<double> > >* all_decode_bboxes);

template <typename Dtype>
void DecodeDenseBBoxes(const vector<vector<LabeledBBox<Dtype> > >& all_loc_preds,
                     const vector<LabeledBBox<Dtype> >& prior_bboxes,
                     const vector<vector<Dtype> >& prior_variances,
                     const int num, const int num_classes, const int code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<vector<LabeledBBox<Dtype> > > >* all_decode_bboxes) {
   CHECK_EQ(all_loc_preds.size(), num);
   all_decode_bboxes->clear();
   all_decode_bboxes->resize(num); // all_decode_bboxes 结构是 [num][num_class-1][num_priors]
   const int num_priors = all_loc_preds[0].size() / (num_classes - 1); // all_loc_preds 形式 [num][num_priors * (num_class-1)]
   for (int i = 0; i < num; ++i) {  // 遍历每个样本
     (*all_decode_bboxes)[i].resize(num_classes-1); 
     for (int j = 0; j < num_classes - 1; ++j) { // 遍历每一个类
       vector<LabeledBBox<Dtype> >& decode_bboxes = (*all_decode_bboxes)[i][j];
       if (code_type == 0) { // 使用 center 解码方式
         for (int p = 0; p < num_priors; ++p) { // 遍历
           const LabeledBBox<Dtype>& pbox = all_loc_preds[i][p * (num_classes - 1) + j]; // p * (num_classes - 1) = start_class 相当于设置起始位置
           LabeledBBox<Dtype> decode_bbox = LabeledBBox_Copy(pbox);
           DecodeBBox_Center(prior_bboxes[p].bbox, prior_variances[p],variance_encoded_in_target,
             pbox.bbox, &decode_bbox.bbox);
             decode_bboxes.push_back(decode_bbox);
           }
       } else {
         for (int p = 0; p < num_priors; ++p) {
           const LabeledBBox<Dtype>& pbox = all_loc_preds[i][p * (num_classes - 1) + j];
           LabeledBBox<Dtype> decode_bbox = LabeledBBox_Copy(pbox);
           DecodeBBox_Corner(prior_bboxes[p].bbox, prior_variances[p],variance_encoded_in_target,
             pbox.bbox, &decode_bbox.bbox);
             decode_bboxes.push_back(decode_bbox);
           }
       }
     }
  }
}

template void DecodeDenseBBoxes(const vector<vector<LabeledBBox<float> > >& all_loc_preds,
                     const vector<LabeledBBox<float> >& prior_bboxes,
                     const vector<vector<float> >& prior_variances,
                     const int num, const int num_classes, const int code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<vector<LabeledBBox<float> > > >* all_decode_bboxes);
 template void DecodeDenseBBoxes(const vector<vector<LabeledBBox<double> > >& all_loc_preds,
                      const vector<LabeledBBox<double> >& prior_bboxes,
                      const vector<vector<double> >& prior_variances,
                      const int num, const int num_classes, const int code_type,
                      const bool variance_encoded_in_target,
                      vector<vector<vector<LabeledBBox<double> > > >* all_decode_bboxes);
template <typename Dtype>
void MatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                        const vector<LabeledBBox<Dtype> > &pred_bboxes,
                        const int match_type,
                        const Dtype overlap_threshold,
                        const Dtype negative_threshold,
                        vector<Dtype> *match_overlaps,
                        map<int, int> *match_indices,
                        vector<int> *neg_indices,
                        bool flag_noperson,
                        bool flag_withotherpositive) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  neg_indices->clear();
  // init overlaps
  match_overlaps->resize(num_pred, 0.);
  // init status
  vector<int> match_status;
  match_status.resize(num_pred, 0);
  // gt pool
  int num_gt = gt_bboxes.size();
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  vector<int> gt_indices;
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // compute overlaps [MAX]
  map<int, map<int, Dtype> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_gt; ++j) {
        Dtype overlap = pred_bboxes[i].bbox.compute_iou(gt_bboxes[gt_indices[j]].bbox);
        if (overlap > 1e-6) {
          (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
          overlaps[i][j] = overlap;
        }
      }
  }
  // gt index pool
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  // ---------------------------------------------------------------------------
  // Start Matching, Step 1: find the max anchor for each gt
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // search for max overlap
    for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          continue;
        } 
        if (it->second[j] > max_overlap) {
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    // search ending
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // choose other POSITIVE SAMPLES
  if(flag_withotherpositive){
    switch (match_type) {
      case 1:
        break;
      case 0:
        for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int i = it->first;
          if (match_status[i]) {
            continue;
          }
          // 遍历所有gt-box,查找最大IOU
          int max_gt_idx = -1;
          float max_overlap = -1;
          for (int j = 0; j < num_gt; ++j) {
            if (it->second.find(j) == it->second.end()) {
              continue;
            }
            float overlap = it->second[j];
            if (overlap > max_overlap) {
              max_gt_idx = j;
              max_overlap = overlap;
            }
          }
          if (max_gt_idx != -1) {
            if (max_overlap > overlap_threshold) {
              // POS
              CHECK_EQ(match_status[i], 0);
              match_status[i] = 1;
              (*match_indices)[i] = gt_indices[max_gt_idx];
              (*match_overlaps)[i] = max_overlap;
            } else if (max_overlap < negative_threshold) {
              // NEG
              neg_indices->push_back(i);
              (*match_overlaps)[i] = max_overlap;
            } else;
          } else {
            // NEG
            neg_indices->push_back(i);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown matching type.";
        break;
    }
  }
  return;
}

template void MatchAnchorsAndGTs(const vector<LabeledBBox<float> > &gt_bboxes,
                                 const vector<LabeledBBox<float> > &pred_bboxes,
                                 const int match_type,
                                 const float overlap_threshold,
                                 const float negative_threshold,
                                 vector<float> *match_overlaps,
                                 map<int, int> *match_indices,
                                 vector<int> *neg_indices,
                                 bool flag_noperson,
                                 bool flag_withotherpositive);
template void MatchAnchorsAndGTs(const vector<LabeledBBox<double> > &gt_bboxes,
                                const vector<LabeledBBox<double> > &pred_bboxes,
                                const int match_type,
                                const double overlap_threshold,
                                const double negative_threshold,
                                vector<double> *match_overlaps,
                                map<int, int> *match_indices,
                                vector<int> *neg_indices,
                                bool flag_noperson,
                                bool flag_withotherpositive);
template <typename Dtype>
void MatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                        const vector<LabeledBBox<Dtype> > &pred_bboxes,
                        const int match_type,
                        const Dtype overlap_threshold,
                        const Dtype negative_threshold,
                        vector<Dtype> *match_overlaps,
                        map<int, int> *match_indices,
                        vector<int> *neg_indices,
                        vector<bool> flags_of_anchor,
                        bool flag_noperson,
                        bool flag_withotherpositive,
                        bool flag_matchallneg) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  neg_indices->clear();
  // init overlaps
  match_overlaps->resize(num_pred, 0.);
  // init status
  vector<int> match_status;
  match_status.resize(num_pred, 0);
  // gt pool
  int num_gt = gt_bboxes.size();
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  vector<int> gt_indices;
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // compute overlaps [MAX]
  map<int, map<int, Dtype> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      for (int j = 0; j < num_gt; ++j) {
        Dtype overlap = pred_bboxes[i].bbox.compute_iou(gt_bboxes[gt_indices[j]].bbox);

        if (overlap > 1e-6) {
          (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        }
        if(flag_matchallneg){
          overlaps[i][j] = overlap;
        }else{
          if (overlap > 1e-6) {
            overlaps[i][j] = overlap;
          }
        }                 
      }
    }
  }
  // gt index pool
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  // ---------------------------------------------------------------------------
  // Start Matching, Step 1: find the max anchor for each gt
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // search for max overlap
    for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          continue;
        }
        if (it->second[j] > max_overlap) {
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    // search ending
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // choose other POSITIVE SAMPLES
  if(flag_withotherpositive){
    switch (match_type) {
      case 1:
        break;
      case 0:
        for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int i = it->first;
          if (match_status[i]) {
            continue;
          }
          // 遍历所有gt-box,查找最大IOU
          int max_gt_idx = -1;
          float max_overlap = -1;
          for (int j = 0; j < num_gt; ++j) {
            if (it->second.find(j) == it->second.end()) {
              continue;
            }
            float overlap = it->second[j];
            if (overlap > max_overlap) {
              max_gt_idx = j;
              max_overlap = overlap;
            }
          }
          if (max_gt_idx != -1) {
            if (max_overlap > overlap_threshold) {
              // POS
              CHECK_EQ(match_status[i], 0);
              match_status[i] = 1;
              (*match_indices)[i] = gt_indices[max_gt_idx];
              (*match_overlaps)[i] = max_overlap;
            } else if (max_overlap < negative_threshold) {
              // NEG
              neg_indices->push_back(i);
              (*match_overlaps)[i] = max_overlap;
            } else;
          } else {
            // NEG
            neg_indices->push_back(i);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown matching type.";
        break;
    }
  }
  return;
}

template void MatchAnchorsAndGTs(const vector<LabeledBBox<float> > &gt_bboxes,
                                 const vector<LabeledBBox<float> > &pred_bboxes,
                                 const int match_type,
                                 const float overlap_threshold,
                                 const float negative_threshold,
                                 vector<float> *match_overlaps,
                                 map<int, int> *match_indices,
                                 vector<int> *neg_indices,
                                 vector<bool> flags_of_anchor,
                                 bool flag_noperson,
                                 bool flag_withotherpositive,
                                 bool flag_matchallneg);
template void MatchAnchorsAndGTs(const vector<LabeledBBox<double> > &gt_bboxes,
                                const vector<LabeledBBox<double> > &pred_bboxes,
                                const int match_type,
                                const double overlap_threshold,
                                const double negative_threshold,
                                vector<double> *match_overlaps,
                                map<int, int> *match_indices,
                                vector<int> *neg_indices,
                                vector<bool> flags_of_anchor,
                                bool flag_noperson,
                                bool flag_withotherpositive,
                                bool flag_matchallneg);

template <typename Dtype>
void MatchAnchorsAndGTsWeightIou(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                        const vector<LabeledBBox<Dtype> > &pred_bboxes,
                        const int match_type,
                        const Dtype overlap_threshold,
                        const Dtype negative_threshold,
                        vector<Dtype> *match_overlaps,
                        map<int, int> *match_indices,
                        vector<int> *neg_indices,
                        vector<bool> flags_of_anchor,
                        bool flag_noperson,
                        bool flag_withotherpositive,
                        const float sigma) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  neg_indices->clear();
  // init overlaps
  match_overlaps->resize(num_pred, 0.);
  // init status
  vector<int> match_status;
  match_status.resize(num_pred, 0);
  // gt pool
  int num_gt = gt_bboxes.size();
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  vector<int> gt_indices;
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // compute overlaps [MAX]
  map<int, map<int, Dtype> > overlaps;
  map<int, map<int, Dtype> > expdists;
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      for (int j = 0; j < num_gt; ++j) {
        vector<Dtype> overlap = pred_bboxes[i].bbox.compute_iou_expdist(gt_bboxes[gt_indices[j]].bbox,sigma);
        if (overlap[0] > 1e-6) {
          (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap[0]);
          overlaps[i][j] = overlap[0];
          expdists[i][j] = overlap[1];
        }
      }
    }
  }
  // gt index pool
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  map<int, int> match_indices_tmp;
  // ---------------------------------------------------------------------------
  // Start Matching, Step 1: find the max anchor for each gt
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // search for max overlap
    for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          continue;
        } 
        Dtype overlap = it->second[j];
        if (overlap > max_overlap) {
          max_idx = i;
          max_gt_idx = j;
          max_overlap = overlap;
        }
      }
    }
    // search ending
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      match_indices_tmp[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // choose other POSITIVE SAMPLES
  if(flag_withotherpositive){
    switch (match_type) {
      case 1:
        break;
      case 0:
        for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int i = it->first;
          if (match_status[i]) {
            continue;
          }
          // 遍历所有gt-box,查找最大IOU
          int max_gt_idx = -1;
          float max_overlap = -1;
          for (int j = 0; j < num_gt; ++j) {
            if (it->second.find(j) == it->second.end()) {
              continue;
            }
            Dtype overlap = it->second[j];
            if (overlap > max_overlap) {
              max_gt_idx = j;
              max_overlap = overlap;
            }

          }
          if (max_gt_idx != -1) {
            if (max_overlap > overlap_threshold) {
              // POS
              CHECK_EQ(match_status[i], 0);
              match_status[i] = 1;
              match_indices_tmp[i] = gt_indices[max_gt_idx];
              (*match_overlaps)[i] = max_overlap;
            } else if (max_overlap < negative_threshold) {
              // NEG
              neg_indices->push_back(i);
              (*match_overlaps)[i] = max_overlap;
            } else;
          } else {
            // NEG
            neg_indices->push_back(i);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown matching type.";
        break;
    }
  }
  for(typename map<int,int>::iterator it = match_indices_tmp.begin(); it!=match_indices_tmp.end(); ++it){
    int ianchor = it->first;
    int igt = it->second;
    Dtype overlap = overlaps[ianchor][igt];
    Dtype expd = expdists[ianchor][igt];
    Dtype v = overlap*expd;
    if(v>overlap_threshold){
      (*match_indices)[ianchor] = igt;
    }
  }
  return;
}
template void MatchAnchorsAndGTsWeightIou(const vector<LabeledBBox<float> > &gt_bboxes,
                                 const vector<LabeledBBox<float> > &pred_bboxes,
                                 const int match_type,
                                 const float overlap_threshold,
                                 const float negative_threshold,
                                 vector<float> *match_overlaps,
                                 map<int, int> *match_indices,
                                 vector<int> *neg_indices,
                                 vector<bool> flags_of_anchor,
                                 bool flag_noperson,
                                 bool flag_withotherpositive,
                                 const float sigma);
template void MatchAnchorsAndGTsWeightIou(const vector<LabeledBBox<double> > &gt_bboxes,
                                const vector<LabeledBBox<double> > &pred_bboxes,
                                const int match_type,
                                const double overlap_threshold,
                                const double negative_threshold,
                                vector<double> *match_overlaps,
                                map<int, int> *match_indices,
                                vector<int> *neg_indices,
                                vector<bool> flags_of_anchor,
                                bool flag_noperson,
                                bool flag_withotherpositive,
                                const float sigma);

template <typename Dtype>
void MatchAnchorsAndGTsCheckExtraCoverage(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                        const vector<LabeledBBox<Dtype> > &pred_bboxes,
                        const int match_type,
                        const Dtype overlap_threshold,
                        const Dtype negative_threshold,
                        vector<Dtype> *match_overlaps,
                        map<int, int> *match_indices,
                        vector<int> *neg_indices,
                        vector<bool> flags_of_anchor,
                        bool flag_noperson,
                        bool flag_withotherpositive,
                        const float cover_extracheck=0.7) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  neg_indices->clear();
  // init overlaps
  match_overlaps->resize(num_pred, 0.);
  // init status
  vector<int> match_status;
  match_status.resize(num_pred, 0);
  // gt pool
  int num_gt = gt_bboxes.size();
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  vector<int> gt_indices;
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // compute overlaps [MAX]
  map<int, map<int, Dtype> > overlaps;
  map<int, map<int, Dtype> > coverage;
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      for (int j = 0; j < num_gt; ++j) {
        vector<Dtype> overlap = gt_bboxes[gt_indices[j]].bbox.compute_iou_coverage(pred_bboxes[i].bbox);
        coverage[i][j] = overlap[1];
        // vector<Dtype> overlap = pred_bboxes[i].bbox.compute_iou_expdist(gt_bboxes[gt_indices[j]].bbox,sigma);
        if (overlap[0] > 1e-6) {
          (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap[0]);
          overlaps[i][j] = overlap[0];    
        }
      }
    }
  }
  // gt index pool
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  map<int, int> match_indices_tmp;
  // ---------------------------------------------------------------------------
  // Start Matching, Step 1: find the max anchor for each gt
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // search for max overlap
    for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          continue;
        } 
        Dtype overlap = it->second[j];
        if (overlap > max_overlap) {
          max_idx = i;
          max_gt_idx = j;
          max_overlap = overlap;
        }
      }
    }
    // search ending
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      match_indices_tmp[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // choose other POSITIVE SAMPLES
  if(flag_withotherpositive){
    switch (match_type) {
      case 1:
        break;
      case 0:
        for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int i = it->first;
          if (match_status[i]) {
            continue;
          }
          // 遍历所有gt-box,查找最大IOU
          int max_gt_idx = -1;
  
          float max_overlap = -1;
          for (int j = 0; j < num_gt; ++j) {
            if (it->second.find(j) == it->second.end()) {
              continue;
            }
            Dtype overlap = it->second[j];
            if (overlap > max_overlap) {
              max_gt_idx = j;
              max_overlap = overlap;
            }
          }
          if (max_gt_idx != -1) {
            if (max_overlap > overlap_threshold) {
              // POS
              CHECK_EQ(match_status[i], 0);
              match_status[i] = 1;
              match_indices_tmp[i] = gt_indices[max_gt_idx];
              (*match_overlaps)[i] = max_overlap;
            } else if (max_overlap < negative_threshold) {
              // NEG
              neg_indices->push_back(i);
              (*match_overlaps)[i] = max_overlap;
            } else;
          } else {
            // NEG
            neg_indices->push_back(i);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown matching type.";
        break;
    }
  }

  for(typename map<int,int>::iterator it = match_indices_tmp.begin(); it!=match_indices_tmp.end(); ++it){
    int ianchor = it->first;
    int igt = it->second;
    bool flag_pos = true;
    for(int j=0;j<num_gt;j++){
      if(j!=igt){
        // LOG(INFO)<<ianchor<<","<<j;
        if(coverage[ianchor][j]>cover_extracheck)  flag_pos = false;      
      }
    }
    if(flag_pos){
      (*match_indices)[ianchor] = igt;
    }
    // else{
    //   LOG(INFO)<<ianchor<<","<<igt<<" removed!";
    // }
  }
  // LOG(INFO)<<match_indices_tmp.size()<<" "<<(*match_indices).size();
  return;
}
template void MatchAnchorsAndGTsCheckExtraCoverage(const vector<LabeledBBox<float> > &gt_bboxes,
                                 const vector<LabeledBBox<float> > &pred_bboxes,
                                 const int match_type,
                                 const float overlap_threshold,
                                 const float negative_threshold,
                                 vector<float> *match_overlaps,
                                 map<int, int> *match_indices,
                                 vector<int> *neg_indices,
                                 vector<bool> flags_of_anchor,
                                 bool flag_noperson,
                                 bool flag_withotherpositive,
                                 const float cover_extracheck);
template void MatchAnchorsAndGTsCheckExtraCoverage(const vector<LabeledBBox<double> > &gt_bboxes,
                                const vector<LabeledBBox<double> > &pred_bboxes,
                                const int match_type,
                                const double overlap_threshold,
                                const double negative_threshold,
                                vector<double> *match_overlaps,
                                map<int, int> *match_indices,
                                vector<int> *neg_indices,
                                vector<bool> flags_of_anchor,
                                bool flag_noperson,
                                bool flag_withotherpositive,
                                const float cover_extracheck);

template <typename Dtype>
void MatchAnchorsAndGTsRemoveLargeMargin(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                        const vector<LabeledBBox<Dtype> > &pred_bboxes,
                        const int match_type,
                        const Dtype overlap_threshold,
                        const Dtype negative_threshold,
                        vector<Dtype> *match_overlaps,
                        map<int, int> *match_indices,
                        vector<int> *neg_indices,
                        vector<bool> flags_of_anchor,
                        bool flag_noperson,
                        bool flag_withotherpositive,
                        const float margin_ratio) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  neg_indices->clear();
  // init overlaps
  match_overlaps->resize(num_pred, 0.);
  // init status
  vector<int> match_status;
  match_status.resize(num_pred, 0);
  // gt pool
  int num_gt = gt_bboxes.size();
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  vector<int> gt_indices;
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // compute overlaps [MAX]
  map<int, map<int, Dtype> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      for (int j = 0; j < num_gt; ++j) {
        Dtype overlap = pred_bboxes[i].bbox.compute_iou(gt_bboxes[gt_indices[j]].bbox);
        if (overlap > 1e-6) {
          (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
          overlaps[i][j] = overlap;    
        }
      }
    }
  }
  // gt index pool
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  map<int, int> match_indices_tmp;
  // ---------------------------------------------------------------------------
  // Start Matching, Step 1: find the max anchor for each gt
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // search for max overlap
    for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          continue;
        } 
        Dtype overlap = it->second[j];
        if (overlap > max_overlap) {
          max_idx = i;
          max_gt_idx = j;
          max_overlap = overlap;
        }
      }
    }
    // search ending
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      match_indices_tmp[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // choose other POSITIVE SAMPLES
  if(flag_withotherpositive){
    switch (match_type) {
      case 1:
        break;
      case 0:
        for (typename map<int, map<int, Dtype> >::iterator it = overlaps.begin();
             it != overlaps.end(); ++it) {
          int i = it->first;
          if (match_status[i]) {
            continue;
          }
          // 遍历所有gt-box,查找最大IOU
          int max_gt_idx = -1;
  
          float max_overlap = -1;
          for (int j = 0; j < num_gt; ++j) {
            if (it->second.find(j) == it->second.end()) {
              continue;
            }
            Dtype overlap = it->second[j];
            if (overlap > max_overlap) {
              max_gt_idx = j;
              max_overlap = overlap;
            }
          }
          if (max_gt_idx != -1) {
            if (max_overlap > overlap_threshold) {
              // POS
              CHECK_EQ(match_status[i], 0);
              match_status[i] = 1;
              match_indices_tmp[i] = gt_indices[max_gt_idx];
              (*match_overlaps)[i] = max_overlap;
            } else if (max_overlap < negative_threshold) {
              // NEG
              neg_indices->push_back(i);
              (*match_overlaps)[i] = max_overlap;
            } else;
          } else {
            // NEG
            neg_indices->push_back(i);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown matching type.";
        break;
    }
  }

  for(typename map<int,int>::iterator it = match_indices_tmp.begin(); it!=match_indices_tmp.end(); ++it){
    int ianchor = it->first;
    int igt = it->second;
    BoundingBox<Dtype> gt_b = gt_bboxes[igt].bbox;
    BoundingBox<Dtype> anchor_b = pred_bboxes[ianchor].bbox;
    float gt_w = (float)(gt_b.x2_ - gt_b.x1_);
    float gt_h = (float)(gt_b.y2_ - gt_b.y1_);
    float maring_w = gt_w * margin_ratio;
    float maring_h = gt_h * margin_ratio;
    float margin_left = (float)(gt_b.x1_ - anchor_b.x1_);
    float margin_right = (float)(anchor_b.x2_ - gt_b.x2_);
    float margin_top = (float)(gt_b.y1_ - anchor_b.y1_);
    float margin_bottom = (float)(anchor_b.y2_ - gt_b.y2_);
    if (margin_left<maring_w && margin_right<maring_w && margin_bottom<maring_h && margin_top<maring_h){
      (*match_indices)[ianchor] = igt;
    }  
  }
  // LOG(INFO)<<match_indices_tmp.size()<<" "<<(*match_indices).size();
  return;
}
template void MatchAnchorsAndGTsRemoveLargeMargin(const vector<LabeledBBox<float> > &gt_bboxes,
                                 const vector<LabeledBBox<float> > &pred_bboxes,
                                 const int match_type,
                                 const float overlap_threshold,
                                 const float negative_threshold,
                                 vector<float> *match_overlaps,
                                 map<int, int> *match_indices,
                                 vector<int> *neg_indices,
                                 vector<bool> flags_of_anchor,
                                 bool flag_noperson,
                                 bool flag_withotherpositive,
                                 const float margin_ratio);
template void MatchAnchorsAndGTsRemoveLargeMargin(const vector<LabeledBBox<double> > &gt_bboxes,
                                const vector<LabeledBBox<double> > &pred_bboxes,
                                const int match_type,
                                const double overlap_threshold,
                                const double negative_threshold,
                                vector<double> *match_overlaps,
                                map<int, int> *match_indices,
                                vector<int> *neg_indices,
                                vector<bool> flags_of_anchor,
                                bool flag_noperson,
                                bool flag_withotherpositive,
                                const float margin_ratio);

template <typename Dtype>
void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                              const vector<LabeledBBox<Dtype> > &pred_bboxes,
                              const Dtype overlap_threshold,
                              const Dtype negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  neg_indices->clear();
  int num_gt = gt_bboxes.size();
  vector<bool> flag_gtmatch(num_gt,false);
  vector<bool> flag_neg_indices(num_pred,false);
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  // search
  for (int i = 0; i < num_pred; ++i) {
      const LabeledBBox<Dtype>& pbox = pred_bboxes[i];
      Dtype max_iou = 0;
      // scan all GT-boxes
      for (int j = 0; j < num_gt; ++j) {
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[j];
        Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
        if (iou > overlap_threshold) {
          match_indices->push_back(std::make_pair(i,j));
          flag_gtmatch[j] = true;
        }
        if (iou > max_iou) {
          max_iou = iou;
        }
      }
      if (max_iou < negative_threshold) {
        flag_neg_indices[i] = true;
      }
  }

  if(flag_forcematchallgt){
    for(int i=0;i<num_gt;i++){
      if(!flag_gtmatch[i]){
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[i];
        float area = gtbox.bbox.compute_area();
        if(area<area_check_max){
          Dtype max_iou = 0;
          int id_match = -1;
          for(int j =0;j<num_pred;j++){
              const LabeledBBox<Dtype>& pbox = pred_bboxes[j];
              Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
              if(iou>max_iou){
                id_match = j;
                max_iou = iou;
              }
          }
          // LOG(INFO)<<max_iou;
          match_indices->push_back(std::make_pair(id_match,i));
          flag_neg_indices[id_match] = false;
        }
      }
    }
  }
  for(int i=0;i<num_pred;i++){
    if(flag_neg_indices[i]){
      neg_indices->push_back(i);
    }
  }
}

template void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<float> > &gt_bboxes,
                              const vector<LabeledBBox<float> > &pred_bboxes,
                              const float overlap_threshold,
                              const float negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max);
template void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<double> > &gt_bboxes,
                            const vector<LabeledBBox<double> > &pred_bboxes,
                            const double overlap_threshold,
                            const double negative_threshold,
                            vector<pair<int, int> >*match_indices,
                            vector<int> *neg_indices,
                            bool flag_noperson,
                            bool flag_forcematchallgt,
                            float area_check_max);

template <typename Dtype>
void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                              const vector<LabeledBBox<Dtype> > &pred_bboxes,
                              const Dtype overlap_threshold,
                              const Dtype negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              vector<bool> flags_of_anchor,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  neg_indices->clear();
  int num_gt = gt_bboxes.size();
  vector<bool> flag_gtmatch(num_gt,false);
  vector<bool> flag_neg_indices(num_pred,false);
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  // search
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      const LabeledBBox<Dtype>& pbox = pred_bboxes[i];
      Dtype max_iou = 0;
      // scan all GT-boxes
      for (int j = 0; j < num_gt; ++j) {
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[j];
        Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
        if (iou > overlap_threshold) {
          match_indices->push_back(std::make_pair(i,j));
          flag_gtmatch[j] = true;
        }
        if (iou > max_iou) {
          max_iou = iou;
        }
      }
      if (max_iou < negative_threshold) {
        flag_neg_indices[i] = true;
      }
    }
  }

  if(flag_forcematchallgt){
    for(int i=0;i<num_gt;i++){
      if(!flag_gtmatch[i]){
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[i];
        float area = gtbox.bbox.compute_area();
        if(area<area_check_max){
          Dtype max_iou = 0;
          int id_match = -1;
          for(int j =0;j<num_pred;j++){
            if(flags_of_anchor[j]){
              const LabeledBBox<Dtype>& pbox = pred_bboxes[j];
              Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
              if(iou>max_iou){
                id_match = j;
                max_iou = iou;
              }
            }
          }
          // LOG(INFO)<<max_iou;
          match_indices->push_back(std::make_pair(id_match,i));
          flag_neg_indices[id_match] = false;
        }
      }
    }
  }
  for(int i=0;i<num_pred;i++){
    if(flag_neg_indices[i]){
      neg_indices->push_back(i);
    }
  }
}

template void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<float> > &gt_bboxes,
                              const vector<LabeledBBox<float> > &pred_bboxes,
                              const float overlap_threshold,
                              const float negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              vector<bool> flags_of_anchor,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max);
template void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<double> > &gt_bboxes,
                            const vector<LabeledBBox<double> > &pred_bboxes,
                            const double overlap_threshold,
                            const double negative_threshold,
                            vector<pair<int, int> >*match_indices,
                            vector<int> *neg_indices,
                            vector<bool> flags_of_anchor,
                            bool flag_noperson,
                            bool flag_forcematchallgt,
                            float area_check_max);


template <typename Dtype>
void ExhaustMatchAnchorsAndGTsIgnoreGt(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                              const vector<LabeledBBox<Dtype> > &pred_bboxes,
                              const Dtype overlap_threshold,
                              const Dtype negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              vector<bool> flags_of_anchor,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  neg_indices->clear();
  int num_gt = gt_bboxes.size();
  vector<bool> flag_gtmatch(num_gt,false);
  vector<bool> flag_neg_indices(num_pred,false);
  if (flag_noperson){
    if(num_gt == 0){
       for (int i = 0; i < num_pred; ++i) {
          neg_indices->push_back(i);
       }
       return;
    }
  } else {
    if (num_gt == 0) {
      return;
    }
  } 
  // search
  for (int i = 0; i < num_pred; ++i) {
    if(flags_of_anchor[i]){
      const LabeledBBox<Dtype>& pbox = pred_bboxes[i];
      Dtype max_iou = 0;
      // scan all GT-boxes
      LabeledBBox<Dtype> gtbox_tmp;  // 选取最大的iou时, 将产生该iou的 gtbox传入 tmp中, 检查是否是 ignore gt.
      for (int j = 0; j < num_gt; ++j) {
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[j];
        Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
        if (iou > overlap_threshold) {
          if(gtbox.ignore_gt == false){            
            match_indices->push_back(std::make_pair(i,j));  // 如果不是被忽略的gt, 匹配到的anchor 存入match队列
          } 

            flag_gtmatch[j] = true;
          }
        
        if (iou > max_iou){
          max_iou = iou;
          gtbox_tmp = gt_bboxes[j];
        }
      }
      if (max_iou < negative_threshold) {
        if(gtbox_tmp.ignore_gt == false){
          flag_neg_indices[i] = true;
        }
      }// find neg sample 
    }
  }

  if(flag_forcematchallgt){
    for(int i=0;i<num_gt;i++){
      if(!flag_gtmatch[i]){
        const LabeledBBox<Dtype>& gtbox = gt_bboxes[i];
        float area = gtbox.bbox.compute_area();
        if(area<area_check_max){
          Dtype max_iou = 0;
          int id_match = -1;
          for(int j =0;j<num_pred;j++){
            if(flags_of_anchor[j]){
              const LabeledBBox<Dtype>& pbox = pred_bboxes[j];
              Dtype iou = pbox.bbox.compute_iou(gtbox.bbox);
              if(iou>max_iou){
                id_match = j;
                max_iou = iou;
              }
            }
          }
          // LOG(INFO)<<max_iou;
          match_indices->push_back(std::make_pair(id_match,i));
          flag_neg_indices[id_match] = false;
        }
      }
    }
  }
  for(int i=0;i<num_pred;i++){
    if(flag_neg_indices[i]){
      neg_indices->push_back(i);
    }
  }
}

template void ExhaustMatchAnchorsAndGTsIgnoreGt(const vector<LabeledBBox<float> > &gt_bboxes,
                              const vector<LabeledBBox<float> > &pred_bboxes,
                              const float overlap_threshold,
                              const float negative_threshold,
                              vector<pair<int, int> >*match_indices,
                              vector<int> *neg_indices,
                              vector<bool> flags_of_anchor,
                              bool flag_noperson,
                              bool flag_forcematchallgt,
                              float area_check_max);
template void ExhaustMatchAnchorsAndGTsIgnoreGt(const vector<LabeledBBox<double> > &gt_bboxes,
                            const vector<LabeledBBox<double> > &pred_bboxes,
                            const double overlap_threshold,
                            const double negative_threshold,
                            vector<pair<int, int> >*match_indices,
                            vector<int> *neg_indices,
                            vector<bool> flags_of_anchor,
                            bool flag_noperson,
                            bool flag_forcematchallgt,
                            float area_check_max);


template <typename Dtype>
void RemoveCrowds(const vector<LabeledBBox<Dtype> >& gt_bboxes,
                  const map<int, int>& match_indices,
                  map<int, int>* new_indices) {
  // filter anchors where its matching-gt is crowd
  for (map<int, int>::const_iterator it = match_indices.begin(); it != match_indices.end(); ++it) {
    int index = it->first;
    int gt_index = it->second;
    // check gt
    CHECK_LT(gt_index, gt_bboxes.size());
    const LabeledBBox<Dtype>& gt_bbox = gt_bboxes[gt_index];
    if (gt_bbox.iscrowd == 0) {
      // push
      (*new_indices)[index] = gt_index;
    }
  }
}

template void RemoveCrowds(const vector<LabeledBBox<float> >& gt_bboxes,
                           const map<int, int>& match_indices,
                           map<int, int>* new_indices);
template void RemoveCrowds(const vector<LabeledBBox<double> >& gt_bboxes,
                           const map<int, int>& match_indices,
                           map<int, int>* new_indices);

template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const Dtype size_threshold, map<int, vector<LabeledBBox<Dtype> > >* all_gt_bboxes,int ndim_label) {
  all_gt_bboxes->clear();
  bool keep_all_gts = true;
  if (gt_labels.size() > 0) keep_all_gts = false;
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * ndim_label;
    int bindex = gt_data[start_idx];
    if (bindex < 0) continue;
    int cid = gt_data[start_idx + 1];
    if (!keep_all_gts) {
      bool found = false;
      for (int l = 0; l < gt_labels.size(); ++l) {
        if (cid == gt_labels[l]) {
          found = true;
          break;
        }
      }
      if (!found) {
        continue;
      }
    }
    int pid = gt_data[start_idx + 2];
    int is_diff = gt_data[start_idx + 3];
    int iscrowd = gt_data[start_idx + 4];
    Dtype xmin = gt_data[start_idx + 5];
    Dtype ymin = gt_data[start_idx + 6];
    Dtype xmax = gt_data[start_idx + 7];
    Dtype ymax = gt_data[start_idx + 8];
    int ignore_gt = 0;
    if (ndim_label == int(10)){
      ignore_gt = gt_data[start_idx + 9];
    }
    if (!use_difficult_gt && is_diff) {
      continue;
    }
    if (iscrowd) {
      continue;
    }
    // get this gt_box
    LabeledBBox<Dtype> gt_bbox;
    gt_bbox.bindex = bindex;
    gt_bbox.cid = cid;
    gt_bbox.pid = pid;
    gt_bbox.is_diff = is_diff > 0 ? true : false;
    gt_bbox.iscrowd = iscrowd > 0 ? true : false;
    gt_bbox.score = 1;
    gt_bbox.ignore_gt = ignore_gt > 0 ? true : false;
    gt_bbox.bbox.x1_ = xmin;
    gt_bbox.bbox.y1_ = ymin;
    gt_bbox.bbox.x2_ = xmax;
    gt_bbox.bbox.y2_ = ymax;
    if (gt_bbox.bbox.compute_area() < size_threshold) {
      continue;
    }
    (*all_gt_bboxes)[bindex].push_back(gt_bbox);
  }
}

template void GetGTBBoxes(const float* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const float size_threshold, map<int, vector<LabeledBBox<float> > >* all_gt_bboxes,int ndim_label);
template void GetGTBBoxes(const double* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const double size_threshold, map<int, vector<LabeledBBox<double> > >* all_gt_bboxes,int ndim_label);

template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const vector<Dtype> size_threshold, map<int, vector<LabeledBBox<Dtype> > >* all_gt_bboxes,int ndim_label) {
  all_gt_bboxes->clear();
  bool keep_all_gts = true;
  if (gt_labels.size() > 0) keep_all_gts = false;
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * ndim_label;
    int bindex = gt_data[start_idx];
    if (bindex < 0) continue;
    int cid = gt_data[start_idx + 1];
    if (!keep_all_gts) {
      bool found = false;
      for (int l = 0; l < gt_labels.size(); ++l) {
        if (cid == gt_labels[l]) {
          found = true;
          break;
        }
      }
      if (!found) {
        continue;
      }
    }
    int pid = gt_data[start_idx + 2];
    int is_diff = gt_data[start_idx + 3];
    int iscrowd = gt_data[start_idx + 4];
    Dtype xmin = gt_data[start_idx + 5];
    Dtype ymin = gt_data[start_idx + 6];
    Dtype xmax = gt_data[start_idx + 7];
    Dtype ymax = gt_data[start_idx + 8];
    int ignore_gt = 0;
    if (ndim_label == int(10)){
      ignore_gt = gt_data[start_idx + 9];
      LOG(INFO) << " ignore_gt  " << ignore_gt;
    }
    if (!use_difficult_gt && is_diff) {
      continue;
    }
    if (iscrowd) {
      continue;
    }
    // get this gt_box
    LabeledBBox<Dtype> gt_bbox;
    gt_bbox.bindex = bindex;
    gt_bbox.cid = cid;
    gt_bbox.pid = pid;
    gt_bbox.is_diff = is_diff > 0 ? true : false;
    gt_bbox.iscrowd = iscrowd > 0 ? true : false;
    gt_bbox.score = 1;
    gt_bbox.ignore_gt = ignore_gt > 0 ? true : false;
    gt_bbox.bbox.x1_ = xmin;
    gt_bbox.bbox.y1_ = ymin;
    gt_bbox.bbox.x2_ = xmax;
    gt_bbox.bbox.y2_ = ymax;
    float gt_area = gt_bbox.bbox.compute_area();
    if ( gt_area< size_threshold[0] || gt_area > size_threshold[1]) {
      continue;
    }
    (*all_gt_bboxes)[bindex].push_back(gt_bbox);
  }
}

template void GetGTBBoxes(const float* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const vector<float> size_threshold, map<int, vector<LabeledBBox<float> > >* all_gt_bboxes,int ndim_label);
template void GetGTBBoxes(const double* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const vector<double> size_threshold, map<int, vector<LabeledBBox<double> > >* all_gt_bboxes,int ndim_label);

template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 map<int, map<int, vector<LabeledBBox<Dtype> > > >* all_gt_bboxes,int ndim_label) {
  all_gt_bboxes->clear();
  bool keep_all_gts = true;
  if (gt_labels.size() > 0) keep_all_gts = false;
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * ndim_label;
    int bindex = gt_data[start_idx];
    if (bindex < 0) continue;
    int cid = gt_data[start_idx + 1];
    if (!keep_all_gts) {
      bool found = false;
      for (int l = 0; l < gt_labels.size(); ++l) {
        if (cid == gt_labels[l]) {
          found = true;
          break;
        }
      }
      if (!found) {
        continue;
      }
    }
    int pid = gt_data[start_idx + 2];
    int is_diff = gt_data[start_idx + 3];
    int iscrowd = gt_data[start_idx + 4];
    Dtype xmin = gt_data[start_idx + 5];
    Dtype ymin = gt_data[start_idx + 6];
    Dtype xmax = gt_data[start_idx + 7];
    Dtype ymax = gt_data[start_idx + 8];
    if (!use_difficult_gt && is_diff) {
      continue;
    }
    if (iscrowd) {
      continue;
    }
    // get this gt_box
    LabeledBBox<Dtype> gt_bbox;
    gt_bbox.bindex = bindex;
    gt_bbox.cid = cid;
    gt_bbox.pid = pid;
    gt_bbox.is_diff = is_diff > 0 ? true : false;
    gt_bbox.iscrowd = iscrowd > 0 ? true : false;
    gt_bbox.score = 1;
    gt_bbox.bbox.x1_ = xmin;
    gt_bbox.bbox.y1_ = ymin;
    gt_bbox.bbox.x2_ = xmax;
    gt_bbox.bbox.y2_ = ymax;
    (*all_gt_bboxes)[bindex][cid].push_back(gt_bbox);
  }
}

template void GetGTBBoxes(const float* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 map<int, map<int, vector<LabeledBBox<float> > > >* all_gt_bboxes,int ndim_label);
template void GetGTBBoxes(const double* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                map<int, map<int, vector<LabeledBBox<double> > > >* all_gt_bboxes,int ndim_label);

template <typename Dtype>
void GetLocPreds(const Dtype *loc_data, const int num, const int num_priors,
                 vector<vector<LabeledBBox<Dtype> > > *loc_preds) {
  loc_preds->clear();
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<LabeledBBox<Dtype> >& item_bboxes = (*loc_preds)[i];
    for (int p = 0; p < num_priors; ++p) {
      int start_idx = p * 4;
      LabeledBBox<Dtype> bbox;
      bbox.bbox.x1_ = loc_data[start_idx];
      bbox.bbox.y1_ = loc_data[start_idx + 1];
      bbox.bbox.x2_ = loc_data[start_idx + 2];
      bbox.bbox.y2_ = loc_data[start_idx + 3];
      item_bboxes.push_back(bbox);
    }
    loc_data += num_priors * 4;
  }
}

template void GetLocPreds(const float *loc_data, const int num, const int num_priors,
                          vector<vector<LabeledBBox<float> > > *loc_preds);
template void GetLocPreds(const double *loc_data, const int num, const int num_priors,
                          vector<vector<LabeledBBox<double> > > *loc_preds);

template <typename Dtype>
void GetMaxScores(const Dtype *conf_data, const int num,
                  const int num_priors, const int num_classes,
                  vector<vector<pair<int, Dtype> > > *all_max_conf) {
  all_max_conf->clear();
  all_max_conf->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<pair<int, Dtype> > &max_conf = (*all_max_conf)[i];
    for (int p = 0; p < num_priors; ++p) {
      int max_id = 0;
      float max_val = -1.0;
      for (int j = 0; j < num_classes; ++j) {
        if (conf_data[p * num_classes + j] > max_val) {
          max_id = j;
          max_val = conf_data[p * num_classes + j];
        }
      }
      max_conf.push_back(std::make_pair(max_id, max_val));
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetMaxScores(const float *conf_data, const int num,
                           const int num_priors, const int num_classes,
                           vector<vector<pair<int, float> > > *all_max_conf);
template void GetMaxScores(const double *conf_data, const int num,
                           const int num_priors, const int num_classes,
                           vector<vector<pair<int, double> > > *all_max_conf);

template <typename Dtype>
void GetMaxScores(const Dtype *conf_data, const int num,
                  const int num_priors, const int num_classes, const bool class_major,
                  vector<vector<pair<int, Dtype> > > *all_max_conf) {
  all_max_conf->clear();
  all_max_conf->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<pair<int, Dtype> > &max_conf = (*all_max_conf)[i];
    for (int p = 0; p < num_priors; ++p) {
      int max_id = 0;
      Dtype max_val = -1.0;
      for (int j = 0; j < num_classes; ++j) {
        if (class_major) {
          if (conf_data[j * num_priors + p] > max_val) {
            max_id = j;
            max_val = conf_data[j * num_priors + p];
          }
        } else {
          if (conf_data[p * num_classes + j] > max_val) {
            max_id = j;
            max_val = conf_data[p * num_classes + j];
          }
        }
      }
      max_conf.push_back(std::make_pair(max_id, max_val));
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetMaxScores(const float *conf_data, const int num,
                           const int num_priors, const int num_classes,
                           const bool class_major,
                           vector<vector<pair<int, float> > > *all_max_conf);
template void GetMaxScores(const double *conf_data, const int num,
                           const int num_priors, const int num_classes,
                           const bool class_major,
                           vector<vector<pair<int, double> > > *all_max_conf);

template <typename Dtype>
void GetConfScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         vector<vector<vector<Dtype> > > *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    (*conf_preds)[i].resize(num_classes);
    for (int j = 0; j < num_classes; ++j) {
      vector<Dtype> &label_scores = (*conf_preds)[i][j];
      for (int p = 0; p < num_priors; ++p) {
       label_scores.push_back(conf_data[p * num_classes + j]);
      }
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetConfScores(const float *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         vector<vector<vector<float> > > *conf_preds);
template void GetConfScores(const double *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         vector<vector<vector<double> > > *conf_preds);

template <typename Dtype>
void GetConfScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major,
                         vector<vector<vector<Dtype> > > *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    (*conf_preds)[i].resize(num_classes);
    for (int j = 0; j < num_classes; ++j) {
      vector<Dtype> &label_scores = (*conf_preds)[i][j];
      if (class_major) {
        label_scores.assign(conf_data + num_priors * j,
                            conf_data + num_priors * (j + 1));
      } else {
        for (int p = 0; p < num_priors; ++p) {
          label_scores.push_back(conf_data[p * num_classes + j]);
        }
      }
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetConfScores(const float *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major,
                         vector<vector<vector<float> > > *conf_preds);
template void GetConfScores(const double *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major,
                         vector<vector<vector<double> > > *conf_preds);

template <typename Dtype>
void GetHDMScores(const Dtype *conf_data, const int num, const int num_priors,
                  const int num_classes, const int loss_type,
                  vector<vector<Dtype> > *all_max_scores) {
  all_max_scores->clear();
  for (int i = 0; i < num; ++i) {
    vector<Dtype> max_scores;
    for (int p = 0; p < num_priors; ++p) {
      int start_idx = p * num_classes;
      Dtype maxval = -FLT_MAX;
      Dtype maxval_pos = -FLT_MAX;
      for (int c = 0; c < num_classes; ++c) {
        maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        if (c > 0) {
          maxval_pos = std::max<Dtype>(conf_data[start_idx + c], maxval_pos);
        }
      }
      if (loss_type == 0) {
        Dtype sum = 0;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        maxval_pos = std::exp(maxval_pos - maxval) / sum;
      } else if (loss_type == 1) {
        maxval_pos = 1. / (1. + exp(-maxval_pos));
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      max_scores.push_back(maxval_pos);
    }
    all_max_scores->push_back(max_scores);
    conf_data += num_priors * num_classes;
  }
}

template void GetHDMScores(const float *conf_data, const int num, const int num_priors,
                           const int num_classes, const int loss_type,
                           vector<vector<float> > *all_max_scores);
template void GetHDMScores(const double *conf_data, const int num, const int num_priors,
                          const int num_classes, const int loss_type,
                          vector<vector<double> > *all_max_scores);

template <typename Dtype>
void GetAnchorBBoxes(const Dtype *prior_data, const int num_priors,
                    vector<LabeledBBox<Dtype> > *prior_bboxes,
                    vector<vector<Dtype> > *prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    LabeledBBox<Dtype> bbox;
    bbox.bbox.x1_ = prior_data[start_idx];
    bbox.bbox.y1_ = prior_data[start_idx + 1];
    bbox.bbox.x2_ = prior_data[start_idx + 2];
    bbox.bbox.y2_ = prior_data[start_idx + 3];
    prior_bboxes->push_back(bbox);
    int start_var_idx = (num_priors + i) * 4;
    vector<Dtype> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_var_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

template void GetAnchorBBoxes(const float *prior_data, const int num_priors,
                              vector<LabeledBBox<float> > *prior_bboxes,
                              vector<vector<float> > *prior_variances);
template void GetAnchorBBoxes(const double *prior_data, const int num_priors,
                              vector<LabeledBBox<double> > *prior_bboxes,
                              vector<vector<double> > *prior_variances);

template <typename Dtype>
void GetDetections(const Dtype *det_data, const int num_det,
    map<int, map<int, vector<LabeledBBox<Dtype> > > > *all_detections) {
  all_detections->clear();
  if(num_det <= 0) return;
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int bindex = det_data[start_idx];
    if (bindex < 0) continue;
    int cid = det_data[start_idx + 1];
    LabeledBBox<Dtype> bbox;
    bbox.bindex = bindex;
    bbox.cid = cid;
    bbox.score    = det_data[start_idx + 2];
    bbox.bbox.x1_ = det_data[start_idx + 3];
    bbox.bbox.y1_ = det_data[start_idx + 4];
    bbox.bbox.x2_ = det_data[start_idx + 5];
    bbox.bbox.y2_ = det_data[start_idx + 6];
    bbox.bbox.clip();
    (*all_detections)[bindex][cid].push_back(bbox);
  }
}

template void GetDetections(const float *det_data, const int num_det,
                            map<int, map<int, vector<LabeledBBox<float> > > > *all_detections);
template void GetDetections(const double *det_data, const int num_det,
                            map<int, map<int, vector<LabeledBBox<double> > > > *all_detections);

template <typename Dtype>
void GetTopKs(const vector<Dtype> &scores, const vector<int> &indices,
              const int top_k, vector<pair<Dtype, int> > *score_index_vec) {
  CHECK_EQ(scores.size(), indices.size());
  for (int i = 0; i < scores.size(); ++i) {
    score_index_vec->push_back(std::make_pair(scores[i], indices[i]));
  }
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   PairDescend<int, Dtype>);
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template void GetTopKs(const vector<float> &scores, const vector<int> &indices,
              const int top_k, vector<pair<float, int> > *score_index_vec);
template void GetTopKs(const vector<double> &scores, const vector<int> &indices,
              const int top_k, vector<pair<double, int> > *score_index_vec);

template <typename Dtype>
void NmsByBoxes(const vector<LabeledBBox<Dtype> > &bboxes, const vector<Dtype> &scores,
              const Dtype threshold, const int top_k, const bool reuse_overlaps,
              map<int, map<int, Dtype> > *overlaps, vector<int> *indices) {
  CHECK_EQ(bboxes.size(), scores.size()) << "bboxes and scores have different size.";
  vector<int> idx(boost::counting_iterator<int>(0),
                  boost::counting_iterator<int>(scores.size()));
  vector<pair<Dtype, int> > score_index_vec;
  GetTopKs(scores, idx, top_k, &score_index_vec);
  indices->clear();
  while (score_index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = score_index_vec.front().second;
    const LabeledBBox<Dtype> &best_bbox = bboxes[best_idx];
    if (best_bbox.bbox.compute_area() < 1e-5) {
      score_index_vec.erase(score_index_vec.begin());
      continue;
    }
    indices->push_back(best_idx);
    // Erase the best box.
    score_index_vec.erase(score_index_vec.begin());

    if (top_k > -1 && indices->size() >= top_k) {
      // Stop if finding enough bboxes for nms.
      break;
    }

    // Compute overlap between best_bbox and other remaining bboxes.
    // Remove a bbox if the overlap with best_bbox is larger than nms_threshold.
    for (typename vector<pair<Dtype, int> >::iterator it = score_index_vec.begin();
         it != score_index_vec.end();) {
      int cur_idx = it->second;
      const LabeledBBox<Dtype> &cur_bbox = bboxes[cur_idx];
      if (cur_bbox.bbox.compute_area() < 1e-5) {
        // Erase small box.
        it = score_index_vec.erase(it);
        continue;
      }
      Dtype cur_overlap = 0.;
      // reuse overlaps
      if (reuse_overlaps) {
        if (overlaps->find(best_idx) != overlaps->end() &&
            overlaps->find(best_idx)->second.find(cur_idx) !=
                (*overlaps)[best_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[best_idx][cur_idx];
        } else if (overlaps->find(cur_idx) != overlaps->end() &&
                   overlaps->find(cur_idx)->second.find(best_idx) !=
                       (*overlaps)[cur_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[cur_idx][best_idx];
        } else {
          cur_overlap = cur_bbox.bbox.compute_iou(best_bbox.bbox);
          // Store the overlap for future use.
          (*overlaps)[best_idx][cur_idx] = cur_overlap;
        }
      } else {
        cur_overlap = cur_bbox.bbox.compute_iou(best_bbox.bbox);
      }

      // Remove it if necessary
      if (cur_overlap > threshold) {
        it = score_index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

template void NmsByBoxes(const vector<LabeledBBox<float> > &bboxes, const vector<float> &scores,
                         const float threshold, const int top_k, const bool reuse_overlaps,
                         map<int, map<int, float> > *overlaps, vector<int> *indices);
template void NmsByBoxes(const vector<LabeledBBox<double> > &bboxes, const vector<double> &scores,
                        const double threshold, const int top_k, const bool reuse_overlaps,
                        map<int, map<int, double> > *overlaps, vector<int> *indices);

void NmsByOverlappedGrid(const bool *overlapped, const int num, vector<int> *indices) {
  vector<int> index_vec(boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(num));
  indices->clear();
  while (index_vec.size() != 0) {
    int best_idx = index_vec.front();
    indices->push_back(best_idx);
    index_vec.erase(index_vec.begin());
    for (vector<int>::iterator it = index_vec.begin(); it != index_vec.end();) {
      int cur_idx = *it;
      if (overlapped[best_idx * num + cur_idx]) {
        it = index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

template <typename Dtype>
void GetTopKsWithThreshold(const vector<Dtype> &scores, const Dtype threshold,
                           const int top_k, vector<pair<Dtype, int> > *score_index_vec) {
  for (int i = 0; i < scores.size(); ++i) {
    // if (scores[i] > 0.1){
    //   LOG(INFO)<<"hzw score:"<<scores[i];
    // }
   // LOG(INFO)<<"score:"<<scores[i];
   if (scores[i] > threshold) {
     score_index_vec->push_back(std::make_pair(scores[i], i));
   }
  }
  if (score_index_vec->size() == 0) return;

  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   PairDescend<int, Dtype>);
  // LOG(INFO)<<score_index_vec->size();
  if (top_k > -1 && top_k < score_index_vec->size()) {
   score_index_vec->resize(top_k);
  }
}

template void GetTopKsWithThreshold(const vector<float> &scores, const float threshold,
                           const int top_k, vector<pair<float, int> > *score_index_vec);
template void GetTopKsWithThreshold(const vector<double> &scores, const double threshold,
                          const int top_k, vector<pair<double, int> > *score_index_vec);

template <typename Dtype>
void NmsFast(const vector<LabeledBBox<Dtype> > &bboxes,
             const vector<Dtype> &scores,
             const Dtype conf_threshold,
             const Dtype nms_threshold,
             const int top_k,
             vector<int> *indices) {
  CHECK_EQ(bboxes.size(), scores.size()) << "bboxes and scores have different size.";
  vector<pair<Dtype, int> > score_index_vec;
  GetTopKsWithThreshold(scores, conf_threshold, top_k, &score_index_vec);
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        Dtype overlap = bboxes[idx].bbox.compute_iou(bboxes[kept_idx].bbox);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
}

template void NmsFast(const vector<LabeledBBox<float> > &bboxes,
                      const vector<float> &scores,
                      const float conf_threshold,
                      const float nms_threshold,
                      const int top_k,
                      vector<int> *indices);
template void NmsFast(const vector<LabeledBBox<double> > &bboxes,
                      const vector<double> &scores,
                      const double conf_threshold,
                      const double nms_threshold,
                      const int top_k,
                      vector<int> *indices);

template <typename Dtype>
void NmsFastWithVoting(vector<LabeledBBox<Dtype> > *bboxes,
                       const vector<Dtype> &scores,
                       const Dtype conf_threshold,
                       const Dtype nms_threshold,
                       const int top_k,
                       const Dtype voting_thre,
                       vector<int> *indices) {
  CHECK_EQ(bboxes->size(), scores.size()) << "bboxes and scores have different size.";
  vector<pair<Dtype, int> > score_index_vec;
  // LOG(INFO)<<"hzw scores.size()"<<scores.size();
  GetTopKsWithThreshold(scores, conf_threshold, top_k, &score_index_vec);
  // LOG(INFO)<<"score_index_vec.size()"<<score_index_vec.size();
  if (score_index_vec.size() == 0) return;
  // Do nms.
  indices->clear();
  map<int, vector<BoundingBox<Dtype> > > box_voting;
  map<int, vector<Dtype> > box_voting_score;
  while (score_index_vec.size() != 0) {
   // LOG(INFO)<<"hzw nms"<<"indices->size()"<<indices->size();
   const int idx = score_index_vec.front().second;
   bool keep = true;
   for (int k = 0; k < indices->size(); ++k) {
     const int kept_idx = (*indices)[k];
     Dtype overlap = (*bboxes)[idx].bbox.compute_iou((*bboxes)[kept_idx].bbox);
     if (overlap >= voting_thre) {
       box_voting[kept_idx].push_back((*bboxes)[idx].bbox);
       box_voting_score[kept_idx].push_back(scores[idx]);
     }
     if (overlap >= nms_threshold){
       keep = false;
     }
   }
   if (keep) {
     // LOG(INFO)<<"hzw"<<"keep";
     indices->push_back(idx);
     box_voting[idx].push_back((*bboxes)[idx].bbox);
     box_voting_score[idx].push_back(scores[idx]);
   }
   score_index_vec.erase(score_index_vec.begin());
  }
  // voting
  for (typename map<int, vector<BoundingBox<Dtype> > >::iterator it = box_voting.begin();
      it != box_voting.end(); ++it) {
    int kept_id = it->first;
    vector<BoundingBox<Dtype> > &voting_list = it->second;
    if (voting_list.size()==1) continue;
    vector<Dtype> &voting_score = box_voting_score[kept_id];
    Dtype all_score = 0.0;
    for (int i = 0; i < voting_score.size(); ++i) {
     all_score += voting_score[i];
    }
    Dtype xmin = 0.0, xmax = 0.0 , ymin = 0.0 , ymax= 0.0;
    for (int i = 0; i < voting_list.size(); ++i) {
     Dtype scale = voting_score[i] / all_score;
     xmin += voting_list[i].x1_ * scale;
     ymin += voting_list[i].y1_ * scale;
     xmax += voting_list[i].x2_ * scale;
     ymax += voting_list[i].y2_ * scale;
    }
    (*bboxes)[kept_id].bbox.x1_ = xmin;
    (*bboxes)[kept_id].bbox.y1_ = ymin;
    (*bboxes)[kept_id].bbox.x2_ = xmax;
    (*bboxes)[kept_id].bbox.y2_ = ymax;
  }
}

template void NmsFastWithVoting(vector<LabeledBBox<float> > *bboxes,
                                const vector<float> &scores,
                                const float conf_threshold,
                                const float nms_threshold,
                                const int top_k,
                                const float voting_thre,
                                vector<int> *indices);
template void NmsFastWithVoting(vector<LabeledBBox<double> > *bboxes,
                                const vector<double> &scores,
                                const double conf_threshold,
                                const double nms_threshold,
                                const int top_k,
                                const double voting_thre,
                                vector<int> *indices);

template <typename Dtype>
void NmsFastWithVotingChangeScore(vector<LabeledBBox<Dtype> > *bboxes,
                        vector<Dtype> *scores,
                       const Dtype conf_threshold,
                       const Dtype nms_threshold,
                       const int top_k,
                       const Dtype voting_thre,
                       vector<int> *indices) {
  CHECK_EQ(bboxes->size(), scores->size()) << "bboxes and scores have different size.";
  vector<pair<Dtype, int> > score_index_vec;
  // LOG(INFO)<<"hzw scores.size()"<<scores.size();
  GetTopKsWithThreshold(*scores, conf_threshold, top_k, &score_index_vec);
  // LOG(INFO)<<"score_index_vec.size()"<<score_index_vec.size();
  if (score_index_vec.size() == 0) return;
  // Do nms.
  indices->clear();
  map<int, vector<BoundingBox<Dtype> > > box_voting;
  map<int, vector<Dtype> > box_voting_score;
  while (score_index_vec.size() != 0) {
   // LOG(INFO)<<"hzw nms"<<"indices->size()"<<indices->size();
   const int idx = score_index_vec.front().second;
   bool keep = true;
   for (int k = 0; k < indices->size(); ++k) {
     const int kept_idx = (*indices)[k];
     Dtype overlap = (*bboxes)[idx].bbox.compute_iou((*bboxes)[kept_idx].bbox);
     if (overlap >= voting_thre) {
       box_voting[kept_idx].push_back((*bboxes)[idx].bbox);
       box_voting_score[kept_idx].push_back((*scores)[idx]);
     }
     if (overlap >= nms_threshold){
       keep = false;
     }
   }
   if (keep) {
     // LOG(INFO)<<"hzw"<<"keep";
     indices->push_back(idx);
     box_voting[idx].push_back((*bboxes)[idx].bbox);
     box_voting_score[idx].push_back((*scores)[idx]);
   }
   score_index_vec.erase(score_index_vec.begin());
  }
  // voting
  for (typename map<int, vector<BoundingBox<Dtype> > >::iterator it = box_voting.begin();
      it != box_voting.end(); ++it) {
    int kept_id = it->first;
    vector<BoundingBox<Dtype> > &voting_list = it->second;
    if (voting_list.size()==1) continue;
    vector<Dtype> &voting_score = box_voting_score[kept_id];
    Dtype all_score = 0.0;
    for (int i = 0; i < voting_score.size(); ++i) {
     all_score += voting_score[i];
    }
    Dtype all_score2 = 0.0;
    for (int i = 0; i < voting_score.size(); ++i) {
        all_score2 += voting_score[i] * voting_score[i];
    }
    (*scores)[kept_id] = Dtype(all_score2 / all_score);
    Dtype xmin = 0.0, xmax = 0.0 , ymin = 0.0 , ymax= 0.0;
    for (int i = 0; i < voting_list.size(); ++i) {
     Dtype scale = voting_score[i] / all_score;
     xmin += voting_list[i].x1_ * scale;
     ymin += voting_list[i].y1_ * scale;
     xmax += voting_list[i].x2_ * scale;
     ymax += voting_list[i].y2_ * scale;
    }
    (*bboxes)[kept_id].bbox.x1_ = xmin;
    (*bboxes)[kept_id].bbox.y1_ = ymin;
    (*bboxes)[kept_id].bbox.x2_ = xmax;
    (*bboxes)[kept_id].bbox.y2_ = ymax;
  }      
}

template void NmsFastWithVotingChangeScore(vector<LabeledBBox<float> > *bboxes,
                                 vector<float> *scores,
                                const float conf_threshold,
                                const float nms_threshold,
                                const int top_k,
                                const float voting_thre,
                                vector<int> *indices);
template void NmsFastWithVotingChangeScore(vector<LabeledBBox<double> > *bboxes,
                                 vector<double> *scores,
                                const double conf_threshold,
                                const double nms_threshold,
                                const int top_k,
                                const double voting_thre,
                                vector<int> *indices);

template <typename Dtype>
void ToCorner(const BoundingBox<Dtype>& input, BoundingBox<Dtype>* output) {
  output->x1_ = input.x1_ - input.x2_  / 2.;
  output->x2_ = input.x1_ + input.x2_  / 2.;
  output->y1_ = input.y1_ - input.y2_  / 2.;
  output->y2_ = input.y1_ + input.y2_  / 2.;
}

template void ToCorner(const BoundingBox<float>& input, BoundingBox<float>* output);
template void ToCorner(const BoundingBox<double>& input, BoundingBox<double>* output);

template <typename Dtype>
void ToCenter(const BoundingBox<Dtype>& input, BoundingBox<Dtype>* output) {
  output->x1_ = (input.x1_ + input.x2_) / 2.;
  output->y1_ = (input.y1_ + input.y2_) / 2.;
  output->x2_ = input.x2_ - input.x1_;
  output->y2_ = input.y2_ - input.y1_;
}

template void ToCenter(const BoundingBox<float>& input, BoundingBox<float>* output);
template void ToCenter(const BoundingBox<double>& input, BoundingBox<double>* output);

template <typename Dtype>
int GetBBoxLevel(const BoundingBox<Dtype>& bbox, map<int, Dtype> &size_thre) {
  float size = bbox.compute_area();
  int level = -1;
  if (size > size_thre[0])
    level ++;
  if (size > size_thre[1])
    level ++;
  if (size > size_thre[2])
    level ++;
  if (size > size_thre[3])
    level ++;
  if (size > size_thre[4])
    level ++;
  if (size > size_thre[5])
    level ++;
  if (size > size_thre[6])
    level ++;
  return level;
}

template int GetBBoxLevel(const BoundingBox<float>& bbox, map<int, float> &size_thre);
template int GetBBoxLevel(const BoundingBox<double>& bbox, map<int, double> &size_thre);

template <typename Dtype>
void GetLeveledGTBBoxes(map<int, Dtype> &size_thre,
                        map<int, map<int, vector<LabeledBBox<Dtype> > > > &all_gtboxes,
                        vector<map<int, map<int, vector<LabeledBBox<Dtype> > > > > *leveld_gtboxes) {
  for (typename map<int, map<int, vector<LabeledBBox<Dtype> > > >::iterator it = all_gtboxes.begin();
       it != all_gtboxes.end(); ++it) {
    int bindex = it->first;
    for (typename map<int, vector<LabeledBBox<Dtype> > >::iterator iit = it->second.begin();
        iit != it->second.end(); ++iit) {
      int cid = iit->first;
    // LOG(INFO)<<"hzw gtcid"<<cid;
      vector<LabeledBBox<Dtype> >& gtboxes = iit->second;
      if (gtboxes.size() == 0) continue;
      for (int j = 0; j < gtboxes.size(); ++j) {
        int level = GetBBoxLevel(gtboxes[j].bbox, size_thre);
        if (level < 0) continue;
        for (int l = 0; l < level + 1; ++l) {
          map<int, map<int, vector<LabeledBBox<Dtype> > > > &l_gtboxes = (*leveld_gtboxes)[l];
          l_gtboxes[bindex][cid].push_back(gtboxes[j]);
        }
      }
    }
  }
}

template void GetLeveledGTBBoxes(map<int, float> &size_thre,
                                 map<int, map<int, vector<LabeledBBox<float> > > > &all_gtboxes,
                                 vector<map<int, map<int, vector<LabeledBBox<float> > > > > *leveld_gtboxes);
template void GetLeveledGTBBoxes(map<int, double> &size_thre,
                                 map<int, map<int, vector<LabeledBBox<double> > > > &all_gtboxes,
                                 vector<map<int, map<int, vector<LabeledBBox<double> > > > > *leveld_gtboxes);

template <typename Dtype>
void GetLeveledEvalDetections(map<int, map<int, vector<LabeledBBox<Dtype> > > >& l_gtboxes,
                              map<int, map<int, vector<LabeledBBox<Dtype> > > >& all_detections,
                              const Dtype size_thre, const Dtype iou_threshold,
                              const int num_classes, const int level, const int diff,
                              const vector<int>& gt_labels,
                              vector<vector<Dtype> >* l_res) {
  l_res->clear();
  map<int, int> num_gts;
  for (typename map<int, map<int, vector<LabeledBBox<Dtype> > > >::iterator it = l_gtboxes.begin();
       it != l_gtboxes.end(); ++it) {
     for (typename map<int, vector<LabeledBBox<Dtype> > >::iterator iit = it->second.begin();
         iit != it->second.end(); ++iit) {
        int cid = iit->first;
        int num = iit->second.size();
        if (num_gts.find(cid) == num_gts.end()) {
          num_gts[cid] = num;
        } else {
          num_gts[cid] += num;
        }
     }
  }
  if (gt_labels.size() > 0) {
    // use gt_labels
    for (int i = 0; i < gt_labels.size(); ++i) {
      int c = gt_labels[i];
      vector<Dtype> res;
      res.push_back(diff);
      res.push_back(level);
      res.push_back(-1);
      res.push_back(c);
      if (num_gts.find(c) == num_gts.end()) {
        res.push_back(0);
      } else {
        res.push_back(num_gts[c]);
      }
      res.push_back(-1);
      res.push_back(-1);
      l_res->push_back(res);
    }
  } else {
    // use num_classes
    for (int c = 0; c < num_classes - 1; ++c) {
      vector<Dtype> res;
      res.push_back(diff);
      res.push_back(level);
      res.push_back(-1);
      res.push_back(c);
      if (num_gts.find(c) == num_gts.end()) {
        res.push_back(0);
      } else {
        res.push_back(num_gts[c]);
      }
      res.push_back(-1);
      res.push_back(-1);
      l_res->push_back(res);
    }
  }
  // output detboxes
  for (typename map<int, map<int, vector<LabeledBBox<Dtype> > > >::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int bindex = it->first;
    map<int, vector<LabeledBBox<Dtype> > > &detections = it->second;
    if (l_gtboxes.find(bindex) == l_gtboxes.end()) {
      // 全部是FP
      for (typename map<int, vector<LabeledBBox<Dtype> > >::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int cid = iit->first;
        vector<LabeledBBox<Dtype> > &fp_bboxes = iit->second;
        if(fp_bboxes.size() == 0) continue;
        for (int i = 0; i < fp_bboxes.size(); ++i) {
          if (fp_bboxes[i].bbox.compute_area() > size_thre) {
            vector<Dtype> res;
            res.push_back(diff);
            res.push_back(level);
            res.push_back(bindex);
            res.push_back(cid);
            res.push_back(fp_bboxes[i].score);
            res.push_back(0);
            res.push_back(1);
            l_res->push_back(res);
          }
        }
      }
    } else {
      map<int, vector<LabeledBBox<Dtype> > > &label_bboxes = l_gtboxes.find(bindex)->second;
      for (typename map<int, vector<LabeledBBox<Dtype> > >::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int cid = iit->first;
        vector<LabeledBBox<Dtype> > &bboxes = iit->second;
        if (bboxes.size() == 0) continue;
        if (label_bboxes.find(cid) == label_bboxes.end()) {
          // 全部是FP
          if (bboxes.size() == 0) continue;
          for (int i = 0; i < bboxes.size(); ++i) {
            if (bboxes[i].bbox.compute_area() > size_thre) {
              vector<Dtype> res;
              res.push_back(diff);
              res.push_back(level);
              res.push_back(bindex);
              res.push_back(cid);
              res.push_back(bboxes[i].score);
              res.push_back(0);
              res.push_back(1);
              l_res->push_back(res);
            }
          }
        } else {
          vector<LabeledBBox<Dtype> > &gt_bboxes = label_bboxes.find(cid)->second;
          vector<bool> visited(gt_bboxes.size(), false);
          std::sort(bboxes.begin(), bboxes.end(), BBoxDescend<Dtype>);
          for (int i = 0; i < bboxes.size(); ++i) {
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              Dtype overlap = bboxes[i].bbox.compute_iou(gt_bboxes[j].bbox);
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
            if (overlap_max >= iou_threshold) {
              if (!visited[jmax]) {
                // TP
                vector<Dtype> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(bindex);
                res.push_back(cid);
                res.push_back(bboxes[i].score);
                res.push_back(1);
                res.push_back(0);
                l_res->push_back(res);
                visited[jmax] = true;
              } else {
                // FD
                vector<Dtype> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(bindex);
                res.push_back(cid);
                res.push_back(bboxes[i].score);
                res.push_back(0);
                res.push_back(1);
                l_res->push_back(res);
              }
            } else {
              // FP
              if (bboxes[i].bbox.compute_area() > size_thre) {
                vector<Dtype> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(bindex);
                res.push_back(cid);
                res.push_back(bboxes[i].score);
                res.push_back(0);
                res.push_back(1);
                l_res->push_back(res);
              }
            }
          }
        }
      }
    }
  }
}

template void GetLeveledEvalDetections(map<int, map<int, vector<LabeledBBox<float> > > >& l_gtboxes,
                                       map<int, map<int, vector<LabeledBBox<float> > > >& all_detections,
                                       const float size_thre, const float iou_threshold,
                                       const int num_classes, const int level, const int diff,
                                       const vector<int>& gt_labels,
                                       vector<vector<float> >* l_res);
template void GetLeveledEvalDetections(map<int, map<int, vector<LabeledBBox<double> > > >& l_gtboxes,
                                       map<int, map<int, vector<LabeledBBox<double> > > >& all_detections,
                                       const double size_thre, const double iou_threshold,
                                       const int num_classes, const int level, const int diff,
                                       const vector<int>& gt_labels,
                                       vector<vector<double> >* l_res);

template <typename Dtype>
void ShowBBox(const vector<cv::Mat> &images,
              const vector<vector<vector<LabeledBBox<Dtype> > > > &all_dets,
              const VisualizeParameter &visual_param) {
 static clock_t start_clock = clock();
 static clock_t total_time_start = start_clock;
 static long total_run_time = 0;
 static long total_frame = 0;
 const int num_img = images.size();
 if (num_img == 0) {
   return;
 }
 CHECK_EQ(num_img, all_dets.size());
 int num_classes = all_dets[0].size();
 // call FPS
 float fps = num_img / (static_cast<double>(clock() - start_clock) / CLOCKS_PER_SEC);
 total_run_time = clock() - total_time_start;
 total_frame += num_img;
 /*****************************************************************************/
 float run_ftime_sec = static_cast<double>(total_run_time) / CLOCKS_PER_SEC;
 int run_ms = (static_cast<long>(run_ftime_sec * 1000)) % 1000;
 int run_s  = static_cast<int>(run_ftime_sec);
 int run_hour = run_s / 3600;
 run_s = run_s % 3600;
 int run_min = run_s / 60;
 run_s = run_s % 60;
 std::cout << std::setiosflags(std::ios::fixed);
 std::cout << "================================================================" << std::endl;
 std::cout << "Current Frame Number  : " << total_frame << "\n";
 std::cout << "Current Process Speed : " << std::setprecision(1) << fps << " FPS\n";
 if(run_hour>0){
   std::cout << "Current Time          : " << run_hour << " hour " \
             << run_min << " min " << run_s << " s " << run_ms << " ms\n";
 }
 else if(run_min>0){
   std::cout << "Current Time          : " << run_min << " min " \
             << run_s << " s " << run_ms << " ms\n";
 }
 else if(run_s>0){
   std::cout << "Current Time          : " << run_s << " s " << run_ms << " ms\n";
 }
 else{
   std::cout << "Current Time          : " << run_ms << " ms\n";
 }
 std::cout << std::endl;
 // check visual param
 CHECK(visual_param.has_visualize()) << "visualize must be provided.";
 CHECK(visual_param.visualize()) << "visualize must be enabled.";
 CHECK(visual_param.has_color_param())  << "color_param must be provided";
 const ColorParameter &color_param =  visual_param.color_param();
 CHECK(visual_param.has_display_maxsize()) << "display_maxsize must be provided.";
 const int display_max_size = visual_param.display_maxsize();
 CHECK(visual_param.has_line_width()) << "line_width must be provided.";
 const int box_line_width = visual_param.line_width();
 const float conf_threshold = visual_param.conf_threshold();
 const float size_threshold = visual_param.size_threshold();
 /*****************************************************************************/
 const int width = images[0].cols;
 const int height = images[0].rows;
 const int maxLen = (width > height) ? width : height;
 const float ratio = (float)display_max_size / maxLen;
 const int display_width = static_cast<int>(width * ratio);
 const int display_height = static_cast<int>(height * ratio);
 CHECK_EQ(color_param.rgb_size(), num_classes);
 for (int i = 0; i < color_param.rgb_size(); ++i) {
   CHECK_EQ(color_param.rgb(i).val_size(), 3);
 }
 // draw
 for (int i = 0; i < num_img; ++i) {
   cv::Mat display_image;
   cv::resize(images[i], display_image, cv::Size(display_width,display_height), cv::INTER_LINEAR);
   std::cout << "The input image size is   : " << width << "x" << height << '\n';
   std::cout << "The display image size is : " << display_width << "x" << display_height<< '\n';
   std::cout << std::endl;
   for (int j = 0; j < num_classes; ++j) {
     const vector<LabeledBBox<Dtype> >& dets = all_dets[i][j];
     if (dets.size() == 0) continue;
     for (int p = 0; p < dets.size(); ++p) {
       const LabeledBBox<Dtype>& tbbox = dets[p];
       const Color& rgb = color_param.rgb(tbbox.cid);
       cv::Scalar line_color(rgb.val(2),rgb.val(1),rgb.val(0));
       if (tbbox.score < conf_threshold) continue;
       if (tbbox.bbox.compute_area() < size_threshold) continue;
       std::cout << "A person is detected: " << std::setprecision(3) << tbbox.score << "\n";
       cv::Point top_left_pt(static_cast<int>(tbbox.bbox.x1_ * display_width),
                             static_cast<int>(tbbox.bbox.y1_ * display_height));
       cv::Point bottom_right_pt(static_cast<int>(tbbox.bbox.x2_ * display_width),
                             static_cast<int>(tbbox.bbox.y2_ * display_height));
       cv::rectangle(display_image, top_left_pt, bottom_right_pt, line_color, box_line_width);
     }
   }
   cv::imshow("remo", display_image);
   if (cv::waitKey(1) == 27) {
     raise(SIGINT);
   }
 }
 start_clock = clock();
}

template void ShowBBox(const vector<cv::Mat> &images,
              const vector<vector<vector<LabeledBBox<float> > > > &all_dets,
              const VisualizeParameter &visual_param);
template void ShowBBox(const vector<cv::Mat> &images,
            const vector<vector<vector<LabeledBBox<double> > > > &all_dets,
            const VisualizeParameter &visual_param);

}
