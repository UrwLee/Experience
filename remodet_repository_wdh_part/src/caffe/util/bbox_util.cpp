#include <algorithm>
#include <csignal>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "boost/iterator/counting_iterator.hpp"

#include "caffe/util/bbox_util.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

bool SortBBoxAscend(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  return bbox1.score() > bbox2.score();
}

template <typename T>
bool SortScorePairAscend(const pair<float, T> &pair1,
                         const pair<float, T> &pair2) {
  return pair1.first < pair2.first;
}

// Explicit initialization.
template bool SortScorePairAscend(const pair<float, int> &pair1,
                                  const pair<float, int> &pair2);
template bool SortScorePairAscend(const pair<float, pair<int, int> > &pair1,
                                  const pair<float, pair<int, int> > &pair2);

template <typename T>
bool SortScorePairDescend(const pair<float, T> &pair1,
                          const pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int> &pair1,
                                   const pair<float, int> &pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> > &pair1,
                                   const pair<float, pair<int, int> > &pair2);

NormalizedBBox UnitBBox() {
  NormalizedBBox unit_bbox;
  unit_bbox.set_xmin(0.);
  unit_bbox.set_ymin(0.);
  unit_bbox.set_xmax(1.);
  unit_bbox.set_ymax(1.);
  return unit_bbox;
}

void IntersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                   NormalizedBBox *intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

float BBoxSize(const NormalizedBBox &bbox, const bool normalized) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}

void ClipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ScaleBBox(const NormalizedBBox &bbox, const int height, const int width,
               NormalizedBBox *scale_bbox) {
  scale_bbox->set_xmin(bbox.xmin() * width);
  scale_bbox->set_ymin(bbox.ymin() * height);
  scale_bbox->set_xmax(bbox.xmax() * width);
  scale_bbox->set_ymax(bbox.ymax() * height);
  scale_bbox->clear_size();
  bool normalized = !(width > 1 || height > 1);
  scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
  scale_bbox->set_difficult(bbox.difficult());
}

void LocateBBox(const NormalizedBBox &src_bbox, const NormalizedBBox &bbox,
                NormalizedBBox *loc_bbox) {
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
  loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
  loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
  loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
  loc_bbox->set_difficult(bbox.difficult());
}

bool ProjectBBox(const NormalizedBBox &src_bbox, const NormalizedBBox &bbox,
                 NormalizedBBox *proj_bbox) {
  if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin() ||
      bbox.ymin() >= src_bbox.ymax() || bbox.ymax() <= src_bbox.ymin()) {
    return false;
  }
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
  proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
  proj_bbox->set_difficult(bbox.difficult());
  ClipBBox(*proj_bbox, proj_bbox);
  if (BBoxSize(*proj_bbox) > 0) {
    return true;
  } else {
    return false;
  }
}

float JaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                     const bool normalized) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
  } else {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

float BBoxCoverage(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_size = BBoxSize(intersect_bbox);
  if (intersect_size > 0) {
    float bbox1_size = BBoxSize(bbox1);
    return intersect_size / bbox1_size;
  } else {
    return 0.;
  }
}

bool MeetEmitConstraint(const NormalizedBBox &src_bbox,
                        const NormalizedBBox &bbox,
                        const EmitConstraint &emit_constraint) {
  EmitType emit_type = emit_constraint.emit_type();
  // 中心发布准则!
  if (emit_type == EmitConstraint_EmitType_CENTER) {
    // 目标box的中心
    float x_center = (bbox.xmin() + bbox.xmax()) / 2;
    float y_center = (bbox.ymin() + bbox.ymax()) / 2;
    // src_bbox是裁剪的crop_bbox
    // 如果目标box不在crop_box范围内,则不满足发布条件!
    if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
        y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
      return true;
    } else {
      return false;
    }
  } else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
  // 最小交集发布准则!
  // 只有目标box与裁剪box的交集满足一定条件
  // 即:目标box的大部分区域应落在裁剪的box区域中,才可以.
    float bbox_coverage = BBoxCoverage(bbox, src_bbox);
    return bbox_coverage > emit_constraint.emit_overlap();
  } else {
    LOG(FATAL) << "Unknown emit type.";
    return false;
  }
}

bool MeetEmitConstraint(const NormalizedBBox &src_bbox,
                        const NormalizedBBox &bbox,
                        const EmitConstraint &emit_constraint,
                        const int emitType) {
  EmitType emit_type = emit_constraint.emit_type();
  // 中心发布准则!
  if (emit_type == EmitConstraint_EmitType_CENTER) {
    // 目标box的中心
    float x_center = (bbox.xmin() + bbox.xmax()) / 2;
    float y_center = (bbox.ymin() + bbox.ymax()) / 2;
    // src_bbox是裁剪的crop_bbox
    // 如果目标box不在crop_box范围内,则不满足发布条件!
    if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
        y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
      return true;
    } else {
      return false;
    }
  } else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
  // 最小交集发布准则!
  // 只有目标box与裁剪box的交集满足一定条件
  // 即:目标box的大部分区域应落在裁剪的box区域中,才可以.
    float bbox_coverage = BBoxCoverage(bbox, src_bbox);
    float emit_overlap = 0;
    if (emitType == 0)
      emit_overlap = emit_constraint.emit_overlap();
    else if (emitType == 1)
      emit_overlap = emit_constraint.emit_overlap_dir();
    else if (emitType == 2)
      emit_overlap = emit_constraint.emit_overlap_pose();
    else
      LOG(FATAL) << "Only 0/1/2 is allowed.";
    return bbox_coverage > emit_overlap;
  } else {
    LOG(FATAL) << "Unknown emit type.";
    return false;
  }
}

/**
 *
 */
void EncodeBBox(const NormalizedBBox &prior_bbox,
                const vector<float> &prior_variance, const CodeType code_type,
                const bool encode_variance_in_target,
                const NormalizedBBox &bbox, NormalizedBBox *encode_bbox) {
  //使用CORNER的编码方式
  //编码的输出是bbox与prior之间的坐标差值
  //采用坐标是box的四个坐标边界：xmin/ymin/xmax/ymax
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    /**
     * 目标中已包含偏差
     * 直接使用bbox与prior之差
     */
    if (encode_variance_in_target) {
      encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
      encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
      encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
      encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());
    } else {
      // Encode variance in bbox.
      // 采用此种方式
      CHECK_EQ(prior_variance.size(), 4);
      for (int i = 0; i < prior_variance.size(); ++i) {
        CHECK_GT(prior_variance[i], 0);
      }
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) /
                            prior_variance[0]);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) /
                            prior_variance[1]);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) /
                            prior_variance[2]);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) /
                            prior_variance[3]);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    //采用CENTOR方式
    //采用此种方式
    //
    //计算prior的长宽
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    //计算prior的中心坐标
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    /**
     * 计算bbox的长宽和中心坐标
     */
    float bbox_width = bbox.xmax() - bbox.xmin();
    CHECK_GT(bbox_width, 0);
    float bbox_height = bbox.ymax() - bbox.ymin();
    CHECK_GT(bbox_height, 0);
    float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
    float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

    //不使用
    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
      encode_bbox->set_xmax(log(bbox_width / prior_width));
      encode_bbox->set_ymax(log(bbox_height / prior_height));
    } else {
      // Encode variance in bbox.
      /**
       * 中心坐标编码
       * 编码后，采用的坐标是：
       * 中心坐标差（x/y）
       * 长宽的对数值
       */
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width /
                            prior_variance[0]);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height /
                            prior_variance[1]);
      encode_bbox->set_xmax(log(bbox_width / prior_width) / prior_variance[2]);
      encode_bbox->set_ymax(log(bbox_height / prior_height) /
                            prior_variance[3]);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
}

void DecodeBBox(const NormalizedBBox &prior_bbox,
                const vector<float> &prior_variance, const CodeType code_type,
                const bool variance_encoded_in_target,
                const NormalizedBBox &bbox, NormalizedBBox *decode_bbox) {
  //使用CORNER方式编码
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(prior_bbox.xmin() +
                            prior_variance[0] * bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() +
                            prior_variance[1] * bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() +
                            prior_variance[2] * bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() +
                            prior_variance[3] * bbox.ymax());
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    // 如果使用CENTER方式
    // 分别是<x,y><w,h>
    // prior的长和宽
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    // prior的中心位置
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    // 解码box的中心和长宽
    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    // 如果偏差包含在目标中，仅需包含bbox的偏差信息即可
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(bbox.xmax()) * prior_width;
      decode_bbox_height = exp(bbox.ymax()) * prior_height;
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y =
          prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(prior_variance[2] * bbox.xmax()) * prior_width;
      decode_bbox_height = exp(prior_variance[3] * bbox.ymax()) * prior_height;
    }
    decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
    decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
    decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
    decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
  float bbox_size = BBoxSize(*decode_bbox);
  decode_bbox->set_size(bbox_size);
}

void DecodeBBoxes(const vector<NormalizedBBox> &prior_bboxes,
                  const vector<vector<float> > &prior_variances,
                  const CodeType code_type,
                  const bool variance_encoded_in_target,
                  const vector<NormalizedBBox> &bboxes,
                  vector<NormalizedBBox> *decode_bboxes) {
  // prior/variance/bbox三者应具有完全相同的size
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  // box的数量
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    // variance应该具有4个尺度
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  // 遍历所有的boxes
  for (int i = 0; i < num_bboxes; ++i) {
    NormalizedBBox decode_bbox;
    // 解码
    DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
               variance_encoded_in_target, bboxes[i], &decode_bbox);
    // 送入结果队列
    decode_bboxes->push_back(decode_bbox);
  }
}

void DecodeBBoxesAll(const vector<vector<NormalizedBBox> > &all_loc_preds,
                     const vector<NormalizedBBox> &prior_bboxes,
                     const vector<vector<float> > &prior_variances,
                     const int num, const CodeType code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<NormalizedBBox> > *all_decode_bboxes) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<NormalizedBBox> &decode_bboxes = (*all_decode_bboxes)[i];
    const vector<NormalizedBBox> &label_loc_preds = all_loc_preds[i];
    DecodeBBoxes(prior_bboxes, prior_variances, code_type,
                 variance_encoded_in_target, label_loc_preds,
                 &decode_bboxes);
  }
}

void MatchBBox(const vector<NormalizedBBox> &gt_bboxes,
               const vector<NormalizedBBox> &pred_bboxes,
               const MatchType match_type,
               const float overlap_threshold,
               vector<float> *match_overlaps,
               map<int, int> *match_indices,
               vector<int> *unmatch_indices) {
  // 估计box的数量
  int num_pred = pred_bboxes.size();
  match_overlaps->clear();
  match_indices->clear();
  unmatch_indices->clear();

  match_overlaps->resize(num_pred, 0.);

  // 所有的prior-box匹配状态, 0:未匹配, 1:已匹配
  vector<int> match_status;
  match_status.resize(num_pred, 0);

  // gt编号列表
  int num_gt = 0;
  vector<int> gt_indices;
  num_gt = gt_bboxes.size();
  for (int i = 0; i < num_gt; ++i) {
    gt_indices.push_back(i);
  }
  // 如果没有gt结果，则直接返回，结果为空
  if (num_gt == 0) {
    return;
  }
  // 定义[i,j]->第i个估计box与第j个gt-box之间的iou值
  // 该表完成估计box与gt box之间的iou映射
  map<int, map<int, float> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlap(pred_bboxes[i], gt_bboxes[gt_indices[j]]);
      if (overlap > 1e-6) {
        // 计算第i个估计box的最大iou值
        (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }

  // 双向匹配
  // 初始化为gt box的编号
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  /**
   * 匹配过程：
   * 每次查找未匹配box中[i,j]的iou最大值对,完成匹配!
   */
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    // 查找iou最大值对
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      //估计box的编号
      int i = it->first;
      if (match_status[i]) {
        // 该估计box已经匹配过了，跳过
        continue;
      }
      // 遍历剩下的每个gt box，查找最大交集对（i,j） -> (overlap)
      for (int p = 0; p < gt_pool.size(); ++p) {
        // 取出其gt box的编号
        int j = gt_pool[p];
        // 在该box的有交集的gt box中寻找其编号
        // 如果没有找到，则不需要匹配了，直接跳过
        if (it->second.find(j) == it->second.end()) {
          continue;
        }
        // 如果找到
        if (it->second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      // 将查找到的最大值对完成匹配!
      CHECK_EQ(match_status[max_idx], 0);
      match_status[max_idx] = 1;
      // 匹配过程
      // 添加到完整匹配列表
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // 对与已经匹配过的gtbox，将其从gt池中擦除
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  /**
   * 双向：前面已经完成，PASS
   * PER_PREDICTION:在剩下的估计box中获取较大交集值的boxes
   */
  switch (match_type) {
  case MultiBoxLossParameter_MatchType_BIPARTITE:
    break;
  case MultiBoxLossParameter_MatchType_PER_PREDICTION:
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if (match_status[i]) {
        continue;
      }
      // 遍历所有gt-box,查找满足阈值条件的最大匹配对
      int max_gt_idx = -1;
      float max_overlap = -1;
      for (int j = 0; j < num_gt; ++j) {
        // 如果两者无交集，跳过
        if (it->second.find(j) == it->second.end()) {
          continue;
        }
        //获取最大交集度的gt box
        float overlap = it->second[j];
        if (overlap >= overlap_threshold && overlap > max_overlap) {
          max_gt_idx = j;
          max_overlap = overlap;
        }
      }
      // 如果最大映射结果！=-1，认为实现了匹配
      if (max_gt_idx != -1) {
        // 匹配!
        CHECK_EQ(match_status[i], 0);
        match_status[i] = 1;
        (*match_indices)[i] = gt_indices[max_gt_idx];
        (*match_overlaps)[i] = max_overlap;
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown matching type.";
    break;
  }

  /**
   * 获取非匹配列表
   */
  for (int i = 0; i < num_pred; ++i) {
    if (match_indices->find(i) == match_indices->end()) {
      unmatch_indices->push_back(i);
    }
  }
  return;
}

template <typename Dtype>
void GetGroundTruth(const Dtype *gt_data, const int num_gt,
                    const int background_label_id, const bool use_difficult_gt,
                    map<int, vector<NormalizedBBox> > *all_gt_bboxes) {
  all_gt_bboxes->clear();
  // 获取所有的记录
  for (int i = 0; i < num_gt; ++i) {
    // 每条记录有8个元素
    int start_idx = i * 8;
    // 获取样本ID
    int item_id = gt_data[start_idx];
    // 如果该样本不含有任何记录,则全部元素为-1,直接跳过
    if (item_id < 0) {
      continue;
    }
    int label = gt_data[start_idx + 1];
    // int instance_id = gt_data[start_idx + 3];
    // 检查,不能是背景,否则出错
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      continue;
    }
    // box
    NormalizedBBox bbox;
    bbox.set_label(label);
    // 设置其坐标
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    // 设置其diff
    bbox.set_difficult(difficult);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, vector<NormalizedBBox> > *all_gt_bboxes);
template void GetGroundTruth(const double *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, vector<NormalizedBBox> > *all_gt_bboxes);

template <typename Dtype>
void GetGroundTruth(const Dtype *gt_data, const int num_gt,
                    const int background_label_id, const bool use_difficult_gt,
                    map<int, map<int, vector<NormalizedBBox> > > *all_gt_bboxes) {
  all_gt_bboxes->clear();
  vector<NormalizedBBox> gt_box;
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if (item_id < 0) {
      break;
    }
    // int body_id = gt_data[start_idx + 1];
    int part_type = gt_data[start_idx + 1];
    // int instance_id = gt_data[start_idx + 3];
    CHECK_NE(background_label_id, part_type)
        << "Found background label in the dataset.";
    // CHECK_EQ(part_type, 1) << "only body-part is supported.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) continue;
    NormalizedBBox bbox;
    bbox.set_label(part_type);
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    if (bbox_size <= 0.001) continue;
    bbox.set_size(bbox_size);
    bbox.set_score(Dtype(1.0));
    (*all_gt_bboxes)[item_id][part_type].push_back(bbox);
    gt_box.push_back(bbox);
  }

  for (int i=0; i < gt_box.size(); ++i){
    // LOG(INFO) << "GT index: " << i+1 << "  size: " << BBoxSize(gt_box[i]);
    // LOG(INFO) << "$gt_size: " << BBoxSize(gt_box[i]);
    for (int j=0; j< gt_box.size(); ++j){
      if (i==j) continue;
      float overlap = JaccardOverlap(gt_box[i], gt_box[j], true);
      if (overlap == 0) continue;
      // LOG(INFO) << "    GT index: " << j+1  << " overlap: " << overlap;
    }
  }

}

// Explicit initialization.
template void GetGroundTruth(const float *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, LabelBBox> *all_gt_bboxes);
template void GetGroundTruth(const double *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, LabelBBox> *all_gt_bboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype *loc_data, const int num,
                       const int num_priors,
                       vector<vector<NormalizedBBox> > *loc_preds) {
  loc_preds->clear();
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<NormalizedBBox>& item_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_priors; ++p) {
      int start_idx = p * 4;
      NormalizedBBox bbox;
      bbox.set_xmin(loc_data[start_idx]);
      bbox.set_ymin(loc_data[start_idx + 1]);
      bbox.set_xmax(loc_data[start_idx + 2]);
      bbox.set_ymax(loc_data[start_idx + 3]);
      item_bbox.push_back(bbox);
    }
    loc_data += num_priors * 4;
  }
}

// Explicit initialization.
template void GetLocPredictions(const float *loc_data, const int num,
                                const int num_priors,
                                vector<vector<NormalizedBBox> > *loc_preds);
template void GetLocPredictions(const double *loc_data, const int num,
                                const int num_priors,
                                vector<vector<NormalizedBBox> > *loc_preds);

template <typename Dtype>
void GetMaxConfScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int start, const int len,
                         vector<vector<pair<int, float> > > *all_max_conf) {
  CHECK_LE(start+len, num_classes);
  CHECK_GE(start, 0);
  CHECK_GT(len, 0);
  all_max_conf->clear();
  all_max_conf->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<pair<int, float> > &max_conf = (*all_max_conf)[i];
    for (int p = 0; p < num_priors; ++p) {
      int max_id = 0;
      float max_val = -1.0;
      for (int j = 0; j < len; ++j) {
        if (conf_data[p * num_classes + start + j] > max_val) {
          max_id = j;
          max_val = conf_data[p * num_classes + start + j];
        }
      }
      max_conf.push_back(std::make_pair(max_id, max_val));
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetMaxConfScores(const float *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int start, const int len,
                         vector<vector<pair<int, float> > > *all_max_conf);
template void GetMaxConfScores(const double *conf_data, const int num,
                        const int num_priors, const int num_classes,
                        const int start, const int len,
                        vector<vector<pair<int, float> > > *all_max_conf);

template <typename Dtype>
void GetMaxConfScores(const Dtype *conf_data, const int num,
                     const int num_priors, const int num_classes,
                     const int start, const int len, const bool class_major,
                     vector<vector<pair<int, float> > > *all_max_conf) {
  CHECK_LE(start+len, num_classes);
  CHECK_GE(start, 0);
  CHECK_GT(len, 0);
  all_max_conf->clear();
  all_max_conf->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<pair<int, float> > &max_conf = (*all_max_conf)[i];
    for (int p = 0; p < num_priors; ++p) {
      int max_id = 0;
      float max_val = -1.0;
      // 查找最大值
      for (int j = 0; j < len; ++j) {
        if (class_major) {
          if (conf_data[p + (start + j) * num_priors] > max_val) {
            max_id = j;
            max_val = conf_data[p + (start + j) * num_priors];
          }
        } else {
          if (conf_data[p * num_classes + start + j] > max_val) {
            max_id = j;
            max_val = conf_data[p * num_classes + start + j];
          }
        }
      }
      max_conf.push_back(std::make_pair(max_id, max_val));
    }
    conf_data += num_priors * num_classes;
  }
}

template void GetMaxConfScores(const float *conf_data, const int num,
                     const int num_priors, const int num_classes,
                     const int start, const int len, const bool class_major,
                     vector<vector<pair<int, float> > > *all_max_conf);
template void GetMaxConfScores(const double *conf_data, const int num,
                    const int num_priors, const int num_classes,
                    const int start, const int len, const bool class_major,
                    vector<vector<pair<int, float> > > *all_max_conf);

template <typename Dtype>
void GetConfidenceScores_yolo(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int index,
                         vector<vector<float> >  *conf_preds) {
  CHECK_LT(index, num_classes);
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<float> &label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_priors; ++p) {
      int id = p * (num_classes + 1);
      label_scores.push_back(conf_data[id] * conf_data[id+index]);
    }
    conf_data += num_priors * (num_classes+1);
  }
}

// Explicit initialization.
template void GetConfidenceScores_yolo(const float *conf_data, const int num,
                                  const int num_priors, const int num_classes,
                                  const int index,
                                  vector<vector<float> >  *conf_preds);
template void GetConfidenceScores_yolo(const double *conf_data, const int num,
                                  const int num_priors, const int num_classes,
                                  const int index,
                                  vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int index,
                         vector<vector<float> >  *conf_preds) {
  CHECK_LT(index, num_classes);
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    vector<float> &label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_priors; ++p) {
      label_scores.push_back(conf_data[p * num_classes + index]);
    }
    conf_data += num_priors * num_classes;
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float *conf_data, const int num,
                                  const int num_priors, const int num_classes,
                                  const int index,
                                  vector<vector<float> >  *conf_preds);
template void GetConfidenceScores(const double *conf_data, const int num,
                                  const int num_priors, const int num_classes,
                                  const int index,
                                  vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major, const int index,
                         vector<vector<float> >  *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    // 获取保存地址
    vector<float> &label_scores = (*conf_preds)[i];
    // 直接复制
    if (class_major) {
      label_scores.assign(conf_data + num_priors * index,
                          conf_data + num_priors * (index + 1));
    } else {
      for (int p = 0; p < num_priors; ++p) {
        int start_idx = p * num_classes;
        label_scores.push_back(conf_data[start_idx + index]);
      }
    }
    // 跳到下一个样本
    conf_data += num_priors * num_classes;
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float *conf_data, const int num,
                                 const int num_priors, const int num_classes,
                                 const bool class_major, const int index,
                                 vector<vector<float> >  *conf_preds);
template void GetConfidenceScores(const double *conf_data, const int num,
                                  const int num_priors, const int num_classes,
                                  const bool class_major, const int index,
                                  vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetMaxConfidenceScores_yolo(const Dtype *conf_data, const int num,
                            const int num_priors,
                            const int num_classes,
                            const int background_label_id,
                            const int start, const int len,
                            vector<vector<float> > *all_max_scores) {
  all_max_scores->clear();
  CHECK_GE(start, 0) << "start should be greater than 0.";
  CHECK_GT(len, 0) << "len should be greater than 0.";
  // 遍历所有样本
  for (int i = 0; i < num; ++i) {
    // 评分信息
    vector<float> max_scores;
    // 遍历所有的box
    for (int p = 0; p < num_priors; ++p) {
      int start_idx = p * (num_classes + 1);
      Dtype maxval = -FLT_MAX;
      for (int c = 0; c < len; ++c) {
        maxval = std::max<Dtype>(conf_data[start_idx + start + c], maxval);
      }
      max_scores.push_back(maxval*conf_data[start_idx]);
    }
    // 计算完一个样本后，指向下一个样本
    conf_data += num_priors * (num_classes + 1);
    all_max_scores->push_back(max_scores);
  }
}

// Explicit initialization.
template void GetMaxConfidenceScores_yolo(const float *conf_data, const int num,
                                    const int num_priors,
                                    const int num_classes,
                                    const int background_label_id,
                                    const int start, const int len,
                                    vector<vector<float> > *all_max_scores);
template void GetMaxConfidenceScores_yolo(const double *conf_data, const int num,
                                    const int num_priors,
                                    const int num_classes,
                                    const int background_label_id,
                                    const int start, const int len,
                                    vector<vector<float> > *all_max_scores);
template <typename Dtype>
void GetMaxConfidenceScores(const Dtype *conf_data, const int num,
                            const int num_priors,
                            const int num_classes,
                            const int background_label_id,
                            const ConfLossType loss_type,
                            const int start, const int len,
                            vector<vector<float> > *all_max_scores) {
  all_max_scores->clear();
  CHECK_GE(start, 0) << "start should be greater than 0.";
  CHECK_GT(len, 0) << "len should be greater than 0.";
  // 遍历所有样本
  for (int i = 0; i < num; ++i) {
    // 评分信息
    vector<float> max_scores;
    // 遍历所有的box
    for (int p = 0; p < num_priors; ++p) {
      int start_idx = p * num_classes;
      Dtype maxval = -FLT_MAX;
      Dtype maxval_pos = -FLT_MAX;
      for (int c = 0; c < len; ++c) {
        maxval = std::max<Dtype>(conf_data[start_idx + start + c], maxval);
        if (c != background_label_id) {
          // 查找正例的最大值
          maxval_pos = std::max<Dtype>(conf_data[start_idx + start + c], maxval_pos);
        }
      }
      // 如果是softmax损失
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        // 计算softmax结果
        Dtype sum = 0.;
        // 计算sigma
        for (int c = 0; c < len; ++c) {
          sum += std::exp(conf_data[start_idx + start + c] - maxval);
        }
        // 计算最大正例的概率
        maxval_pos = std::exp(maxval_pos - maxval) / sum;
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        maxval_pos = 1. / (1. + exp(-maxval_pos));
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      max_scores.push_back(maxval_pos);
    }
    // 计算完一个样本后，指向下一个样本
    conf_data += num_priors * num_classes;
    all_max_scores->push_back(max_scores);
  }
}

// Explicit initialization.
template void GetMaxConfidenceScores(const float *conf_data, const int num,
                                    const int num_priors,
                                    const int num_classes,
                                    const int background_label_id,
                                    const ConfLossType loss_type,
                                    const int start, const int len,
                                    vector<vector<float> > *all_max_scores);
template void GetMaxConfidenceScores(const double *conf_data, const int num,
                                    const int num_priors,
                                    const int num_classes,
                                    const int background_label_id,
                                    const ConfLossType loss_type,
                                    const int start, const int len,
                                    vector<vector<float> > *all_max_scores);

template <typename Dtype>
void GetPriorBBoxes(const Dtype *prior_data, const int num_priors,
                    vector<NormalizedBBox> *prior_bboxes,
                    vector<vector<float> > *prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    prior_bboxes->push_back(bbox);
  }
  // prior_variances
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

// Explicit initialization.
template void GetPriorBBoxes(const float *prior_data, const int num_priors,
                             vector<NormalizedBBox> *prior_bboxes,
                             vector<vector<float> > *prior_variances);
template void GetPriorBBoxes(const double *prior_data, const int num_priors,
                             vector<NormalizedBBox> *prior_bboxes,
                             vector<vector<float> > *prior_variances);

template <typename Dtype>
void GetDetectionResults(
    const Dtype *det_data, const int num_det, const int background_label_id,
    map<int, map<int, vector<NormalizedBBox> > > *all_detections) {
  all_detections->clear();
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int item_id = det_data[start_idx];
    if (item_id < 0) continue;
    int label = det_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the detection results.";
    NormalizedBBox bbox;
    bbox.set_score(det_data[start_idx + 2]);
    bbox.set_xmin(det_data[start_idx + 3]);
    bbox.set_ymin(det_data[start_idx + 4]);
    bbox.set_xmax(det_data[start_idx + 5]);
    bbox.set_ymax(det_data[start_idx + 6]);
    ClipBBox(bbox, &bbox);
    float bbox_size = BBoxSize(bbox);
    if (bbox_size <= 0.001) continue;
    bbox.set_size(bbox_size);
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void
GetDetectionResults(const float *det_data, const int num_det,
                    const int background_label_id,
                    map<int, map<int, vector<NormalizedBBox> > > *all_detections);
template void
GetDetectionResults(const double *det_data, const int num_det,
                    const int background_label_id,
                    map<int, map<int, vector<NormalizedBBox> > > *all_detections);

void GetTopKScoreIndex(const vector<float> &scores, const vector<int> &indices,
                       const int top_k,
                       vector<pair<float, int> > *score_index_vec) {
  CHECK_EQ(scores.size(), indices.size());

  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    score_index_vec->push_back(std::make_pair(scores[i], indices[i]));
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

void ApplyNMS(const vector<NormalizedBBox> &bboxes, const vector<float> &scores,
              const float threshold, const int top_k, const bool reuse_overlaps,
              map<int, map<int, float> > *overlaps, vector<int> *indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  vector<int> idx(boost::counting_iterator<int>(0),
                  boost::counting_iterator<int>(scores.size()));
  vector<pair<float, int> > score_index_vec;
  GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

  // Do nms.
  indices->clear();
  while (score_index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = score_index_vec.front().second;
    const NormalizedBBox &best_bbox = bboxes[best_idx];
    if (BBoxSize(best_bbox) < 1e-5) {
      // Erase small box.
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
    for (vector<pair<float, int> >::iterator it = score_index_vec.begin();
         it != score_index_vec.end();) {
      int cur_idx = it->second;
      const NormalizedBBox &cur_bbox = bboxes[cur_idx];
      if (BBoxSize(cur_bbox) < 1e-5) {
        // Erase small box.
        it = score_index_vec.erase(it);
        continue;
      }
      float cur_overlap = 0.;
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
          cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
          // Store the overlap for future use.
          (*overlaps)[best_idx][cur_idx] = cur_overlap;
        }
      } else {
        cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
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

void ApplyNMS(const bool *overlapped, const int num, vector<int> *indices) {
  vector<int> index_vec(boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(num));
  // Do nms.
  indices->clear();
  while (index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = index_vec.front();
    indices->push_back(best_idx);
    // Erase the best box.
    index_vec.erase(index_vec.begin());

    for (vector<int>::iterator it = index_vec.begin(); it != index_vec.end();) {
      int cur_idx = *it;

      // Remove it if necessary
      if (overlapped[best_idx * num + cur_idx]) {
        it = index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void GetMaxScoreIndex(const vector<float> &scores, const float threshold,
                      const int top_k,
                      vector<pair<float, int> > *score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  if (score_index_vec->size() == 0) return;
  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
  // LOG(INFO) << "the last score: " << score_index_vec->back().first << "   propose: " << score_index_vec->size();
}

void ApplyNMSFast(const vector<NormalizedBBox> &bboxes,
                  const vector<float> &scores, const float score_threshold,
                  const float nms_threshold, const int top_k,
                  vector<int> *indices) {
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);
  // Do nms.
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
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


void ApplyNMSFastUnit(vector<NormalizedBBox> *bboxes,
                    const vector<float> &scores,
                    const float score_threshold,
                    const float nms_threshold,
                    const int top_k,
                    vector<int> *indices) {
  CHECK_EQ(bboxes->size(), scores.size())
      << "bboxes and scores have different size.";
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);
  vector<pair<float, int> > score_index_vec_temp;
  score_index_vec_temp = score_index_vec;
  // LOG(INFO) << "====================NMS======================";
  // LOG(INFO) << "[After THRE] -> " << score_index_vec.size();
  float box_voting_threshold = 0.75;
  if (score_index_vec.size() == 0) return;
  // Do nms.
  indices->clear();
  map<int, vector<NormalizedBBox> > box_voting;
  map<int, vector<float> > box_voting_score;
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    // LOG(INFO) << idx;
    // float tmp_score = score_index_vec.front().first;
    bool keep = true;
    // 遍历已有的所有列表
    for (int k = 0; k < indices->size(); ++k) {
      const int kept_idx = (*indices)[k];
      float overlap = JaccardOverlap((*bboxes)[idx], (*bboxes)[kept_idx]);
      if (overlap >= box_voting_threshold){
        box_voting[kept_idx].push_back((*bboxes)[idx]);
        box_voting_score[kept_idx].push_back(scores[idx]);
      }
      if (overlap >= nms_threshold){
        keep = false;
      }

      // if (keep) {
      //   const int kept_idx = (*indices)[k];
      //   float overlap = JaccardOverlap((*bboxes)[idx], (*bboxes)[kept_idx]);
      //   keep = overlap <= nms_threshold;
      // } else {
      //   break;
      // }
    }
    if (keep) {
      indices->push_back(idx);
      box_voting[idx].push_back((*bboxes)[idx]);
      box_voting_score[idx].push_back(scores[idx]);
      // LOG(INFO) << "before index: " << idx << "  score: " << scores[idx] << "size: " << BBoxSize((*bboxes)[idx]);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
  for (map<int, vector<NormalizedBBox> >::iterator it = box_voting.begin();
       it != box_voting.end(); ++it) {
    int kept_id = it->first;
    vector<NormalizedBBox> &voting_list = it->second;
    if (voting_list.size()==1) continue;
    // LOG(INFO) << "number of box_voting" << voting_list.size();
    vector<float> &voting_score = box_voting_score[kept_id];
    float all_score = 0.0;
    for (int i=0;i<voting_score.size();++i){
      all_score = all_score + voting_score[i];
    }
    // LOG(INFO) << "before index: " << kept_id << "  score: " << voting_score[0] << "size: " << BBoxSize((*bboxes)[kept_id]) << voting_list[0].xmin() << voting_list[0].ymin() << voting_list[0].xmax() << voting_list[0].ymax();
    // LOG(INFO) << "before index: " << kept_id << "  score: " << voting_score[0] << "size: " << BBoxSize((*bboxes)[kept_id]) << (*bboxes)[kept_id].xmin() << (*bboxes)[kept_id].ymin() << (*bboxes)[kept_id].xmax() << (*bboxes)[kept_id].ymax();
    float xmin = 0.0,xmax = 0.0 ,ymin = 0.0 ,ymax= 0.0;
    for (int i=0;i < voting_list.size();++i){
      float scale = voting_score[i] / all_score;
      xmin = xmin + voting_list[i].xmin() * scale;
      ymin = ymin + voting_list[i].ymin() * scale;
      xmax = xmax + voting_list[i].xmax() * scale;
      ymax = ymax + voting_list[i].ymax() * scale;
      if (voting_list.size() == 2){
        // LOG(INFO) << i << scale << voting_list[i].xmin() << voting_list[i].ymin() << voting_list[i].xmax() << voting_list[i].ymax();
      }
    }
    (*bboxes)[kept_id].set_xmin(xmin);
    (*bboxes)[kept_id].set_xmax(xmax);
    (*bboxes)[kept_id].set_ymin(ymin);
    (*bboxes)[kept_id].set_ymax(ymax);
    // LOG(INFO) << "after index: " << kept_id << "  score: " << voting_score[0] << "size: " << BBoxSize((*bboxes)[kept_id]) << (*bboxes)[kept_id].xmin() << (*bboxes)[kept_id].ymin() << (*bboxes)[kept_id].xmax() << (*bboxes)[kept_id].ymax();

  }
  // LOG(INFO) << "after nms" << indices->size();
}

void AddParts(const vector<NormalizedBBox> &bboxes,
              const vector<float> &body_conf_scores,
              const vector<int> &body_indices,
              int *num_det, map<int, vector<NormalizedBBox> > *parts){
  CHECK_EQ(bboxes.size(), body_conf_scores.size());
  parts->clear();
  for (int i = 0; i < body_indices.size(); ++i) {
    int prior_id = body_indices[i];
    NormalizedBBox tbbox;
    tbbox.CopyFrom(bboxes[prior_id]);
    tbbox.set_score(body_conf_scores[prior_id]);
    ClipBBox(tbbox,&tbbox);
    (*parts)[1].push_back(tbbox);
    (*num_det)++;
  }
}

template <typename Dtype>
void Softmax(const Dtype *input, int n, Dtype *output) {
    int i;
    Dtype sum = 0;
    Dtype largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        Dtype e = exp(input[i] - largest);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

template void Softmax(const float *input, int n, float *output);
template void Softmax(const double *input, int n, double *output);


template <typename Dtype>
NormalizedBBox get_NormalizedBBoxbyLoc(const Dtype *loc_data,
                  const vector<Dtype> prior_width,
                  const vector<Dtype> prior_height,
                  int n, int index, int col, int row, int w, int h,
                  const CodeLocType code_type) {
  NormalizedBBox bbox;
  if (code_type == McBoxLossParameter_CodeLocType_YOLO){
    CHECK_EQ(prior_width.size(), prior_height.size());
    CHECK_GT(prior_width.size(), n);
    Dtype centor_x, centor_y;
    Dtype width, height;
    centor_x = (col + logistic(loc_data[index])) / w;
    centor_y = (row + logistic(loc_data[index+1])) / h;
    width = exp(loc_data[index+2]) * prior_width[n];
    height = exp(loc_data[index+3]) * prior_height[n];
    bbox.set_xmin(centor_x);
    bbox.set_ymin(centor_y);
    bbox.set_xmax(width);
    bbox.set_ymax(height);
  }else{
    Dtype prior_center_x = (Dtype(col) + 0.5) / w;
    Dtype prior_center_y = (row + 0.5) / h;
    // LOG(INFO) << "center_x: " << prior_center_x;
    // LOG(INFO) << "center_y: " << prior_center_y;
    Dtype centor_x = 0.1 * loc_data[index] * prior_width[n] + prior_center_x;
    Dtype centor_y = 0.1 * loc_data[index+1] * prior_height[n] + prior_center_y;
    Dtype width = exp(0.2 * loc_data[index+2]) * prior_width[n];
    Dtype height = exp(0.2 * loc_data[index+3]) * prior_height[n];
    bbox.set_xmin(centor_x);
    bbox.set_ymin(centor_y);
    bbox.set_xmax(width);
    bbox.set_ymax(height);
  }
  return bbox;
}

template NormalizedBBox get_NormalizedBBoxbyLoc(const float *conf_data,
                  const vector<float> prior_width,
                  const vector<float> prior_height,
                  int n, int index, int i, int j, int w, int h,const CodeLocType code_type);
template NormalizedBBox get_NormalizedBBoxbyLoc(const double *conf_data,
                  const vector<double> prior_width,
                  const vector<double> prior_height,
                  int n, int index, int i, int j, int w, int h,const CodeLocType code_type);

void CenterToCorner(const NormalizedBBox& input, NormalizedBBox* output) {
  float xmin = input.xmin() - input.xmax() / 2.;
  float xmax = input.xmin() + input.xmax() / 2.;
  float ymin = input.ymin() - input.ymax() / 2.;
  float ymax = input.ymin() + input.ymax() / 2.;
  output->set_xmin(xmin);
  output->set_ymin(ymin);
  output->set_xmax(xmax);
  output->set_ymax(ymax);
}

void CornerToCenter(const NormalizedBBox& input, NormalizedBBox* output) {
  float center_x = (input.xmin() + input.xmax())/ 2.;
  float center_y = (input.ymin() + input.ymax())/ 2.;
  float width = input.xmax() - input.xmin();
  float height = input.ymax() - input.ymin();
  output->set_xmin(center_x);
  output->set_ymin(center_y);
  output->set_xmax(width);
  output->set_ymax(height);
}


template <typename Dtype>
void Backward_mcbox(const NormalizedBBox& anchor,
                    const Dtype* loc_pred, int idx,
                    vector<Dtype>& prior_width, vector<Dtype>& prior_height,
                    int n, int col, int row, int w, int h,
                    Dtype diff_scale, Dtype* mutable_loc_diff,const CodeLocType code_type) {
  if (code_type == McBoxLossParameter_CodeLocType_YOLO){
    Dtype center_x_gt = anchor.xmin() * w - col;
    Dtype center_y_gt = anchor.ymin() * h - row;
    Dtype w_gt = log(anchor.xmax() / prior_width[n]);
    Dtype h_gt = log(anchor.ymax() / prior_height[n]);
    Dtype act_x = logistic(loc_pred[idx]);
    Dtype act_y = logistic(loc_pred[idx + 1]);
    Dtype pred_w = loc_pred[idx + 2];
    Dtype pred_h = loc_pred[idx + 3];
    mutable_loc_diff[idx] =
      diff_scale * (act_x - center_x_gt) * logistic_gradient(act_x);
    mutable_loc_diff[idx + 1] =
      diff_scale * (act_y - center_y_gt) * logistic_gradient(act_y);
    mutable_loc_diff[idx + 2] =
      diff_scale * (pred_w - w_gt);
    mutable_loc_diff[idx + 3] =
      diff_scale * (pred_h - h_gt);
  }else{
    Dtype prior_center_x = (col + 0.5) / w;
    Dtype prior_center_y = (row + 0.5) / h;
    Dtype encode_x = (anchor.xmin() - prior_center_x) / prior_width[n] / 0.1;
    Dtype encode_y = (anchor.ymin() - prior_center_y) / prior_height[n] / 0.1;
    Dtype encode_w = log(anchor.xmax()/prior_width[n]) / 0.2;
    Dtype encode_h = log(anchor.ymax()/prior_height[n]) / 0.2;
    mutable_loc_diff[idx] = diff_scale * (loc_pred[idx] - encode_x);
    mutable_loc_diff[idx+1] = diff_scale * (loc_pred[idx+1] - encode_y);
    mutable_loc_diff[idx+2] = diff_scale * (loc_pred[idx+2] - encode_w);
    mutable_loc_diff[idx+3] = diff_scale * (loc_pred[idx+3] - encode_h);
  }
    if (diff_scale > 2){
      LOG(INFO) << "x:" << mutable_loc_diff[idx];
      LOG(INFO) << "y:" << mutable_loc_diff[idx + 1];
      LOG(INFO) << "w:" << mutable_loc_diff[idx + 2];
      LOG(INFO) << "h:" << mutable_loc_diff[idx + 3];
    }
}

template void Backward_mcbox(const NormalizedBBox& anchor,
                    const float* loc_pred, int idx,
                    vector<float>& prior_width, vector<float>& prior_height,
                    int n, int col, int row, int w, int h,
                    float diff_scale, float* mutable_loc_diff,const CodeLocType code_type);
template void Backward_mcbox(const NormalizedBBox& anchor,
                    const double* loc_pred, int idx,
                    vector<double>& prior_width, vector<double>& prior_height,
                    int n, int col, int row, int w, int h,
                    double diff_scale, double* mutable_loc_diff,const CodeLocType code_type);

void CumSum(const vector<pair<float, int> > &pairs, vector<int> *cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int> > sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}

void ComputeAP(const vector<pair<float, int> > &tp, const int num_pos,
               const vector<pair<float, int> > &fp, const string ap_version,
               vector<float> *prec, vector<float> *rec, float *ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_EQ(tp[i].second, 1 - fp[i].second);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j - 1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}

int get_bbox_level(NormalizedBBox &box, map<int, float> &size_thre) {
  float size = BBoxSize(box);
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

void get_leveld_gtboxes(map<int, float> &size_thre,
                        map<int, map<int, vector<NormalizedBBox> > > &all_gtboxes,
                        vector<map<int, map<int, vector<NormalizedBBox> > > > *leveld_gtboxes) {
    for (map<int, map<int, vector<NormalizedBBox> > >::iterator it = all_gtboxes.begin();
         it != all_gtboxes.end(); ++it) {
      int item = it->first;
      for (map<int, vector<NormalizedBBox> >::iterator iit = it->second.begin();
          iit != it->second.end(); ++iit) {
          int label = iit->first;
          vector<NormalizedBBox>& gtboxes = iit->second;
          if (gtboxes.size() == 0) continue;
          for (int j = 0; j < gtboxes.size(); ++j) {
            int level = get_bbox_level(gtboxes[j], size_thre);
            if (level < 0) continue;
            // push -> (*leveld_gtboxes)[level][item][label].push_back(gtboxes[j])
            for (int l = 0; l < level + 1; ++l) {
              map<int, map<int, vector<NormalizedBBox> > > &l_gtboxes = (*leveld_gtboxes)[l];
              l_gtboxes[item][label].push_back(gtboxes[j]);
            }
          }
      }
  }
}

void leveld_eval_detections(map<int, map<int, vector<NormalizedBBox> > >& l_gtboxes,
                            map<int, map<int, vector<NormalizedBBox> > >& all_detections,
                            const float size_thre, const float iou_threshold,
                            const int num_classes, const int background_label_id,
                            const int level, const int diff, vector<vector<float> >* l_res) {
  l_res->clear();
  map<int, int> num_gts;
  // 遍历l_gtboxes
  for (map<int, map<int, vector<NormalizedBBox> > >::iterator it = l_gtboxes.begin();
       it != l_gtboxes.end(); ++it) {
    for (map<int, vector<NormalizedBBox> >::iterator iit = it->second.begin();
        iit != it->second.end(); ++iit) {
      int label = iit->first;
      int num = iit->second.size();
      if (num_gts.find(label) == num_gts.end()) {
        num_gts[label] = num;
      } else {
        num_gts[label] += num;
      }
    }
  }
  // output gt nums
  for (int c = 1; c < num_classes; ++c) {
    vector<float> res;
    res.push_back(diff);
    res.push_back(level);
    res.push_back(-1);
    res.push_back(c);
    if (num_gts.find(c) == num_gts.end()) {
      res.push_back(0);
    } else {
      res.push_back(num_gts[c]);
    }
    if (size_thre == 0 && iou_threshold == 0.5){

      //LOG(INFO) << "GT number: " << res.back();
    }
    res.push_back(-1);
    res.push_back(-1);
    l_res->push_back(res);
  }
  // output detboxes
  for (map<int, map<int, vector<NormalizedBBox> > >::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int image_id = it->first;
    map<int, vector<NormalizedBBox> > &detections = it->second;
    if (l_gtboxes.find(image_id) == l_gtboxes.end()) {
      for (map<int, vector<NormalizedBBox> >::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        vector<NormalizedBBox> &fp_bboxes = iit->second;
        if(fp_bboxes.size() == 0) continue;
        for (int i = 0; i < fp_bboxes.size(); ++i) {
          if (BBoxSize(fp_bboxes[i]) > size_thre) {
            // add -> fp
            vector<float> res;
            res.push_back(diff);
            res.push_back(level);
            res.push_back(image_id);
            res.push_back(label);
            res.push_back(fp_bboxes[i].score());
            res.push_back(0);
            res.push_back(1);
            l_res->push_back(res);
            if (size_thre == 0 && iou_threshold == 0.5){
              // LOG(INFO) << "FP output score: " << fp_bboxes[i].score() << " size: " << BBoxSize(fp_bboxes[i]) << "no gt";
            }
          }
        }
      }
    } else {
      map<int, vector<NormalizedBBox> > &label_bboxes = l_gtboxes.find(image_id)->second;
      for (map<int, vector<NormalizedBBox> >::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        vector<NormalizedBBox> &bboxes = iit->second;
        if (bboxes.size() == 0) continue;
        // 未查找到该类,全部是FP
        if (label_bboxes.find(label) == label_bboxes.end()) {
          if (bboxes.size() == 0) continue;
          for (int i = 0; i < bboxes.size(); ++i) {
            // FP
            if (BBoxSize(bboxes[i]) > size_thre) {
              // add -> fp
              vector<float> res;
              res.push_back(diff);
              res.push_back(level);
              res.push_back(image_id);
              res.push_back(label);
              res.push_back(bboxes[i].score());
              res.push_back(0);
              res.push_back(1);
              l_res->push_back(res);
              if (size_thre == 0 && iou_threshold == 0.5){
               // LOG(INFO) << "FP output score: " << bboxes[i].score() << " size: " << BBoxSize(bboxes[i]) << "not match part";
              }
            }
          }
        } else {
          vector<NormalizedBBox> &gt_bboxes = label_bboxes.find(label)->second;
          vector<bool> visited(gt_bboxes.size(), false);
          std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
          for (int i = 0; i < bboxes.size(); ++i) {
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j], true);
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
            if (overlap_max >= iou_threshold) {
              // if (size_thre == 0 && iou_threshold == 0.5){
              //   LOG(INFO) << "====================OUTPUT======================";
              // }
              if (!visited[jmax]) {
                // TP
                vector<float> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(image_id);
                res.push_back(label);
                res.push_back(bboxes[i].score());
                res.push_back(1);
                res.push_back(0);
                l_res->push_back(res);
                visited[jmax] = true;
                if (size_thre == 0 && iou_threshold == 0.5){
                  // LOG(INFO) << "TP output score: " << std::fixed <<  std::right << std::setw(8) << bboxes[i].score() << " size: "  << std::fixed<< std::right << std::setw(8) << BBoxSize(bboxes[i]) << " gtsize: " << std::fixed << std::right << std::setw(8)<< BBoxSize(gt_bboxes[jmax]) << " maxiou: "<< std::fixed << std::right << std::setw(8)<< overlap_max << " gtindex: " << jmax+1;
                }
              } else {
                // FD: Duplicate Detecion -> FP
                vector<float> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(image_id);
                res.push_back(label);
                res.push_back(bboxes[i].score());
                res.push_back(0);
                res.push_back(1);
                l_res->push_back(res);
                if (size_thre == 0 && iou_threshold == 0.5){
                  // LOG(INFO) << "DP output score: " << std::fixed <<  std::right << std::setw(8) << bboxes[i].score() << " size: "  << std::fixed<< std::right << std::setw(8) << BBoxSize(bboxes[i]) << " gtsize: " << std::fixed << std::right << std::setw(8)<< BBoxSize(gt_bboxes[jmax]) << " maxiou: "<< std::fixed << std::right << std::setw(8)<< overlap_max << " gtindex: " << jmax+1;
                }
              }
            } else {
              // FP
              if (BBoxSize(bboxes[i]) > size_thre) {
                vector<float> res;
                res.push_back(diff);
                res.push_back(level);
                res.push_back(image_id);
                res.push_back(label);
                res.push_back(bboxes[i].score());
                res.push_back(0);
                res.push_back(1);
                l_res->push_back(res);
                if (size_thre == 0 && iou_threshold == 0.5){
                  // LOG(INFO) << "FP output score: " << std::fixed <<  std::right << std::setw(8) << bboxes[i].score() << " size: "  << std::fixed<< std::right << std::setw(8) << BBoxSize(bboxes[i]) << " gtsize: " << std::fixed << std::right << std::setw(8)<< BBoxSize(gt_bboxes[jmax]) << " maxiou: "<< std::fixed << std::right << std::setw(8)<< overlap_max << " gtindex: " << jmax+1;
                }
              }
            }
          }
        }
      }
    }
  }
}

#ifdef USE_OPENCV
cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
  case 0:
    r = v;
    g = t;
    b = p;
    break;
  case 1:
    r = q;
    g = v;
    b = p;
    break;
  case 2:
    r = p;
    g = v;
    b = t;
    break;
  case 3:
    r = p;
    g = q;
    b = v;
    break;
  case 4:
    r = t;
    g = p;
    b = v;
    break;
  case 5:
    r = v;
    g = p;
    b = q;
    break;
  default:
    r = 1;
    g = 1;
    b = 1;
    break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}

// http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically
vector<cv::Scalar> GetColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h =
        std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate, 1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

void VisualizeBBox(const vector<cv::Mat> &images,
                   const vector<map<int, vector<NormalizedBBox> > > &all_dets,
                   const VisualizeParameter &visual_param) {
  // 时间信息
  static clock_t start_clock = clock();
  // 累计运行时间
  static clock_t total_time_start = start_clock;
  static long total_run_time = 0;
  static long total_frame = 0;

  const int num_img = images.size();
  if (num_img == 0) {
    return;
  }

  CHECK_EQ(num_img, all_dets.size());

  // 计算FPS
  float fps =
      num_img / (static_cast<double>(clock() - start_clock) / CLOCKS_PER_SEC);
  // 计算总运行时间
  total_run_time = clock() - total_time_start;
  // 计算总帧数
  total_frame += num_img;
  /*****************************************************************************************************/
  //打印运行信息
  float run_ftime_sec = static_cast<double>(total_run_time) / CLOCKS_PER_SEC;
  int run_ms = (static_cast<long>(run_ftime_sec * 1000)) % 1000;
  int run_s  = static_cast<int>(run_ftime_sec);
  int run_hour = run_s / 3600;
  run_s = run_s % 3600;
  int run_min = run_s / 60;
  run_s = run_s % 60;
  //run_ms/s/min/hour -> 运行时间信息
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
  // 检查绘图参数
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
  /*****************************************************************************************************/
  const int width = images[0].cols;
  const int height = images[0].rows;
  const int maxLen = (width > height) ? width:height;
  const float ratio = (float)display_max_size / maxLen;
  const int display_width = static_cast<int>(width * ratio);
  const int display_height = static_cast<int>(height * ratio);
  CHECK_GT(color_param.rgb_size(), 0);
  for (int i = 0; i < color_param.rgb_size(); ++i) {
    CHECK_EQ(color_param.rgb(i).val_size(), 3);
  }

  for (int i = 0; i < num_img; ++i) {
    cv::Mat display_image;
    cv::resize(images[i], display_image, cv::Size(display_width,display_height), cv::INTER_LINEAR);
    std::cout << "The input image size is   : " << width << "x" << height << '\n';
    std::cout << "The display image size is : " << display_width << "x" << display_height<< '\n';
    std::cout << std::endl;

    const map<int, vector<NormalizedBBox> > &dets = all_dets[i];
    if (dets.size() > 0) {
      // 遍历,打印输出
      for (map<int, vector<NormalizedBBox> >::const_iterator it = dets.begin();
           it != dets.end(); ++it) {
        int label = it->first;
        CHECK_GT(label,0);
        const vector<NormalizedBBox> &bboxes = it->second;
        if (bboxes.size() == 0) continue;
        const Color& rgb = color_param.rgb(label - 1);
        cv::Scalar line_color(rgb.val(2),rgb.val(1),rgb.val(0));
        // 定义该类的box
        for (int j = 0; j < bboxes.size(); ++j) {
          const NormalizedBBox &box = bboxes[j];
          if (box.score() < conf_threshold) continue;
          if (BBoxSize(box) < size_threshold) continue;
          std::cout << "A person is detected: " << std::setprecision(3) << box.score() << "\n";
          cv::Point top_left_pt(static_cast<int>(box.xmin() * display_width),
                                static_cast<int>(box.ymin() * display_height));
          cv::Point bottom_right_pt(static_cast<int>(box.xmax() * display_width),
                                static_cast<int>(box.ymax() * display_height));
          cv::rectangle(display_image, top_left_pt, bottom_right_pt, line_color, box_line_width);
        }
      }
    }
    cv::imshow("remo", display_image);
    if (cv::waitKey(1) == 27) {
      raise(SIGINT);
    }
  }
  start_clock = clock();
}

#endif // USE_OPENCV

} // namespace caffe
