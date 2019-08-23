#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

/**
 * 目前仅用于检测器。
 * 后期检测器工作会考虑迁移到Mask目录下。
 * 该头文件已停止更新，其相关方法可以参考mask/bbox_func.hpp
 * 建议直接阅读和使用mask/bbox_func.hpp中的方法。
 */

namespace caffe {

typedef EmitConstraint_EmitType EmitType;
typedef PriorBoxParameter_CodeType CodeType;
typedef MultiBoxLossParameter_MatchType MatchType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_ConfLossType ConfLossType;
typedef ReorgParameter_SampleType SampleType;
typedef McBoxLossParameter_CodeLocType CodeLocType;
typedef VisualizeposeParameter_DrawType DrawType;

typedef map<int, vector<NormalizedBBox> > LabelBBox;

template <typename Dtype>
inline Dtype logistic(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype logistic_gradient(Dtype x) {
  return x * (1. - x);
}

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in ascend order based on the score value.
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in descend order based on the score value.
bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);

// Generate unit bbox [0, 0, 1, 1]
NormalizedBBox UnitBBox();

// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);

// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

// Clip the NormalizedBBox such that the range for each corner is [0, 1].
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox);

// Scale the NormalizedBBox w.r.t. height and width.
void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox);

// Locate bbox in the coordinate system that src_bbox sits.
void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox);

// Project bbox onto the coordinate system defined by src_bbox.
bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox);

// Compute the jaccard (intersection over union IoU) overlap between two bboxes.
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true);

// Compute the coverage of bbox1 by bbox2.
float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Encode a bbox according to a prior bbox.
void EncodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool encode_variance_in_target, const NormalizedBBox& bbox,
    NormalizedBBox* encode_bbox);

// Check if a bbox meet emit constraint w.r.t. src_bbox.
bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
    const NormalizedBBox& bbox, const EmitConstraint& emit_constraint);

bool MeetEmitConstraint(const NormalizedBBox &src_bbox,
                        const NormalizedBBox &bbox,
                        const EmitConstraint &emit_constraint,
                        const int emitType);
// Decode a bbox according to a prior bbox.
void DecodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool variance_encoded_in_target, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox);

// Decode a set of bboxes according to a set of prior bboxes.
void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const vector<vector<NormalizedBBox> > &all_loc_preds,
                     const vector<NormalizedBBox> &prior_bboxes,
                     const vector<vector<float> > &prior_variances,
                     const int num, const CodeType code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<NormalizedBBox> > *all_decode_bboxes);

void AddParts(const vector<NormalizedBBox> &bboxes,
             const vector<float> &body_conf_scores,
             const vector<int> &body_indices,
             int *num_det, map<int, vector<NormalizedBBox> > *parts);

void MatchBBox(const vector<NormalizedBBox> &gt_bboxes,
              const vector<NormalizedBBox> &pred_bboxes,
              const MatchType match_type,
              const float overlap_threshold,
              vector<float> *match_overlaps,
              map<int, int> *match_indices,
              vector<int> *unmatch_indices);

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);
// Store ground truth bboxes of same label in a group.
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, map<int, vector<NormalizedBBox> > > *all_gt_bboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_priors, vector<vector<NormalizedBBox> >* loc_preds);

template <typename Dtype>
void GetMaxConfScores(const Dtype *conf_data, const int num,
                       const int num_priors, const int num_classes,
                       const int start, const int len,
                       vector<vector<pair<int, float> > > *all_max_conf);

template <typename Dtype>
void GetMaxConfScores(const Dtype *conf_data, const int num,
                    const int num_priors, const int num_classes,
                    const int start, const int len, const bool class_major,
                    vector<vector<pair<int, float> > > *all_max_conf);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int index,
                         vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetConfidenceScores_yolo(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const int index,
                         vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major, const int index,
                         vector<vector<float> >  *conf_preds);

template <typename Dtype>
void GetMaxConfidenceScores(const Dtype *conf_data, const int num,
                            const int num_priors,
                            const int num_classes,
                            const int background_label_id,
                            const ConfLossType loss_type,
                            const int start, const int len,
                            vector<vector<float> > *all_max_scores);

template <typename Dtype>
void GetMaxConfidenceScores_yolo(const Dtype *conf_data, const int num,
                            const int num_priors,
                            const int num_classes,
                            const int background_label_id,
                            const int start, const int len,
                            vector<vector<float> > *all_max_scores);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);

void GetTopKScoreIndex(const vector<float>& scores, const int top_k,
                         vector<pair<float, int> >* score_index_vec);

void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);

void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, const bool reuse_overlaps,
      map<int, map<int, float> >* overlaps, vector<int>* indices);

void ApplyNMS(const bool* overlapped, const int num, vector<int>* indices);

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const int top_k, vector<int>* indices);

void ApplyNMSFastUnit(vector<NormalizedBBox> *bboxes,
                    const vector<float> &scores,
                    const float score_threshold,
                    const float nms_threshold,
                    const int top_k,
                    vector<int> *indices);

void get_leveld_gtboxes(map<int, float> &size_thre,
                      map<int, map<int, vector<NormalizedBBox> > > &all_gtboxes,
                      vector<map<int, map<int, vector<NormalizedBBox> > > > *leveld_gtboxes);
int get_bbox_level(NormalizedBBox &box, map<int, float> &size_thre);

void leveld_eval_detections(map<int, map<int, vector<NormalizedBBox> > >& l_gtboxes,
                           map<int, map<int, vector<NormalizedBBox> > >& all_detections,
                           const float size_thre, const float iou_threshold,
                           const int num_classes, const int background_label_id,
                           const int level, const int diff, vector<vector<float> >* l_res);

template <typename Dtype>
void Softmax(const Dtype *input, int n, Dtype *output);

template <typename Dtype>
NormalizedBBox get_NormalizedBBoxbyLoc(const Dtype *loc_data,
                  const vector<Dtype> prior_width,
                  const vector<Dtype> prior_height,
                  int n, int index, int col, int row, int w, int h,const CodeLocType code_type);

void CenterToCorner(const NormalizedBBox& input, NormalizedBBox* output);
void CornerToCenter(const NormalizedBBox& input, NormalizedBBox* output);

template <typename Dtype>
void Backward_mcbox(const NormalizedBBox& anchor,
                    const Dtype* loc_pred, int idx,
                    vector<Dtype>& prior_width, vector<Dtype>& prior_height,
                    int n, int col, int row, int w, int h,
                    Dtype diff_scale, Dtype* mutable_loc_diff,const CodeLocType code_type);

// Compute cumsum of a set of pairs.
void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);

void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);


#ifndef CPU_ONLY  // GPU
template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
                                      const bool normalized = true);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
                                            const Dtype* bbox2);

template <typename Dtype>
void LogisticGPU(const int count, Dtype* data);

template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int out_num,
    const int channels, Dtype* prob);

template <typename Dtype>
void PermuteConfDataToBgClassGPU(const int nthreads,
                  const Dtype* conf_data, const int num_classes,
                  Dtype* bg_data, Dtype* class_data);

template <typename Dtype>
void UpdateConfByObjGPU(const int nthreads, const int num_classes,
      const Dtype* objectness, Dtype* conf_data);

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, Dtype* bbox_data);

template <typename Dtype>
void DecodeBBoxesByLocGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const int w, const int h, const int num_priors,
          Dtype* bbox_data);

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data);

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data);

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob);
#endif  // !CPU_ONLY

#ifdef USE_OPENCV
vector<cv::Scalar> GetColors(const int n);

void VisualizeBBox(const vector<cv::Mat> &images,
                   const vector<map<int, vector<NormalizedBBox> > > &all_part,
                   const VisualizeParameter &visual_param);

void VisualizeBBox(const vector<cv::Mat> &images,
                   const vector<vector<PersonBBox> > &all_person,
                   const vector<map<int, vector<NormalizedBBox> > > &all_part,
                   const VisualizeParameter &visual_param,
                   map<string, int> &name_labels,
                   const bool person_output);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
