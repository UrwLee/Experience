#ifndef CAFFE_MASK_BBOX_FUNC_HPP_
#define CAFFE_MASK_BBOX_FUNC_HPP_

#include <vector>
#include <string>

#include "caffe/tracker/bounding_box.hpp"

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Proto中的类型声明
typedef MultiBoxLossParameter_ConfLossType ConfLossType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_MatchType MatchType;
typedef PriorBoxParameter_CodeType CodeType;

/**
 * 带标签的BBox数据结构
 */
template <typename Dtype>
struct LabeledBBox {
  // bindex -> minibatch中的索引
  int bindex = -1; // index in minibatch
  // categeory id -> 类号
  int cid = -1; // class id
  // 实例id
  int pid = -1; // instance id
  // 是否是diff
  bool is_diff = false; // is_diff
  // 是否是crowd
  bool iscrowd = false; // is_crowd
  // 置信度
  Dtype score = 0; // confidence
  // box位置
  BoundingBox<Dtype> bbox; // localization
};

// Functions
/**
 * 排序
 * @param  bbox1 [bbox1]
 * @param  bbox2 [bbox2]
 * @return       [排序结果]
 */
template <typename Dtype>
bool BBoxAscend(const LabeledBBox<Dtype> &bbox1, const LabeledBBox<Dtype> &bbox2);

/**
 * 排序
 * @param  bbox1 [bbox1]
 * @param  bbox2 [bbox2]
 * @return       [排序结果]
 */
template <typename Dtype>
bool BBoxDescend(const LabeledBBox<Dtype> &bbox1, const LabeledBBox<Dtype> &bbox2);

/**
 * 排序
 * @param  vs1 [vector1]
 * @param  vs2 [vector2]
 * @return     [排序结果]
 */
template <typename Dtype>
bool VectorDescend(const std::vector<Dtype>& vs1, const std::vector<Dtype>& vs2);

/**
 * 排序
 * @param  vs1 [vector1]
 * @param  vs2 [vector2]
 * @return     [排序结果]
 */
template <typename Dtype>
bool VectorAescend(const std::vector<Dtype>& vs1, const std::vector<Dtype>& vs2);

/**
 * Pair排序
 * @param  pair1 [pair1]
 * @param  pair2 [pair2]
 * @return       [排序结果]
 */
template <typename T, typename Dtype>
bool PairAscend(const pair<Dtype, T> &pair1, const pair<Dtype, T> &pair2);

/**
 * Pair排序
 * @param  pair1 [pair1]
 * @param  pair2 [pair2]
 * @return       [排序结果]
 */
template <typename T, typename Dtype>
bool PairDescend(const pair<Dtype, T> &pair1, const pair<Dtype, T> &pair2);


/**
 * Box编码，模式：CORNER
 * @param prior_bbox                [编码的基准box]
 * @param prior_variance            [编码的增益系数]
 * @param encode_variance_in_target [默认为false]
 * @param bbox                      [待编码的box]
 * @param encode_bbox               [编码后的输出box]
 */
template <typename Dtype>
void EncodeBBox_Corner(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* encode_bbox);

/**
* Box编码，模式：CENTER
* @param prior_bbox                [编码的基准box]
* @param prior_variance            [编码的增益系数]
* @param encode_variance_in_target [默认为false]
* @param bbox                      [待编码的box]
* @param encode_bbox               [编码后的输出box]
*/
template <typename Dtype>
void EncodeBBox_Center(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                      const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                      BoundingBox<Dtype>* encode_bbox);

/**
* Box解码，模式：CORNER
* @param prior_bbox                [解码的基准box]
* @param prior_variance            [解码的增益系数]
* @param encode_variance_in_target [默认为false]
* @param bbox                      [待解码的box]
* @param decode_bbox               [解码后的输出box]
*/
template <typename Dtype>
void DecodeBBox_Corner(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                       const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                       BoundingBox<Dtype>* decode_bbox);

/**
* Box解码，模式：CENTER
* @param prior_bbox                [解码的基准box]
* @param prior_variance            [解码的增益系数]
* @param encode_variance_in_target [默认为false]
* @param bbox                      [待解码的box]
* @param decode_bbox               [解码后的输出box]
*/
template <typename Dtype>
void DecodeBBox_Center(const BoundingBox<Dtype>& prior_bbox, const vector<Dtype>& prior_variance,
                      const bool encode_variance_in_target, const BoundingBox<Dtype>& bbox,
                      BoundingBox<Dtype>* decode_bbox);

/**
 * LabeledBBox复制
 */
template <typename Dtype>
LabeledBBox<Dtype> LabeledBBox_Copy(const LabeledBBox<Dtype>& bbox);

/**
 * Boxes解码，模式：CORNER
 * @param prior_bboxes               [解码的参考boxes列表]
 * @param prior_variances            [解码的增益系数列表]
 * @param variance_encoded_in_target [默认是false]
 * @param bboxes                     [待解码的boxes列表]
 * @param decode_bboxes              [解码后的boxes列表]
 */
template <typename Dtype>
void DecodeBBoxes_Corner(const vector<LabeledBBox<Dtype> >& prior_bboxes,
                         const vector<vector<Dtype> >& prior_variances,
                         const bool variance_encoded_in_target,
                         const vector<LabeledBBox<Dtype> >& bboxes,
                         vector<LabeledBBox<Dtype> >* decode_bboxes);

/**
* Boxes解码，模式：CENTER
* @param prior_bboxes               [解码的参考boxes列表]
* @param prior_variances            [解码的增益系数列表]
* @param variance_encoded_in_target [默认是false]
* @param bboxes                     [待解码的boxes列表]
* @param decode_bboxes              [解码后的boxes列表]
*/
template <typename Dtype>
void DecodeBBoxes_Center(const vector<LabeledBBox<Dtype> >& prior_bboxes,
                        const vector<vector<Dtype> >& prior_variances,
                        const bool variance_encoded_in_target,
                        const vector<LabeledBBox<Dtype> >& bboxes,
                        vector<LabeledBBox<Dtype> >* decode_bboxes);

/**
 * 将整个batch的所有估计boxes进行解码
 * @param all_loc_preds              [整个batch的所有估计boxes列表]
 * @param prior_bboxes               [解码参考的boxes列表]
 * @param prior_variances            [解码参考的增益系数列表]
 * @param num                        [样本数]
 * @param code_type                  [0-CENTER, 1-CORNER]
 * @param variance_encoded_in_target [默认是fals]
 * @param all_decode_bboxes          [解码后的boxes列表]
 */
template <typename Dtype>
void DecodeBBoxes(const vector<vector<LabeledBBox<Dtype> > >& all_loc_preds,
                     const vector<LabeledBBox<Dtype> >& prior_bboxes,
                     const vector<vector<Dtype> >& prior_variances,
                     const int num, const int code_type,
                     const bool variance_encoded_in_target,
                     vector<vector<LabeledBBox<Dtype> > >* all_decode_bboxes);

template <typename Dtype>
void DecodeDenseBBoxes(const vector<vector<LabeledBBox<Dtype> > >& all_loc_preds,
                    const vector<LabeledBBox<Dtype> >& prior_bboxes,
                    const vector<vector<Dtype> >& prior_variances,
                    const int num, const int num_classes, const int code_type,
                    const bool variance_encoded_in_target,
                    vector<vector<vector<LabeledBBox<Dtype> > > >* all_decode_bboxes);

/**
 * Anchors和GTBoxes之间的匹配过程
 * 注意：这是单个样本之间的匹配过程
 * @param gt_bboxes          [该样本的所有GTBoxes]
 * @param pred_bboxes        [所有Prior-Boxes]
 * @param match_type         [匹配方式：0-耗尽型，1-(1v1)匹配]
 * @param overlap_threshold  [正例的IOU阈值下限]
 * @param negative_threshold [反例的IOU阈值上限]
 * @param match_overlaps     [每个Prior_boxes的最大匹配IOU]
 * @param match_indices      [完成匹配的Prior列表，<prior编号，　GT编号>]
 * @param neg_indices        [反例列表<prior编号>]
 */
template <typename Dtype>
void MatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                       const vector<LabeledBBox<Dtype> > &pred_bboxes,
                       const int match_type,
                       const Dtype overlap_threshold,
                       const Dtype negative_threshold,
                       vector<Dtype> *match_overlaps,
                       map<int, int> *match_indices,
                       vector<int> *neg_indices,
                       bool flag_noperson=false,
                       bool flag_withotherpositive=true);
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
                       bool flag_noperson=false,
                       bool flag_withotherpositive=true,
                       bool flag_matchallneg = false);

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
                       bool flag_noperson=false,
                       bool flag_withotherpositive=true,
                       const float sigma=0.2);

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
                       bool flag_noperson=false,
                       bool flag_withotherpositive=true,
                       const float cover_extracheck=0.7);
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
                       bool flag_noperson=false,
                       bool flag_withotherpositive=true,
                       const float margin_ratio=0.25,
                       bool flag_only_w=false,const float margin_ratio_h=0.25);
template <typename Dtype>
void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                             const vector<LabeledBBox<Dtype> > &pred_bboxes,
                             const Dtype overlap_threshold,
                             const Dtype negative_threshold,
                             vector<pair<int, int> >*match_indices,
                             vector<int> *neg_indices,
                             vector<bool> flags_of_anchor,
                             bool flag_noperson=false,
                             bool flag_forcematchallgt=false,
                             float area_check_max = 20);
template <typename Dtype>
void ExhaustMatchAnchorsAndGTs(const vector<LabeledBBox<Dtype> > &gt_bboxes,
                             const vector<LabeledBBox<Dtype> > &pred_bboxes,
                             const Dtype overlap_threshold,
                             const Dtype negative_threshold,
                             vector<pair<int, int> >*match_indices,
                             vector<int> *neg_indices,
                             bool flag_noperson=false,
                             bool flag_forcematchallgt=false,
                             float area_check_max = 20);
/**
 * 该方法: unused.
 * 移除匹配列表中所有匹配的GT标记为crowd的prior-boxes对象
 * @param gt_bboxes     [该样本中的所有GT-Boxes]
 * @param match_indices [第一次匹配列表]
 * @param new_indices   [移除后的匹配列表]
 */
template <typename Dtype>
void RemoveCrowds(const vector<LabeledBBox<Dtype> >& gt_bboxes,
                 const map<int, int>& match_indices,
                 map<int, int>* new_indices);

/**
 * 获取GT-Boxes
 * @param gt_data          [gt-data数据指针]
 * @param num_gt           [gt的总数量]
 * @param use_difficult_gt [是否载入is_diff标记为true的GT-Boxes]
 * @param size_threshold   [归一化尺寸小于该值则丢弃]
 * @param all_gt_bboxes    [GT-Boxes对象列表]
 * <int=bindex, [GT-Boxes in this image]>
 */
template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const Dtype size_threshold, map<int, vector<LabeledBBox<Dtype> > >* all_gt_bboxes,int ndim_label=9);
template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                 const vector<Dtype> size_threshold, map<int, vector<LabeledBBox<Dtype> > >* all_gt_bboxes,int ndim_label=9);

/**
 * 获取GT-Boxes
 * @param gt_data          [gt-data数据指针]
 * @param num_gt           [gt的总数量]
 * @param use_difficult_gt [是否载入is_diff标记为true的GT-Boxes]
 * @param all_gt_bboxes    [GT-Boxes对象列表]
 * <int=bindex, <int=cid, [GT-Boxes in this image]> >
 */
template <typename Dtype>
void GetGTBBoxes(const Dtype* gt_data, const int num_gt, const bool use_difficult_gt, const vector<int>& gt_labels,
                map<int, map<int, vector<LabeledBBox<Dtype> > > >* all_gt_bboxes,int ndim_label=9);

/**
 * 获取所有的位置估计结果
 * @param loc_data   [位置回归的数据指针]
 * @param num        [样本数]
 * @param num_priors [每个样本的prior-boxes数量]
 * @param loc_preds  [返回的位置估计Boxes的结果]
 */
template <typename Dtype>
void GetLocPreds(const Dtype *loc_data, const int num, const int num_priors,
                 vector<vector<LabeledBBox<Dtype> > > *loc_preds);

/**
 * 获得每个prior-boxes的最大置信度
 * num_classes = N
 * (p0,p1,...,pN-1) = softmax(a0,a1,...,aN-1)
 * max_conf = max(p0,p1,...,pN-1)
 * @param conf_data    [置信度数据指针]
 * @param num          [样本数]
 * @param num_priors   [每个样本的prior-boxes数量]
 * @param num_classes  [每个box应估计的类数，N=S+1，S为实际估计的物体类别数]
 * @param all_max_conf [输出结果]
 */
template <typename Dtype>
void GetMaxScores(const Dtype *conf_data, const int num,
                 const int num_priors, const int num_classes,
                 vector<vector<pair<int, Dtype> > > *all_max_conf);

/**
* 获得每个prior-boxes的最大置信度
* num_classes = N
* (p0,p1,...,pN-1) = softmax(a0,a1,...,aN-1)
* max_conf = max(p0,p1,...,pN-1)
* @param conf_data    [置信度数据指针]
* @param num          [样本数]
* @param num_priors   [每个样本的prior-boxes数量]
* @param num_classes  [每个box应估计的类数，N=S+1，S为实际估计的物体类别数]
* @param class_major  [0-(N,H,W,C) and 1-(N,C,H,W)]
* @param all_max_conf [输出结果]
*/
template <typename Dtype>
void GetMaxScores(const Dtype *conf_data, const int num,
                 const int num_priors, const int num_classes, const bool class_major,
                 vector<vector<pair<int, Dtype> > > *all_max_conf);

/**
 * 获取某个通道(某一类物体)的置信度信息
 * 每个prior_bboxes都会有一个针对于该类的置信度信息
 * @param conf_data   [置信度数据指针]
 * @param num         [样本数]
 * @param num_priors  [每个样本的prior-boxes数量]
 * @param num_classes [每个box应估计的类数，N=S+1，S为实际估计的物体类别数]
 * @param conf_preds  [返回置信度结果]
 */
 template <typename Dtype>
 void GetConfScores(const Dtype *conf_data, const int num,
                          const int num_priors, const int num_classes,
                          vector<vector<vector<Dtype> > > *conf_preds);

/**
* 获取某个通道(某一类物体)的置信度信息
* 每个prior_bboxes都会有一个针对于该类的置信度信息
* @param conf_data   [置信度数据指针]
* @param num         [样本数]
* @param num_priors  [每个样本的prior-boxes数量]
* @param num_classes [每个box应估计的类数，N=S+1，S为实际估计的物体类别数]
* @param class_major [0-(N,H,W,C) and 1-(N,C,H,W)]
* @param conf_preds  [返回置信度结果]
*/
template <typename Dtype>
void GetConfScores(const Dtype *conf_data, const int num,
                         const int num_priors, const int num_classes,
                         const bool class_major,
                         vector<vector<vector<Dtype> > > *conf_preds);

/**
* 获得每个prior-boxes的非背景最大置信度
* 注意：该最大置信度为非背景类的最大置信度，并不包含背景
* num_classes = N
* (p0,p1,...,pN-1) = softmax(a0,a1,...,aN-1)
* max_conf = max(p1,p2,...,pN-1)
* @param conf_data      [置信度数据指针]
* @param num            [样本数]
* @param num_priors     [每个样本的prior-boxes数量]
* @param num_classes    [每个box应估计的类数，N=S+1，S为实际估计的物体类别数]
* @param loss_type      [0-Softmax, 1-Logistic]
* @param all_max_scores [输出结果]
*/
template <typename Dtype>
void GetHDMScores(const Dtype *conf_data, const int num, const int num_priors,
                 const int num_classes, const int loss_type,
                 vector<vector<Dtype> > *all_max_scores);

/**
 * 获取prior_bboxes的所有信息
 * prior_bboxes　-> 坐标信息
 * prior_variances -> 编码的增益信息
 * @param prior_data      [prior_data数据指针]
 * @param num_priors      [prior_bboxes的数量]
 * @param prior_bboxes    [返回的prior_bboxes数据]
 * @param prior_variances [返回的prior_variances数据]
 */

template <typename Dtype>
void GetARMScores(const Dtype *conf_data, const int num, const int num_priors,
                 const int num_classes, const int loss_type,
                 vector<vector<Dtype> > *neg_scores);

/**
 * 获取prior_bboxes的所有信息
 * prior_bboxes　-> 坐标信息
 * prior_variances -> 编码的增益信息
 * @param prior_data      [prior_data数据指针]
 * @param num_priors      [prior_bboxes的数量]
 * @param prior_bboxes    [返回的prior_bboxes数据]
 * @param prior_variances [返回的prior_variances数据]
 */

template <typename Dtype>
void GetAnchorBBoxes(const Dtype *prior_data, const int num_priors,
                   vector<LabeledBBox<Dtype> > *prior_bboxes,
                   vector<vector<Dtype> > *prior_variances);

/**
 * 获取检测器的结果
 * @param det_data       [检测器的输出数据, (1,1,N,7)]
 * @param num_det        [检测目标的数量]
 * @param all_detections [输出结果]
 * <int=bindex, <int=cid, [Det-Boxes] > >
 */
template <typename Dtype>
void GetDetections(const Dtype *det_data, const int num_det,
   map<int, map<int, vector<LabeledBBox<Dtype> > > > *all_detections);

/**
 * 获取置信度最高的K个检测结果
 * 注意：针对单个样本
 * @param scores          [置信度列表]
 * @param indices         [索引列表]
 * @param top_k           [设定的最大数K]
 * @param score_index_vec [排名最高的<prior_id, score>列表返回]
 */
template <typename Dtype>
void GetTopKs(const vector<Dtype> &scores, const vector<int> &indices,
             const int top_k, vector<pair<Dtype, int> > *score_index_vec);

/**
 * NMS操作：　对某一类目标的检测结果输出执行NMS操作，去除同一目标的多重检测结果
 * 注意：　该方法不用
 * １．将所有boxes按照置信度降序排列
 * ２．如果置信度靠后与置信度靠前的某个box的IOU超过阈值，则表明是对同一物体的多次检测，删除之
 * @param bboxes         [所有检测的boxes]
 * @param scores         [对应的置信度信息]
 * @param threshold      [nms阈值]
 * @param top_k          [top-K]
 * @param reuse_overlaps [默认是false]
 * @param overlaps       [overlaps表]
 * @param indices        [结果返回]
 */
template <typename Dtype>
void NmsByBoxes(const vector<LabeledBBox<Dtype> > &bboxes, const vector<Dtype> &scores,
             const Dtype threshold, const int top_k, const bool reuse_overlaps,
             map<int, map<int, Dtype> > *overlaps, vector<int> *indices);

/**
 * NMS: 该方法不用
 */
void NmsByOverlappedGrid(const bool *overlapped, const int num, vector<int> *indices);

/**
 * 保留前K个置信度最高的结果
 * 注意：带有阈值，置信度超过阈值的才会保留
 */
template <typename Dtype>
void GetTopKsWithThreshold(const vector<Dtype> &scores, const Dtype threshold,
                           const int top_k,
                           vector<pair<Dtype, int> > *score_index_vec);

template <typename Dtype>
void NmsOriSoft(const vector<LabeledBBox<Dtype> > &bboxes,
                const vector<Dtype> &scores,
                const Dtype conf_threshold,
                const Dtype nms_threshold,
                const int top_k,
                vector<int> *indices);

// template void NmsOriSoft(const vector<LabeledBBox<float> > &bboxes,
//                          const vector<float> &scores,
//                          const float conf_threshold,
//                          const float nms_threshold,
//                          const int top_k,
//                          vector<int> *indices);
// template void NmsOriSoft(const vector<LabeledBBox<double> > &bboxes,
//                          const vector<double> &scores,
//                          const double conf_threshold,
//                          const double nms_threshold,
//                          const int top_k,
//                          vector<int> *indices);

template <typename Dtype>
void NmsOriSoftThree(const vector<LabeledBBox<Dtype> > &bboxes,
                     const vector<Dtype> &scores,
                     const Dtype conf_threshold,
                     const Dtype nms_threshold,
                     const int top_k,
                     vector<int> *indices);

// template void NmsOriSoftThree(const vector<LabeledBBox<float> > &bboxes,
//                               const vector<float> &scores,
//                               const float conf_threshold,
//                               const float nms_threshold,
//                               const int top_k,
//                               vector<int> *indices);
// template void NmsOriSoftThree(const vector<LabeledBBox<double> > &bboxes,
//                               const vector<double> &scores,
//                               const double conf_threshold,
//                               const double nms_threshold,
//                               const int top_k,
//                               vector<int> *indices);

template <typename Dtype>
void NmsOriSoftweight04(const vector<LabeledBBox<Dtype> > &bboxes,
                        const vector<Dtype> &scores,
                        const Dtype conf_threshold,
                        const Dtype nms_threshold,
                        const int top_k,
                        vector<int> *indices);

// template void NmsOriSoftweight04(const vector<LabeledBBox<float> > &bboxes,
//                                  const vector<float> &scores,
//                                  const float conf_threshold,
//                                  const float nms_threshold,
//                                  const int top_k,
//                                  vector<int> *indices);
// template void NmsOriSoftweight04(const vector<LabeledBBox<double> > &bboxes,
//                                  const vector<double> &scores,
//                                  const double conf_threshold,
//                                  const double nms_threshold,
//                                  const int top_k,
//                                  vector<int> *indices);

template <typename Dtype>
void NmsOriSoftweight04WithVoting(vector<LabeledBBox<Dtype> > &bboxes,
                                  const vector<Dtype> &scores,
                                  const Dtype conf_threshold,
                                  const Dtype nms_threshold,
                                  const int top_k,
                                  const Dtype voting_thre,
                                  vector<int> *indices);

template <typename Dtype>
void NmsOriSoftPower2WithVoting(vector<LabeledBBox<Dtype> > &bboxes,
                                const vector<Dtype> &scores,
                                const Dtype conf_threshold,
                                const Dtype nms_threshold,
                                const int top_k,
                                const Dtype voting_thre,
                                vector<int> *indices);

// template void NmsOriSoftPower2WithVoting(vector<LabeledBBox<float> > &bboxes,
//     const vector<float> &scores,
//     const float conf_threshold,
//     const float nms_threshold,
//     const int top_k,
//     const float voting_thre,
//     vector<int> *indices);
// template void NmsOriSoftPower2WithVoting(vector<LabeledBBox<double> > &bboxes,
//     const vector<double> &scores,
//     const double conf_threshold,
//     const double nms_threshold,
//     const int top_k,
//     const double voting_thre,
//     vector<int> *indices);


template <typename Dtype>
void NmsOri(const vector<LabeledBBox<Dtype> > &bboxes,
            const vector<Dtype> &scores,
            const Dtype conf_threshold,
            const Dtype nms_threshold,
            const int top_k,
            vector<int> *indices);

// template void NmsOri(const vector<LabeledBBox<float> > &bboxes,
//                      const vector<float> &scores,
//                      const float conf_threshold,
//                      const float nms_threshold,
//                      const int top_k,
//                      vector<int> *indices);
// template void NmsOri(const vector<LabeledBBox<double> > &bboxes,
//                      const vector<double> &scores,
//                      const double conf_threshold,
//                      const double nms_threshold,
//                      const int top_k,
//                      vector<int> *indices);

/**
 * 实际使用的NMS方法
 * NMS操作：　对某一类目标的检测结果输出执行NMS操作，去除同一目标的多重检测结果
 * 注意：　该方法不用
 * １．将所有boxes按照置信度降序排列
 * ２．如果置信度靠后与置信度靠前的某个box的IOU超过阈值，则表明是对同一物体的多次检测，删除之
 * @param bboxes         [所有检测的boxes]
 * @param scores         [对应的置信度信息]
 * @param conf_threshold [置信度阈值]
 * @param nms_threshold  [nms阈值]
 * @param top_k          [top-K]
 * @param indices        [保留的结果索引]
 */
template <typename Dtype>
void NmsFast(const vector<LabeledBBox<Dtype> > &bboxes,
            const vector<Dtype> &scores,
            const Dtype conf_threshold,
            const Dtype nms_threshold,
            const int top_k,
            vector<int> *indices);

/**
 * 带有box-voting的NMS方法
 * @param bboxes         [所有检测的boxes]
 * @param scores         [对应的置信度信息]
 * @param conf_threshold [置信度阈值]
 * @param nms_threshold  [nms阈值]
 * @param top_k          [top-K]
 * @param voting_thre    [与保留目标IOU超过该阈值的，才会进入最终的box-voting计算阶段]
 * @param indices        [保留的结果索引]
 */
template <typename Dtype>
void NmsFastWithVoting(vector<LabeledBBox<Dtype> > *bboxes,
                       const vector<Dtype> &scores,
                       const Dtype conf_threshold,
                       const Dtype nms_threshold,
                       const int top_k,
                       const Dtype voting_thre,
                       vector<int> *indices);

/**
 * 将输入box从center转换为corner模式
 */
template <typename Dtype>
void ToCorner(const BoundingBox<Dtype>& input, BoundingBox<Dtype>* output);

/**
 * 将输入box从corner转换为center模式
 */
template <typename Dtype>
void ToCenter(const BoundingBox<Dtype>& input, BoundingBox<Dtype>* output);

template <typename Dtype>
/**
 * 获取某个归一化box的size-level
 * @param  bbox      [输入box]
 * @param  size_thre [size不同level的阈值表]
 * @return           [该box所属的level]
 */
int GetBBoxLevel(const BoundingBox<Dtype>& bbox, map<int, Dtype> &size_thre);

/**
 * 获取GT-boxes列表中属于某个level的boxes列表
 * @param size_thre      [size不同level的阈值表]
 * @param all_gtboxes    [所有的GT-Boxes列表]
 * @param leveld_gtboxes [不同level的GT-Boxes列表]
 */
template <typename Dtype>
void GetLeveledGTBBoxes(map<int, Dtype> &size_thre,
                        map<int, map<int, vector<LabeledBBox<Dtype> > > > &all_gtboxes,
                        vector<map<int, map<int, vector<LabeledBBox<Dtype> > > > > *leveld_gtboxes);

/**
 * 获取某个diff_level和某个size_level的评估结果
 * diff_level: 针对不同IOU阈值，例如0.5/0.75/0.9
 * size_level: 针对不同size的GT-Boxes，例如0/0.01/0.05/0.1/0.15/0.2/0.25
 * @param l_gtboxes      [某个size_level的GT-Boxes列表]
 * @param all_detections [所有的检测结果]
 * @param size_thre      [size阈值]
 * @param iou_threshold  [iou阈值]
 * @param num_classes    [类别数]
 * @param level          [size的等级]
 * @param diff           [diff的等级]
 * @param l_res          [返回的评估结果]
 */
template <typename Dtype>
void GetLeveledEvalDetections(map<int, map<int, vector<LabeledBBox<Dtype> > > >& l_gtboxes,
                              map<int, map<int, vector<LabeledBBox<Dtype> > > >& all_detections,
                              const Dtype size_thre, const Dtype iou_threshold,
                              const int num_classes, const int level, const int diff,
                              const vector<int>& gt_labels,
                              vector<vector<Dtype> >* l_res);

/**
 * 对检测结果进行可视化
 * @param images       [输入图像列表]
 * @param all_dets     [所有检测结果列表]
 * @param visual_param [可视化参数，见caffe.proto]
 */
template <typename Dtype>
void ShowBBox(const vector<cv::Mat> &images,
              const vector<vector<vector<LabeledBBox<Dtype> > > > &all_dets,
              const VisualizeParameter &visual_param);
}

#endif
