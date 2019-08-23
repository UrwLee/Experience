#ifndef CAFFE_MASK_SAMPLE_H_
#define CAFFE_MASK_SAMPLE_H_

#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

/**
 * 该头文件提供了随机采样器及其方法
 */

/**
 * 将标注信息中的所有目标构成一个boxes集合
 * @param anno          [目标标注结构化信息]
 * @param object_bboxes [boxes列表]
 */
template <typename Dtype>
void GroupObjectBBoxes(const AnnoData<Dtype>& anno,
                       vector<BoundingBox<Dtype> >* object_bboxes);

template <typename Dtype>
void GroupPartBBoxes(const AnnoData<Dtype>& anno,
  vector<BoundingBox<Dtype> >* object_bboxes);
/**
 * box是否满足采样器条件的检查
 * 返回：　TRUE ->  符合条件
 *       FALSE -> 不符合条件
 * @param  sampled_bbox      [采样器bbox]
 * @param  object_bboxes     [目标boxes列表]
 * @param  sample_constraint [采样约束条件]
 * @return                   [返回True/False]
 */
template <typename Dtype>
bool SatisfySampleConstraint(const BoundingBox<Dtype>& sampled_bbox,
                            const vector<BoundingBox<Dtype> >& object_bboxes,
                            const SampleConstraint& sample_constraint);

/**
 * 采样器随机采样一个box
 * @param sampler      [采样器]
 * @param sampled_bbox [采样的box]
 */
template <typename Dtype>
void SampleBBox(const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox);

/**
 * [采样16:9的box]
 * @param sampler      [采样器]
 * @param sampled_bbox [采样得到的box]
 * @param h_max        [最大采样高度]
 * @param w_max        [最大采样宽度]
 */
template <typename Dtype>
void SampleBBox16_9(const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox,const float h_max,const float w_max);
template <typename Dtype>
void SampleBBox16_9(const float boxscale, BoundingBox<Dtype>* sampled_bbox,const float h_max,const float w_max);

template <typename Dtype>
void SampleBBox16_9AroundGT(const float boxscale, BoundingBox<Dtype>* sampled_bbox,const vector<BoundingBox<Dtype> >& object_bboxes,
                const float h_max,const float w_max);
/**
 * 根据BatchSampler采集一系列的boxes
 * @param object_bboxes  [目标boxes列表]
 * @param batch_sampler  [批采样器]
 * @param sampled_bboxes [采样得到的boxes列表]
 */
template <typename Dtype>
void GenerateSamples(const vector<BoundingBox<Dtype> >& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<BoundingBox<Dtype> >* sampled_bboxes);

/**
 * 根据BatchSampler采集一系列的boxes(16:9)
 * @param object_bboxes  [目标boxes列表]
 * @param batch_sampler  [批采样器]
 * @param sampled_bboxes [采样得到的boxes列表]
 * @param h_max          [最大采样高度]
 * @param w_max          [最大采样宽度]
 */
template <typename Dtype>
void GenerateSamples16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                    const BatchSampler& batch_sampler,
                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                    const float h_max,const float w_max);

template <typename Dtype>
void GenerateSamples16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                    const BatchSampler& batch_sampler,
                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                    const float h_max,const float w_max,const float boxscale);

template <typename Dtype>
void GenerateSamples16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                    const BatchSampler& batch_sampler,
                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                    const float h_max,const float w_max,const float boxscale);

template <typename Dtype>
void GenerateSamplesAroundGT16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                    const BatchSampler& batch_sampler,
                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                    const float h_max,const float w_max,const float boxscale);
/**
 * 根据标注信息以及批采样器，获取一系列的采样boxes列表 (16:9)
 * @param anno           [标注信息]
 * @param batch_samplers [批采样器]
 * @param sampled_bboxes [采样得到的boxes列表]
 * @param h_max          [最大采样高度]
 * @param w_max          [最大采样宽度]
 */
template <typename Dtype>
void GenerateBatchSamples16_9(const AnnoData<Dtype>& anno,
                              const vector<BatchSampler>& batch_samplers,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max,const float w_max);
/**
 * 根据标注信息以及批采样器，获取一系列的采样boxes列表
 * @param anno           [标注信息]
 * @param batch_samplers [批采样器]
 * @param sampled_bboxes [采样得到的boxes列表]
 */
template <typename Dtype>
void GenerateBatchSamples(const AnnoData<Dtype>& anno,
                          const vector<BatchSampler>& batch_samplers,
                          vector<BoundingBox<Dtype> >* sampled_bboxes);

template <typename Dtype>
void GenerateBatchSamples4Parts16_9(const AnnoData<Dtype>& anno,
                              const vector<BatchSampler>& batch_samplers,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max,const float w_max);
template <typename Dtype>
void GenerateBatchSamplesRandom16_9(const AnnoData<Dtype>& anno,
                              const vector<BatchSampler>& batch_samplers,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max,const float w_max);

template <typename Dtype>
void GenerateBatchSamples4PartsAroundGT16_9(const AnnoData<Dtype>& anno,
                              const vector<BatchSampler>& batch_samplers,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max,const float w_max);
}

#endif
