#include <algorithm>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes) {
  object_bboxes->clear();
  // 遍历所有的类别
  for (int i = 0; i < anno_datum.annotation_group_size(); ++i) {
    // 遍历每个类下面的boxes
    const AnnotationGroup& anno_group = anno_datum.annotation_group(i);
    for (int j = 0; j < anno_group.annotation_size(); ++j) {
      // 取得其box信息
      const Annotation& anno = anno_group.annotation(j);
      object_bboxes->push_back(anno.bbox());
    }
  }
}

void GroupObjectBBoxes(const vector<AnnotationGroup>& anno_group,
                       vector<NormalizedBBox>* object_bboxes) {
  object_bboxes->clear();
  for (int i = 0; i < anno_group.size(); ++i) {
    const AnnotationGroup& anno = anno_group[i];
    for (int j = 0; j < anno.annotation_size(); ++j) {
      const Annotation& an = anno.annotation(j);
      object_bboxes->push_back(an.bbox());
    }
  }
}

bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const SampleConstraint& sample_constraint) {
  bool has_jaccard_overlap = sample_constraint.has_min_jaccard_overlap() ||
      sample_constraint.has_max_jaccard_overlap();
  bool has_sample_coverage = sample_constraint.has_min_sample_coverage() ||
      sample_constraint.has_max_sample_coverage();
  bool has_object_coverage = sample_constraint.has_min_object_coverage() ||
      sample_constraint.has_max_object_coverage();
  bool satisfy = !has_jaccard_overlap && !has_sample_coverage &&
      !has_object_coverage;
  // 表明没有约束条件,直接返回真,认为是正例
  if (satisfy) {
    // By default, the sampled_bbox is "positive" if no constraints are defined.
    return true;
  }
  // Check constraints.
  bool found = false;
  // 遍历所有标定列表中的boxes
  for (int i = 0; i < object_bboxes.size(); ++i) {
    const NormalizedBBox& object_bbox = object_bboxes[i];
    // Test jaccard overlap.
    if (has_jaccard_overlap) {
      // 判断两者的iou
      const float jaccard_overlap = JaccardOverlap(sampled_bbox, object_bbox);
      // 不满足最小约束条件
      if (sample_constraint.has_min_jaccard_overlap() &&
          jaccard_overlap < sample_constraint.min_jaccard_overlap()) {
        continue;
      }
      // 不满足最大约束条件
      if (sample_constraint.has_max_jaccard_overlap() &&
          jaccard_overlap > sample_constraint.max_jaccard_overlap()) {
        continue;
      }
      // 如果都满足,则认为找到正例,返回True
      found = true;
    }
    // Test sample coverage.
    if (has_sample_coverage) {
      const float sample_coverage = BBoxCoverage(sampled_bbox, object_bbox);
      if (sample_constraint.has_min_sample_coverage() &&
          sample_coverage < sample_constraint.min_sample_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_sample_coverage() &&
          sample_coverage > sample_constraint.max_sample_coverage()) {
        continue;
      }
      found = true;
    }
    // Test object coverage.
    if (has_object_coverage) {
      const float object_coverage = BBoxCoverage(object_bbox, sampled_bbox);
      if (sample_constraint.has_min_object_coverage() &&
          object_coverage < sample_constraint.min_object_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_object_coverage() &&
          object_coverage > sample_constraint.max_object_coverage()) {
        continue;
      }
      found = true;
    }
    if (found) {
      return true;
    }
  }
  return found;
}

/**
 * 根据采样器采样一个box
 * @param sampler      [采样器]
 * @param sampled_bbox [采样得到的box]
 */
void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  // 在最小和最大值之间产生一个随机数
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);

  // Get random aspect ratio.
  CHECK_GE(sampler.max_aspect_ratio(), sampler.min_aspect_ratio());
  CHECK_GT(sampler.min_aspect_ratio(), 0.);
  CHECK_LT(sampler.max_aspect_ratio(), FLT_MAX);
  float aspect_ratio;
  /**
   * min_ar = max{ min_ar, scale^2 }
   * max_ar = min{ max_ar, 1./(scale^2) }
   * 设定长宽比随机范围
   */
  float min_aspect_ratio = std::max<float>(sampler.min_aspect_ratio(),
                                           std::pow(scale, 2.));
  float max_aspect_ratio = std::min<float>(sampler.max_aspect_ratio(),
                                           1 / std::pow(scale, 2.));
  caffe_rng_uniform(1, min_aspect_ratio, max_aspect_ratio, &aspect_ratio);

  // 获得长宽增益系数
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);

  // 随机获取裁剪坐标
  float w_off, h_off;
  caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
  caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

  /**
   * 获取随机裁剪后的box
   * 原图片将按照此参数进行裁剪
   */
  sampled_bbox->set_xmin(w_off);
  sampled_bbox->set_ymin(h_off);
  sampled_bbox->set_xmax(w_off + bbox_width);
  sampled_bbox->set_ymax(h_off + bbox_height);
}

/**
 * Samples生成器
 * @param source_bbox    [原始图片尺寸,0/1]
 * @param object_bboxes  [标定的boxes列表]
 * @param batch_sampler  [采样器]
 * @param sampled_bboxes [采样结果]
 * 注意:采样器实际上就是一个随机裁剪器,获得一个随机的box
 * 如果该随机生成的box与标定boxes列中的某个box满足IOU条件,则认为找到![视为正例]
 */
void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes) {
  int found = 0;
  //
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    // 如果制定了最大采样数,且查找的符合正例的采样数已经达到该值,则直接返回
    // 即:已经满足条件
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    NormalizedBBox sampled_bbox;
    // 根据采样器,产生一个随机的裁剪box
    SampleBBox(batch_sampler.sampler(), &sampled_bbox);
    // 由于source_bbox是[0,1],这一步没有发生任何变化
    LocateBBox(source_bbox, sampled_bbox, &sampled_bbox);
    // Determine if the sampled bbox is positive or negative by the constraint.
    // 根据正例条件
    // 判断该随机采样的box是否与标定列表中的某个box满足正例条件
    // 如果满足,则返回True,表明该随机生成的box是一个正例
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      // 找到一个正例
      // 将其压栈到采样box列表中
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes) {
  sampled_bboxes->clear();
  vector<NormalizedBBox> object_bboxes;
  // 将所有的box入栈,包括所有类下面的所有boxes
  GroupObjectBBoxes(anno_datum, &object_bboxes);
  // 遍历所有的采样器
  // 每个标定的box会在每个采样器下产生一个映射
  for (int i = 0; i < batch_samplers.size(); ++i) {
    if (batch_samplers[i].use_original_image()) {
      NormalizedBBox unit_bbox;
      // 使用原始图片
      unit_bbox.set_xmin(0);
      unit_bbox.set_ymin(0);
      unit_bbox.set_xmax(1);
      unit_bbox.set_ymax(1);
      // 使用原图
      // 所有标定的boxes列表
      // 给定的采样器
      // 功能:使用该采样器获得一个满足条件的box采样
      // 采样器最多采样max_trial次,超过则直接返回
      GenerateSamples(unit_bbox, object_bboxes, batch_samplers[i],
                      sampled_bboxes);
    }
  }
}

void GenerateBatchSamples(const vector<AnnotationGroup>& anno_group,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes) {
  sampled_bboxes->clear();
  vector<NormalizedBBox> object_bboxes;
  GroupObjectBBoxes(anno_group, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    if (batch_samplers[i].use_original_image()) {
      NormalizedBBox unit_bbox;
      unit_bbox.set_xmin(0);
      unit_bbox.set_ymin(0);
      unit_bbox.set_xmax(1);
      unit_bbox.set_ymax(1);
      GenerateSamples(unit_bbox, object_bboxes, batch_samplers[i],
                      sampled_bboxes);
    }
  }
}

}  // namespace caffe
