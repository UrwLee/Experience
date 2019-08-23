#include <algorithm>
#include <vector>

#include "caffe/tracker/basic.hpp"
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/mask/sample.hpp"
#include "caffe/mask/anno_image_loader.hpp"
#include "caffe/pose/pose_image_loader.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
namespace caffe {

// 搜集所有的boxes
// 注意坐标必须归一化
template <typename Dtype>
void GroupObjectBBoxes(const AnnoData<Dtype>& anno,
                       vector<BoundingBox<Dtype> >* object_bboxes) {
  object_bboxes->clear();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    // NOTE: Only cid == 0 (PERSON)
    if (anno.instances[i].cid == 0) {
      object_bboxes->push_back(anno.instances[i].bbox);
    }
  }
}

template void GroupObjectBBoxes(const AnnoData<float>& anno, vector<BoundingBox<float> >* object_bboxes);
template void GroupObjectBBoxes(const AnnoData<double>& anno, vector<BoundingBox<double> >* object_bboxes);

template <typename Dtype>
void GroupObjectBBoxes(const AnnoData<Dtype>& anno,
                       vector<BoundingBox<Dtype> >* object_bboxes, vector<BoundingBox<Dtype> >* head_bboxes) {
  object_bboxes->clear();
  head_bboxes->clear();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    if (anno.instances[i].cid == 0) {
      object_bboxes->push_back(anno.instances[i].bbox);
      head_bboxes->push_back(anno.instances[i].THbbox);
    }
  }
}

template void GroupObjectBBoxes(const AnnoData<float>& anno, vector<BoundingBox<float> >* object_bboxes, vector<BoundingBox<float> >* head_bboxes);
template void GroupObjectBBoxes(const AnnoData<double>& anno, vector<BoundingBox<double> >* object_bboxes, vector<BoundingBox<double> >* head_bboxes);

template <typename Dtype>
void GroupPartBBoxes(const AnnoData<Dtype>& anno,
                     vector<BoundingBox<Dtype> >* object_bboxes) {
  object_bboxes->clear();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    if (anno.instances[i].cid != 0) {
      object_bboxes->push_back(anno.instances[i].bbox);
    }
  }
}
template void GroupPartBBoxes(const AnnoData<float>& anno, vector<BoundingBox<float> >* object_bboxes);
template void GroupPartBBoxes(const AnnoData<double>& anno, vector<BoundingBox<double> >* object_bboxes);

// 检查采样获得的box是否满足条件
template <typename Dtype>
bool SatisfySampleConstraint(const BoundingBox<Dtype>& sampled_bbox,
                             const vector<BoundingBox<Dtype> >& object_bboxes,
                             const SampleConstraint& sample_constraint) {
  bool has_jaccard_overlap = sample_constraint.has_min_jaccard_overlap() ||
                             sample_constraint.has_max_jaccard_overlap();
  bool has_sample_coverage = sample_constraint.has_min_sample_coverage() ||
                             sample_constraint.has_max_sample_coverage();
  bool has_object_coverage = sample_constraint.has_min_object_coverage() ||
                             sample_constraint.has_max_object_coverage();
  bool satisfy = !has_jaccard_overlap && !has_sample_coverage &&
                 !has_object_coverage;

  if (satisfy) {
    return true;
  }
  // Check constraints.
  bool found = false;
  // 遍历所有标定列表中的boxes
  for (int i = 0; i < object_bboxes.size(); ++i) {
    const BoundingBox<Dtype>& object_bbox = object_bboxes[i];
    // Test jaccard overlap.
    if (has_jaccard_overlap) {
      // 判断两者的iou
      const float jaccard_overlap = sampled_bbox.compute_iou(object_bbox);
      if (sample_constraint.has_min_jaccard_overlap() &&
          jaccard_overlap < sample_constraint.min_jaccard_overlap()) {
        continue;
      }
      if (sample_constraint.has_max_jaccard_overlap() &&
          jaccard_overlap > sample_constraint.max_jaccard_overlap()) {
        continue;
      }
      found = true;
    }
    // Test sample coverage.
    if (has_sample_coverage) {
      const float sample_coverage = sampled_bbox.compute_coverage(object_bbox);
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
      const float object_coverage = sampled_bbox.compute_obj_coverage(object_bbox);
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

template bool SatisfySampleConstraint(const BoundingBox<float>& sampled_bbox,
                                      const vector<BoundingBox<float> >& object_bboxes,
                                      const SampleConstraint& sample_constraint);
template bool SatisfySampleConstraint(const BoundingBox<double>& sampled_bbox,
                                      const vector<BoundingBox<double> >& object_bboxes,
                                      const SampleConstraint& sample_constraint);
// 随机获取一个采样box
template <typename Dtype>
void SampleBBox(const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  // CHECK_LE(sampler.max_scale(), 1.);
  float min_scale = sampler.min_scale() > 1.0 ?  0.95 : sampler.min_scale();
  float max_scale = sampler.max_scale() > 1.0 ?  1.0 : sampler.max_scale();
  float scale;
  caffe_rng_uniform(1, min_scale, max_scale, &scale);
  // Get random aspect ratio.
  CHECK_GE(sampler.max_aspect_ratio(), sampler.min_aspect_ratio());
  CHECK_GT(sampler.min_aspect_ratio(), 0.);
  CHECK_LT(sampler.max_aspect_ratio(), FLT_MAX);
  float aspect_ratio;
  float min_aspect_ratio = std::max<float>(sampler.min_aspect_ratio(),
                           std::pow(scale, 2.));
  float max_aspect_ratio = std::min<float>(sampler.max_aspect_ratio(),
                           1 / std::pow(scale, 2.));
  caffe_rng_uniform(1, min_aspect_ratio, max_aspect_ratio, &aspect_ratio);

  // 获得长宽增益系数
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);
  bbox_width = std::min<float>(bbox_width, 1.0);
  bbox_height = std::min<float>(bbox_height, 1.0);

  // 随机获取裁剪坐标
  float w_off, h_off;
  caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
  caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

  sampled_bbox->x1_ = w_off;
  sampled_bbox->y1_ = h_off;
  sampled_bbox->x2_ = w_off + bbox_width;
  sampled_bbox->y2_ = h_off + bbox_height;
}

template void SampleBBox(const Sampler& sampler, BoundingBox<float>* sampled_bbox);
template void SampleBBox(const Sampler& sampler, BoundingBox<double>* sampled_bbox);

// 随机获取一个16:9的采样box
template <typename Dtype>
void SampleBBox16_9(const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox, const float h_max, const float w_max) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  // CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);
  // 获得长宽增益系数
  float bbox_width = scale * w_max;
  float bbox_height = scale * h_max;
  // 随机获取裁剪坐标
  float w_off, h_off;
  float xmin = std::min(1 - bbox_width, 0.f);
  float xmax = std::max(1 - bbox_width, 0.f);
  float ymin = std::min(1 - bbox_height, 0.f);
  float ymax = std::max(1 - bbox_height, 0.f);
  caffe_rng_uniform(1, xmin, xmax, &w_off);
  caffe_rng_uniform(1, ymin, ymax, &h_off);

  sampled_bbox->x1_ = w_off;
  sampled_bbox->y1_ = h_off;
  sampled_bbox->x2_ = w_off + bbox_width;
  sampled_bbox->y2_ = h_off + bbox_height;
}

template void SampleBBox16_9(const Sampler& sampler, BoundingBox<float>* sampled_bbox, const float h_max, const float w_max);
template void SampleBBox16_9(const Sampler& sampler, BoundingBox<double>* sampled_bbox, const float h_max, const float w_max);

template <typename Dtype>
void SampleBBox16_9_ytop(const vector<BoundingBox<Dtype> > obj_boxes, const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox, const float h_max, const float w_max) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  // CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);
  // 获得长宽增益系数
  float bbox_width = scale * w_max;
  float bbox_height = scale * h_max;
  // 随机获取裁剪坐标
  float w_off, h_off;
  float xmin = std::min(1 - bbox_width, 0.f);
  float xmax = std::max(1 - bbox_width, 0.f);
  float ymin = std::min(1 - bbox_height, 0.f);
  float ymax = std::max(1 - bbox_height, 0.f);
  caffe_rng_uniform(1, xmin, xmax, &w_off);
  caffe_rng_uniform(1, ymin, ymax, &h_off);
  for (int i = 0 ; i < obj_boxes.size() ; ++i) {
    if (obj_boxes[i].y1_ < h_off) {
      h_off = obj_boxes[i].y1_;
    }
  }

  sampled_bbox->x1_ = w_off;
  sampled_bbox->y1_ = h_off;
  sampled_bbox->x2_ = w_off + bbox_width;
  sampled_bbox->y2_ = h_off + bbox_height;
}

template void SampleBBox16_9_ytop(const vector<BoundingBox<float> > obj_boxes, const Sampler& sampler, BoundingBox<float>* sampled_bbox, const float h_max, const float w_max);
template void SampleBBox16_9_ytop(const vector<BoundingBox<double> > obj_boxes, const Sampler& sampler, BoundingBox<double>* sampled_bbox, const float h_max, const float w_max);

template <typename Dtype>
void SampleBBox16_9_head(const vector<BoundingBox<Dtype> > head_bboxes, const Sampler& sampler, BoundingBox<Dtype>* sampled_bbox, const float h_max, const float w_max, bool not_found) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  // CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);
  // 获得长宽增益系数
  float bbox_width = scale * w_max;
  float bbox_height = scale * h_max;
  // 随机获取裁剪坐标
  float w_off, h_off;
  float xmin = std::min(1 - bbox_width, 0.f);
  float xmax = std::max(1 - bbox_width, 0.f);
  float ymin = std::min(1 - bbox_height, 0.f);
  float ymax = std::max(1 - bbox_height, 0.f);
  caffe_rng_uniform(1, xmin, xmax, &w_off);
  caffe_rng_uniform(1, ymin, ymax, &h_off);
  for (int i = 0 ; i < head_bboxes.size() ; ++i) {
    if (head_bboxes[i].x1_ < w_off ||
        head_bboxes[i].y1_ < h_off ||
        head_bboxes[i].x2_ > w_off + bbox_width ||
        head_bboxes[i].y2_ > h_off + bbox_height) {
      not_found  = true;
      return;
    }
  }

  sampled_bbox->x1_ = w_off;
  sampled_bbox->y1_ = h_off;
  sampled_bbox->x2_ = w_off + bbox_width;
  sampled_bbox->y2_ = h_off + bbox_height;
}

template void SampleBBox16_9_head(const vector<BoundingBox<float> > head_bboxes, const Sampler& sampler, BoundingBox<float>* sampled_bbox, const float h_max, const float w_max, bool not_found);
template void SampleBBox16_9_head(const vector<BoundingBox<double> > head_bboxes, const Sampler& sampler, BoundingBox<double>* sampled_bbox, const float h_max, const float w_max, bool not_found);

template <typename Dtype>
void SampleBBox16_9(const float boxscale, BoundingBox<Dtype>* sampled_bbox, const float h_max, const float w_max) {

  // 获得长宽增益系数
  float bbox_width = boxscale * w_max;
  float bbox_height = boxscale * h_max;
  // 随机获取裁剪坐标
  float w_off, h_off;
  float xmin = std::min(1 - bbox_width, 0.f);
  float xmax = std::max(1 - bbox_width, 0.f);
  float ymin = std::min(1 - bbox_height, 0.f);
  float ymax = std::max(1 - bbox_height, 0.f);
  caffe_rng_uniform(1, xmin, xmax, &w_off);
  caffe_rng_uniform(1, ymin, ymax, &h_off);

  sampled_bbox->x1_ = w_off;
  sampled_bbox->y1_ = h_off;
  sampled_bbox->x2_ = w_off + bbox_width;
  sampled_bbox->y2_ = h_off + bbox_height;
}

template void SampleBBox16_9(const float boxscale, BoundingBox<float>* sampled_bbox, const float h_max, const float w_max);
template void SampleBBox16_9(const float boxscale, BoundingBox<double>* sampled_bbox, const float h_max, const float w_max);

template <typename Dtype>
void SampleBBox16_9AroundGT(const Sampler& sampler, const vector<BoundingBox<Dtype> >& object_bboxes,
                            BoundingBox<Dtype>* sampled_bbox, const float h_max, const float w_max) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  // CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);
  // 获得长宽增益系数
  float bbox_width = scale * w_max;
  float bbox_height = scale * h_max;
  float w_off, h_off;
  float xmin, xmax, ymin, ymax;

  if (scale < 1.0) {
    if (object_bboxes.size() > 0) {
      int idx = caffe_rng_rand() % object_bboxes.size(); // 随机选择gt
      BoundingBox<Dtype> gt_bbox;
      gt_bbox = object_bboxes[idx];
      float gt_x1 = std::max((float)(gt_bbox.x1_), 0.f);
      float gt_x2 = std::max((float)(gt_bbox.x2_), 0.f);
      float gt_y1 = std::max((float)(gt_bbox.y1_), 0.f);
      float gt_y2 = std::max((float)(gt_bbox.y2_), 0.f); // gt ( 0~1 )

      /*
         两种情况: 1.bbox_width > gt_w , 2.bbox_width < gt_w
      */
      //    LOG(INFO) << gt_x1<<","<<gt_x2<<";"<<gt_y1<<","<<gt_y2;
      xmin = std::min( std::max((gt_x2 - bbox_width ), 0.f) , gt_x1 ); // 在gt周围crop，
      xmax = std::max( std::max((gt_x2 - bbox_width ), 0.f) , gt_x1 );
      ymin = std::min( std::max((gt_y2 - bbox_height), 0.f) , gt_y1 );
      ymax = std::max( std::max((gt_y2 - bbox_height), 0.f) , gt_y1 );
      caffe_rng_uniform(1, xmin, xmax, &w_off);
      caffe_rng_uniform(1, ymin, ymax, &h_off);
      sampled_bbox->x1_ = w_off;
      sampled_bbox->y1_ = h_off;
      sampled_bbox->x2_ = std::min((w_off + bbox_width), 1.f);
      sampled_bbox->y2_ = std::min((h_off + bbox_height), 1.f);
      //    LOG(INFO) << "bbox_width:" << bbox_width << ", " << " bbox_height:" << bbox_height;
      //  LOG(INFO) << "xmin, xmax " << xmin << "_" << xmax;
      //  LOG(INFO) << "ymin, ymax " << ymin << "_" << ymax;
    }
    else {
      xmin = std::min(1 - bbox_width, 0.f); //
      xmax = std::max(1 - bbox_width, 0.f);
      ymin = std::min(1 - bbox_height, 0.f);
      ymax = std::max(1 - bbox_height, 0.f);

      caffe_rng_uniform(1, xmin, xmax, &w_off);
      caffe_rng_uniform(1, ymin, ymax, &h_off);

      sampled_bbox->x1_ = w_off;
      sampled_bbox->y1_ = h_off;
      sampled_bbox->x2_ = w_off + bbox_width;
      sampled_bbox->y2_ = h_off + bbox_height;
    }
  }
  else {
    //  随机获取裁剪坐标
    xmin = std::min(1 - bbox_width, 0.f);
    xmax = std::max(1 - bbox_width, 0.f);
    ymin = std::min(1 - bbox_height, 0.f);
    ymax = std::max(1 - bbox_height, 0.f);

    caffe_rng_uniform(1, xmin, xmax, &w_off);
    caffe_rng_uniform(1, ymin, ymax, &h_off);

    sampled_bbox->x1_ = w_off;
    sampled_bbox->y1_ = h_off;
    sampled_bbox->x2_ = w_off + bbox_width;
    sampled_bbox->y2_ = h_off + bbox_height;
  }
//  LOG(INFO) << w_off*512 << "*" << h_off*288 << "*" << (w_off + bbox_width)*512 << "*" << (h_off + bbox_height) * 288;
//  LOG(INFO) << "xmin, xmax " << xmin*512 << "_" << xmax*512;
//  LOG(INFO) << "ymin, ymax " << ymin*288 << "_" << ymax*288;


}

template void SampleBBox16_9AroundGT(const Sampler& sampler, const vector<BoundingBox<float> >& object_bboxes,
                                     BoundingBox<float>* sampled_bbox, const float h_max, const float w_max);
template void SampleBBox16_9AroundGT(const Sampler& sampler, const vector<BoundingBox<double> >& object_bboxes,
                                     BoundingBox<double>* sampled_bbox, const float h_max, const float w_max);

// 基于采样器获取裁剪box
template <typename Dtype>
void GenerateSamples(const vector<BoundingBox<Dtype> >& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<BoundingBox<Dtype> >* sampled_bboxes) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    BoundingBox<Dtype> sampled_bbox;
    SampleBBox(batch_sampler.sampler(), &sampled_bbox);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamples(const vector<BoundingBox<float> >& object_bboxes,
                              const BatchSampler& batch_sampler,
                              vector<BoundingBox<float> >* sampled_bboxes);
template void GenerateSamples(const vector<BoundingBox<double> >& object_bboxes,
                              const BatchSampler& batch_sampler,
                              vector<BoundingBox<double> >* sampled_bboxes);
// 生成采样box
template <typename Dtype>
void GenerateSamples16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                         const BatchSampler& batch_sampler,
                         vector<BoundingBox<Dtype> >* sampled_bboxes,
                         const float h_max, const float w_max) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    BoundingBox<Dtype> sampled_bbox;
    SampleBBox16_9(batch_sampler.sampler(), &sampled_bbox, h_max, w_max);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamples16_9(const vector<BoundingBox<float> >& object_bboxes,
                                  const BatchSampler& batch_sampler,
                                  vector<BoundingBox<float> >* sampled_bboxes,
                                  const float h_max, const float w_max);
template void GenerateSamples16_9(const vector<BoundingBox<double> >& object_bboxes,
                                  const BatchSampler& batch_sampler,
                                  vector<BoundingBox<double> >* sampled_bboxes,
                                  const float h_max, const float w_max);

template <typename Dtype>
void GenerateSamples16_9_ytop(const vector<BoundingBox<Dtype> >& object_bboxes,
                              const BatchSampler& batch_sampler,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max, const float w_max) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    BoundingBox<Dtype> sampled_bbox;
    SampleBBox16_9_ytop(object_bboxes, batch_sampler.sampler(), &sampled_bbox, h_max, w_max);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamples16_9_ytop(const vector<BoundingBox<float> >& object_bboxes,
                                       const BatchSampler& batch_sampler,
                                       vector<BoundingBox<float> >* sampled_bboxes,
                                       const float h_max, const float w_max);
template void GenerateSamples16_9_ytop(const vector<BoundingBox<double> >& object_bboxes,
                                       const BatchSampler& batch_sampler,
                                       vector<BoundingBox<double> >* sampled_bboxes,
                                       const float h_max, const float w_max);

template <typename Dtype>
void GenerateSamples16_9_havehead(const vector<BoundingBox<Dtype> >& object_bboxes, const vector<BoundingBox<Dtype> >& head_bboxes,
                                  const BatchSampler& batch_sampler,
                                  vector<BoundingBox<Dtype> >* sampled_bboxes,
                                  const float h_max, const float w_max) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    BoundingBox<Dtype> sampled_bbox;
    bool not_found = false;
    SampleBBox16_9_head(head_bboxes, batch_sampler.sampler(), &sampled_bbox, h_max, w_max, not_found);
    if (not_found) {
      continue;
    }
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamples16_9_havehead(const vector<BoundingBox<float> >& object_bboxes, const vector<BoundingBox<float> >& head_bboxes,
    const BatchSampler& batch_sampler,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateSamples16_9_havehead(const vector<BoundingBox<double> >& object_bboxes, const vector<BoundingBox<double> >& head_bboxes,
    const BatchSampler& batch_sampler,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);

// 生成采样box
template <typename Dtype>
void GenerateSamplesAroundGT16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                                 const BatchSampler& batch_sampler,
                                 vector<BoundingBox<Dtype> >* sampled_bboxes,
                                 const float h_max, const float w_max) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) { // func:
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) { // func: 超过max_sample 跳出
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    BoundingBox<Dtype> sampled_bbox;
    // SampleBBox16_9(batch_sampler.sampler(), &sampled_bbox, h_max, w_max);
    SampleBBox16_9AroundGT(batch_sampler.sampler(), object_bboxes, &sampled_bbox, h_max, w_max);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamplesAroundGT16_9(const vector<BoundingBox<float> >& object_bboxes,
    const BatchSampler& batch_sampler,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateSamplesAroundGT16_9(const vector<BoundingBox<double> >& object_bboxes,
    const BatchSampler& batch_sampler,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);


// 生成采样box
template <typename Dtype>
void GenerateSamples16_9(const vector<BoundingBox<Dtype> >& object_bboxes,
                         const BatchSampler& batch_sampler,
                         vector<BoundingBox<Dtype> >* sampled_bboxes,
                         const float h_max, const float w_max, const float boxscale) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    BoundingBox<Dtype> sampled_bbox;
    SampleBBox16_9(boxscale, &sampled_bbox, h_max, w_max);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

template void GenerateSamples16_9(const vector<BoundingBox<float> >& object_bboxes,
                                  const BatchSampler& batch_sampler,
                                  vector<BoundingBox<float> >* sampled_bboxes,
                                  const float h_max, const float w_max, const float boxscale);
template void GenerateSamples16_9(const vector<BoundingBox<double> >& object_bboxes,
                                  const BatchSampler& batch_sampler,
                                  vector<BoundingBox<double> >* sampled_bboxes,
                                  const float h_max, const float w_max, const float boxscale);

// 生成多个采样器的boxes
template <typename Dtype>
void GenerateBatchSamples16_9(const AnnoData<Dtype>& anno,
                              const vector<BatchSampler>& batch_samplers,
                              vector<BoundingBox<Dtype> >* sampled_bboxes,
                              const float h_max, const float w_max) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  GroupObjectBBoxes(anno, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    GenerateSamples16_9(object_bboxes, batch_samplers[i],
                        sampled_bboxes, h_max, w_max);
  }
}

template void GenerateBatchSamples16_9(const AnnoData<float>& anno,
                                       const vector<BatchSampler>& batch_samplers,
                                       vector<BoundingBox<float> >* sampled_bboxes,
                                       const float h_max, const float w_max);
template void GenerateBatchSamples16_9(const AnnoData<double>& anno,
                                       const vector<BatchSampler>& batch_samplers,
                                       vector<BoundingBox<double> >* sampled_bboxes,
                                       const float h_max, const float w_max);

template <typename Dtype>
void GenerateBatchSamples16_9_ytop(const AnnoData<Dtype>& anno,
                                   const vector<BatchSampler>& batch_samplers,
                                   vector<BoundingBox<Dtype> >* sampled_bboxes,
                                   const float h_max, const float w_max) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  GroupObjectBBoxes(anno, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    GenerateSamples16_9_ytop(object_bboxes, batch_samplers[i],
                             sampled_bboxes, h_max, w_max);
  }
}

template void GenerateBatchSamples16_9_ytop(const AnnoData<float>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateBatchSamples16_9_ytop(const AnnoData<double>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);

template <typename Dtype>
void GenerateBatchSamples16_9_havehead(const AnnoData<Dtype>& anno,
                                       const vector<BatchSampler>& batch_samplers,
                                       vector<BoundingBox<Dtype> >* sampled_bboxes,
                                       const float h_max, const float w_max) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  vector<BoundingBox<Dtype> > head_bboxes;
  GroupObjectBBoxes(anno, &object_bboxes, &head_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    GenerateSamples16_9_havehead(object_bboxes, head_bboxes, batch_samplers[i],
                                 sampled_bboxes, h_max, w_max);
  }
}

template void GenerateBatchSamples16_9_havehead(const AnnoData<float>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateBatchSamples16_9_havehead(const AnnoData<double>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);

// 生成多个采样器的boxes
template <typename Dtype>
void GenerateBatchSamples(const AnnoData<Dtype>& anno,
                          const vector<BatchSampler>& batch_samplers,
                          vector<BoundingBox<Dtype> >* sampled_bboxes) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  GroupObjectBBoxes(anno, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    GenerateSamples(object_bboxes, batch_samplers[i],
                    sampled_bboxes);
  }
}

template void GenerateBatchSamples(const AnnoData<float>& anno,
                                   const vector<BatchSampler>& batch_samplers,
                                   vector<BoundingBox<float> >* sampled_bboxes);
template void GenerateBatchSamples(const AnnoData<double>& anno,
                                   const vector<BatchSampler>& batch_samplers,
                                   vector<BoundingBox<double> >* sampled_bboxes);
// ADD
template <typename Dtype>
void GenerateBatchSamples4Parts16_9(const AnnoData<Dtype>& anno,
                                    const vector<BatchSampler>& batch_samplers,
                                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                                    const float h_max, const float w_max) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  GroupPartBBoxes<Dtype>(anno, &object_bboxes);   // Use PartBoxes
  for (int i = 0; i < batch_samplers.size(); ++i) {
    GenerateSamples16_9(object_bboxes, batch_samplers[i],
                        sampled_bboxes, h_max, w_max);
  }
}
template void GenerateBatchSamples4Parts16_9(const AnnoData<float>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateBatchSamples4Parts16_9(const AnnoData<double>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);





template <typename Dtype>
void GenerateBatchSamples4PartsAroundGT16_9(const AnnoData<Dtype>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<Dtype> >* sampled_bboxes,
    const float h_max, const float w_max) {
  sampled_bboxes->clear();
  vector<BoundingBox<Dtype> > object_bboxes;
  GroupPartBBoxes<Dtype>(anno, &object_bboxes);   // func: 把gt实例 放入object_bboxes,
  for (int i = 0; i < batch_samplers.size(); ++i) { // func:
    GenerateSamplesAroundGT16_9(object_bboxes, batch_samplers[i],
                                sampled_bboxes, h_max, w_max);
  }
}
template void GenerateBatchSamples4PartsAroundGT16_9(const AnnoData<float>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateBatchSamples4PartsAroundGT16_9(const AnnoData<double>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);
template <typename Dtype>
void GenerateBatchSamplesRandom16_9(const AnnoData<Dtype>& anno,
                                    const vector<BatchSampler>& batch_samplers,
                                    vector<BoundingBox<Dtype> >* sampled_bboxes,
                                    const float h_max, const float w_max) {
  sampled_bboxes->clear();

  for (int i = 0; i < batch_samplers.size(); ++i) {
    vector<BoundingBox<Dtype> > object_bboxes;

    BatchSampler b_sampler = batch_samplers[i];
    CHECK_GE(b_sampler.sampler().max_scale(), b_sampler.sampler().min_scale());
    CHECK_GT(b_sampler.sampler().min_scale(), 0.);
    float scale;
    caffe_rng_uniform(1, b_sampler.sampler().min_scale(), b_sampler.sampler().max_scale(), &scale);
    if (scale > 0.75) {
      GroupObjectBBoxes(anno, &object_bboxes);
    } else {
      GroupPartBBoxes<Dtype>(anno, &object_bboxes);   // Use PartBoxes
    }
    GenerateSamples16_9(object_bboxes, b_sampler,
                        sampled_bboxes, h_max, w_max, scale);
  }


}
template void GenerateBatchSamplesRandom16_9(const AnnoData<float>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<float> >* sampled_bboxes,
    const float h_max, const float w_max);
template void GenerateBatchSamplesRandom16_9(const AnnoData<double>& anno,
    const vector<BatchSampler>& batch_samplers,
    vector<BoundingBox<double> >* sampled_bboxes,
    const float h_max, const float w_max);

}  // namespace caffe
