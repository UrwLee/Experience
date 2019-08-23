#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include "thrust/functional.h"
#include "thrust/sort.h"

#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
    const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template __host__ __device__ float BBoxSizeGPU(const float* bbox,
    const bool normalized);
template __host__ __device__ double BBoxSizeGPU(const double* bbox,
    const bool normalized);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
    const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSizeGPU(bbox1);
    const Dtype bbox2_size = BBoxSizeGPU(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template __host__ __device__ float JaccardOverlapGPU(const float* bbox1,
    const float* bbox2);
template __host__ __device__ double JaccardOverlapGPU(const double* bbox1,
    const double* bbox2);

template <typename Dtype>
__device__ Dtype Min(const Dtype x, const Dtype y) {
  return x < y ? x : y;
}

template <typename Dtype>
__device__ Dtype Max(const Dtype x, const Dtype y) {
  return x > y ? x : y;
}

template <typename Dtype>
__device__ void ClipBBoxGPU(const Dtype* bbox, Dtype* clip_bbox) {
  for (int i = 0; i < 4; ++i) {
    clip_bbox[i] = Max(Min(bbox[i], Dtype(1.)), Dtype(0.));
  }
}

template __device__ void ClipBBoxGPU(const float* bbox, float* clip_bbox);
template __device__ void ClipBBoxGPU(const double* bbox, double* clip_bbox);

template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int d = (index / 4) % num_priors;
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(xmax) * prior_width;
        decode_bbox_height = exp(ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width =
          exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
          exp(prior_data[vi + 3] * ymax) * prior_height;
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else {
      // Unknown code type.
    }
  }
}

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, Dtype* bbox_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data, code_type,
      variance_encoded_in_target, num_priors, bbox_data);
  CUDA_POST_KERNEL_CHECK;
}

template void DecodeBBoxesGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, float* bbox_data);
template void DecodeBBoxesGPU(const int nthreads,
          const double* loc_data, const double* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, double* bbox_data);

template <typename Dtype>
__global__ void DecodeBBoxesByLocKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const int w, const int h, const int num_priors,
          Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int l = index % 4;
    int temp = index / 4;
    const int n = temp % num_priors;
    temp /= num_priors;
    const int j = temp % w;
    temp /= w;
    const int i = temp % h;

    const Dtype pw = prior_data[2*n];
    const Dtype ph = prior_data[2*n+1];

    const Dtype pred_cx = loc_data[index - l];
    const Dtype pred_cy = loc_data[index - l + 1];
    const Dtype pred_w = loc_data[index - l + 2];
    const Dtype pred_h = loc_data[index - l + 3];

    Dtype center_x, center_y, width, height;
    center_x = (j + 1. / (1. + exp(-pred_cx))) / w;
    center_y = (i + 1. / (1. + exp(-pred_cy))) / h;
    width = exp(pred_w) * pw;
    height = exp(pred_h) * ph;

    switch (l) {
      case 0:
        bbox_data[index] = center_x - width / 2.;
        break;
      case 1:
        bbox_data[index] = center_y - height / 2.;
        break;
      case 2:
        bbox_data[index] = center_x + width / 2.;
        break;
      case 3:
        bbox_data[index] = center_y + height / 2.;
        break;
    }
  }
}

template <typename Dtype>
void DecodeBBoxesByLocGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const int w, const int h, const int num_priors,
          Dtype* bbox_data) {
  DecodeBBoxesByLocKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data,
      w, h, num_priors, bbox_data);
  CUDA_POST_KERNEL_CHECK;
}

template void DecodeBBoxesByLocGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const int w, const int h, const int num_priors,
          float* bbox_data);
template void DecodeBBoxesByLocGPU(const int nthreads,
          const double* loc_data, const double* prior_data,
          const int w, const int h, const int num_priors,
          double* bbox_data);

template <typename Dtype>
__global__ void PermuteDataKernel(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
    new_data[new_index] = data[index];
  }
}

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  PermuteDataKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, data, num_classes, num_data,
      num_dim, new_data);
  CUDA_POST_KERNEL_CHECK;
}

template void PermuteDataGPU(const int nthreads,
          const float* data, const int num_classes, const int num_data,
          const int num_dim, float* new_data);
template void PermuteDataGPU(const int nthreads,
          const double* data, const int num_classes, const int num_data,
          const int num_dim, double* new_data);

// permute
template <typename Dtype>
__global__ void PermuteConfDataToBgClassKernel(const int nthreads,
                    const Dtype* conf_data, const int num_classes,
                    Dtype* bg_data, Dtype* class_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % (num_classes + 1);
    const int box_idx = index / (num_classes + 1);

    if (c == 0) {
      // push to bg_data
      bg_data[box_idx] = conf_data[index];
    } else {
      // push to class_data
      const int class_idx = box_idx * num_classes + c - 1;
      class_data[class_idx] = conf_data[index];
    }
  }
}

template <typename Dtype>
void PermuteConfDataToBgClassGPU(const int nthreads,
                    const Dtype* conf_data, const int num_classes,
                    Dtype* bg_data, Dtype* class_data) {
  PermuteConfDataToBgClassKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads,conf_data,
      num_classes,bg_data,class_data);
}

template void PermuteConfDataToBgClassGPU(const int nthreads,
                    const float* conf_data, const int num_classes,
                    float* bg_data, float* class_data);
template void PermuteConfDataToBgClassGPU(const int nthreads,
                    const double* conf_data, const int num_classes,
                    double* bg_data, double* class_data);
// logistic
template <typename Dtype>
__global__ void kernel_logistic(const int count, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    data[index] = 1. / (1. + exp(-data[index]));
  }
}

template <typename Dtype>
void LogisticGPU(const int count, Dtype* data) {
  kernel_logistic<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, data);
}

template void LogisticGPU(const int count, float* data);
template void LogisticGPU(const int count, double* data);

// Max
template <typename Dtype>
__global__ void kernel_channel_max(const int out_dim,
    const int channels, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, out_dim) {
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[index * channels + c], maxval);
    }
    out[index] = maxval;
  }
}

// Sub
template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int channels, const Dtype* channel_data,
    const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels;
    data[index] = channel_data[index] - channel_max[n];
  }
}

// exp
template <typename Dtype>
__global__ void kernel_exp(const int count, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    data[index] = exp(data[index]);
  }
}

// sum
template <typename Dtype>
__global__ void kernel_channel_sum(const int out_dim,
            const int channels, const Dtype* data,
            Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, out_dim) {
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[index * channels + c];
    }
    channel_sum[index] = sum;
  }
}

// div
template <typename Dtype>
__global__ void kernel_channel_div(const int count,
      const int channels, const Dtype* channel_sum,
      Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels;
    data[index] /= channel_sum[n];
  }
}

// SoftMaxGPU
template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int out_num,
    const int channels, Dtype* prob) {
  vector<int> shape(1,1);
  shape[0] = out_num;
  Blob<Dtype> scale(shape);
  Dtype* scale_data = scale.mutable_gpu_data();
  const int count = out_num * channels;
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(out_num),
        CAFFE_CUDA_NUM_THREADS>>>(out_num, channels, data, scale_data);
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, channels, data, scale_data, prob);
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, prob);
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(out_num),
        CAFFE_CUDA_NUM_THREADS>>>(out_num, channels, prob, scale_data);
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, channels, scale_data, prob);
}

template void SoftMaxGPU(const float* data, const int out_num,
    const int channels, float* prob);
template void SoftMaxGPU(const double* data, const int out_num,
    const int channels, double* prob);

template <typename Dtype>
__global__ void kernel_channel_multiply(const int count,
      const int channels, const Dtype* channel_scalar,
      Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels;
    data[index] *= channel_scalar[n];
  }
}

template <typename Dtype>
void UpdateConfByObjGPU(const int nthreads, const int num_classes,
      const Dtype* objectness, Dtype* conf_data) {
  kernel_channel_multiply<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_classes,
        objectness, conf_data);
}

template void UpdateConfByObjGPU(const int nthreads,
  const int num_classes, const float* objectness, float* conf_data);
template void UpdateConfByObjGPU(const int nthreads,
  const int num_classes, const double* objectness, double* conf_data);

template <typename Dtype>
__global__ void ComputeOverlappedKernel(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_bboxes;
    const int i = (index / num_bboxes) % num_bboxes;
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    const int c = (index / num_bboxes / num_bboxes) % num_classes;
    const int n = index / num_bboxes / num_bboxes / num_classes;
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = ((n * num_bboxes + i) * num_classes + c) * 4;
    const int start_loc_j = ((n * num_bboxes + j) * num_classes + c) * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, num_bboxes, num_classes,
      overlap_threshold, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedGPU(const int nthreads,
          const float* bbox_data, const int num_bboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);
template void ComputeOverlappedGPU(const int nthreads,
          const double* bbox_data, const int num_bboxes, const int num_classes,
          const double overlap_threshold, bool* overlapped_data);

template <typename Dtype>
__global__ void ComputeOverlappedByIdxKernel(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_idx;
    const int i = (index / num_idx);
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = idx[i] * 4;
    const int start_loc_j = idx[j] * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedByIdxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, overlap_threshold,
      idx, num_idx, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedByIdxGPU(const int nthreads,
          const float* bbox_data, const float overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
template void ComputeOverlappedByIdxGPU(const int nthreads,
          const double* bbox_data, const double overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices) {
  // Keep part of detections whose scores are higher than confidence threshold.
  vector<int> idx;
  vector<Dtype> confidences;
  for (int i = 0; i < num_bboxes; ++i) {
    if (conf_data[i] > confidence_threshold) {
      idx.push_back(i);
      confidences.push_back(conf_data[i]);
    }
  }
  int num_remain = confidences.size();
  if (num_remain == 0) {
    return;
  }
  // Sort detections based on score.
  thrust::sort_by_key(&confidences[0], &confidences[0] + num_remain, &idx[0],
      thrust::greater<Dtype>());
  if (top_k > -1 && top_k < num_remain) {
    num_remain = top_k;
  }

  // Compute overlap between remaining detections.
  Blob<int> idx_blob(1, 1, 1, num_remain);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(idx.begin(), idx.begin() + num_remain, idx_data);

  Blob<bool> overlapped(1, 1, num_remain, num_remain);
  const int total_bboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedByIdxGPU<Dtype>(total_bboxes, bbox_data, nms_threshold,
      idx_blob.gpu_data(), num_remain, overlapped_data);

  // Do non-maximum suppression based on overlapped results.
  const bool* overlapped_results = overlapped.cpu_data();
  vector<int> selected_indices;
  ApplyNMS(overlapped_results, num_remain, &selected_indices);

  // Put back the selected information.
  for (int i = 0; i < selected_indices.size(); ++i) {
    indices->push_back(idx[selected_indices[i]]);
  }
}

template
void ApplyNMSGPU(const float* bbox_data, const float* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);
template
void ApplyNMSGPU(const double* bbox_data, const double* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

template <typename Dtype>
__global__ void GetDetectionsKernel(const int nthreads,
          const Dtype* bbox_data, const Dtype* conf_data, const int image_id,
          const int label, const int* indices, const bool clip_bbox,
          Dtype* detection_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int det_idx = indices[index];
    detection_data[index * 7] = image_id;
    detection_data[index * 7 + 1] = label;
    detection_data[index * 7 + 2] = conf_data[det_idx];
    if (clip_bbox) {
      ClipBBoxGPU(&(bbox_data[det_idx * 4]), &(detection_data[index * 7 + 3]));
    } else {
      for (int i = 0; i < 4; ++i) {
        detection_data[index * 7 + 3 + i] = bbox_data[det_idx * 4 + i];
      }
    }
  }
}

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob) {
  // Store selected indices in array.
  int num_det = indices.size();
  if (num_det == 0) {
    return;
  }
  Blob<int> idx_blob(1, 1, 1, num_det);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(indices.begin(), indices.end(), idx_data);
  // Prepare detection_blob.
  detection_blob->Reshape(1, 1, num_det, 7);
  Dtype* detection_data = detection_blob->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GetDetectionsKernel<Dtype><<<CAFFE_GET_BLOCKS(num_det),
      CAFFE_CUDA_NUM_THREADS>>>(num_det, bbox_data, conf_data, image_id, label,
      idx_blob.gpu_data(), clip_bbox, detection_data);
  CUDA_POST_KERNEL_CHECK;
}

template void GetDetectionsGPU(const float* bbox_data, const float* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<float>* detection_blob);
template void GetDetectionsGPU(const double* bbox_data, const double* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<double>* detection_blob);

}  // namespace caffe
