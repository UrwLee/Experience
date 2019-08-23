#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
  const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
  const int interpolate_times, const Dtype* bottom_rois, const int b1_width, Dtype* top_data, int* argmax_data, Dtype* w_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, ph, pw) is an element in the pooled output
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		bottom_rois += n * b1_width;
		int roi_batch_ind = bottom_rois[0];

		Dtype roi_start_w = bottom_rois[b1_width-4] * width;
		Dtype roi_start_h = bottom_rois[b1_width-3] * height;
		Dtype roi_end_w = bottom_rois[b1_width-2] * width;
		Dtype roi_end_h = bottom_rois[b1_width-1] * height;
		// clipping
		roi_start_w = max(roi_start_w, Dtype(0));
    roi_start_h = max(roi_start_h, Dtype(0));
		roi_end_w = min(Dtype(width - 1), roi_end_w);
		roi_end_h = min(Dtype(height - 1), roi_end_h);
    // Dtype roi_height = roi_end_h - roi_start_h + 1;
    // Dtype roi_width = roi_end_w - roi_start_w + 1;
		Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
		Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
		const Dtype bin_size_h = static_cast<Dtype>(roi_height)
			/ static_cast<Dtype>(pooled_height);
		const Dtype bin_size_w = static_cast<Dtype>(roi_width)
			/ static_cast<Dtype>(pooled_width);

		bottom_data += (roi_batch_ind * channels + c) * height * width;

		int argmax_temp_data[4];
		Dtype w_temp_data[4];
		Dtype start_x = 0.25, start_y = 0.25;
		if (interpolate_times == 1) {
			start_x = 0.5;
			start_y = 0.5;
		}
		Dtype dfValue = 0, maxValue = 0;
		for (int inter_index = 0; inter_index < interpolate_times; ++inter_index) {
			int index_x = inter_index % 2;
			int index_y = inter_index / 2;
			Dtype off_x = index_x * 0.5 + start_x;
			Dtype off_y = index_y * 0.5 + start_y;
			Dtype hcenter = static_cast<Dtype>(ph + off_y)* bin_size_h;
			Dtype wcenter = static_cast<Dtype>(pw + off_x)* bin_size_w;

			hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height - 1));
			wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width - 1));

			int hstart = min(max(hcenter, Dtype(0)), Dtype(height - 1));
			int wstart = min(max(wcenter, Dtype(0)), Dtype(width - 1));
			int hend = min(max(hstart + 1, 0), height - 1);
			int wend = min(max(wstart + 1, 0), width - 1);

			Dtype fX0 = wcenter - wstart;
			Dtype fX1 = wend - wcenter;
			Dtype fY0 = hcenter - hstart;
			Dtype fY1 = hend - hcenter;
			Dtype fFactorA = fY1 * fX1;
			Dtype fFactorB = fY1 * fX0;
			Dtype fFactorC = fY0 * fX1;
			Dtype fFactorD = fY0 * fX0;

			dfValue = bottom_data[hstart * width + wstart] * fFactorA
				      + bottom_data[hstart * width + wend] * fFactorB
				      + bottom_data[hend * width + wstart] * fFactorC
				      + bottom_data[hend * width + wend] * fFactorD;

			if (inter_index == 0) {
				maxValue = dfValue - 1;
			}

			argmax_temp_data[0] = hstart * width + wstart;
			argmax_temp_data[1] = hstart * width + wend;
			argmax_temp_data[2] = hend * width + wstart;
			argmax_temp_data[3] = hend * width + wend;

			w_temp_data[0] = fFactorA;
			w_temp_data[1] = fFactorB;
			w_temp_data[2] = fFactorC;
			w_temp_data[3] = fFactorD;

			if (dfValue > maxValue || inter_index == 0) {
				maxValue = dfValue;
				top_data[index] = dfValue;
				for (int s = 0; s < 4; ++s) {
					w_data[4 * index + s] = w_temp_data[s];
					argmax_data[4 * index + s] = argmax_temp_data[s];
				}
			}
		}
	}
}

template <typename Dtype>
void RoiAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  const int b1_width = bottom[1]->width();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_rois = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = idx_.mutable_gpu_data();
	Dtype* w_data = coeff_.mutable_gpu_data();
	int count = top[0]->count();
	// NOLINT_NEXT_LINE(whitespace/operators)
	ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    roiResizedHeight_, roiResizedWidth_, inter_times_, bottom_rois, b1_width, top_data, argmax_data, w_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
	const int* argmax_data, const Dtype* w_data,
	const int channels, const int height, const int width,
	const int pooled_height, const int pooled_width, const int w_num,
	const Dtype* bottom_rois, const int b1_width, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index % channels;
		int n = index / channels;
		bottom_rois += n * b1_width;
		int roi_batch_ind = bottom_rois[0];
    top_diff += index * pooled_height * pooled_width;
    argmax_data += index * pooled_height * pooled_width * w_num;
    w_data += index * pooled_height * pooled_width * w_num;
    bottom_diff += (roi_batch_ind * channels + c) * height * width;
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int offs = ph * pooled_width + pw;
        for (int index = 0; index < w_num; ++index) {
          int arg_max = argmax_data[offs * w_num + index];
          Dtype w = w_data[offs * w_num + index];
          if (arg_max >= 0) {
            bottom_diff[arg_max] += top_diff[offs] * w;
          }
        }
      }
    }
	}
}

template <typename Dtype>
void RoiAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
  const int b1_width = bottom[1]->width();
	const Dtype* bottom_rois = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	caffe_gpu_set(count, Dtype(0.), bottom_diff);
	const int* argmax_data = idx_.gpu_data();
	const Dtype* w_data = coeff_.gpu_data();
	const int nthreads = top[0]->num() * top[0]->channels();
	int w_num = 4;
	// NOLINT_NEXT_LINE(whitespace/operators)
	ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, top_diff, argmax_data, w_data, bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width(), roiResizedHeight_, roiResizedWidth_, w_num,
    bottom_rois, b1_width, bottom_diff);
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RoiAlignLayer);

}  // namespace caffe
