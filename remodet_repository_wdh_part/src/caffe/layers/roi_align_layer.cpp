#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"

namespace caffe {

template <typename Dtype>
void RoiAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const RoiAlignParameter& roi_align_param = this->layer_param_.roi_align_param();
	roiResizedWidth_ = roi_align_param.roi_resized_width();
	roiResizedHeight_ = roi_align_param.roi_resized_height();
	inter_times_ = roi_align_param.inter_times();
	CHECK(inter_times_ == 1 || inter_times_ == 4) << "interpolate_times must be 1 or 4.";
	// unused.
	spatial_scale_ = roi_align_param.spatial_scale();
}

template <typename Dtype>
void RoiAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if (bottom[1]->width() != 7 && bottom[1]->width() != 9) {
		LOG(FATAL) << "bottom[1]: the ROI-instances must has a width of 7 or 9.";
	}
	top[0]->Reshape(bottom[1]->height(), bottom[0]->channels(), roiResizedHeight_, roiResizedWidth_);
	idx_.Reshape(bottom[1]->height(), bottom[0]->channels(), roiResizedHeight_, roiResizedWidth_ * 4);
	coeff_.Reshape(bottom[1]->height(), bottom[0]->channels(), roiResizedHeight_, roiResizedWidth_ * 4);
}

template <typename Dtype>
void RoiAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	// Number of ROIs
	int b1_width = bottom[1]->width();
	int num_rois = bottom[1]->height();
	int batch_size = bottom[0]->num();
	int top_count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top_count, Dtype(-FLT_MAX), top_data);
	int* argmax_data = idx_.mutable_cpu_data();
	Dtype* w_data = coeff_.mutable_cpu_data();
	caffe_set(top_count * 4, -1, argmax_data);
	caffe_set(top_count * 4, Dtype(0), w_data);

	const int fw = bottom[0]->width();
	const int fh = bottom[0]->height();
	const int channels = bottom[0]->channels();
	for (int n = 0; n < num_rois; ++n) {
		int roi_batch_ind = bottom_rois[0];
		CHECK_GE(roi_batch_ind, 0);
		CHECK_LT(roi_batch_ind, batch_size);
		Dtype roi_start_w = bottom_rois[b1_width-4] * fw;
		Dtype roi_start_h = bottom_rois[b1_width-3] * fh;
		Dtype roi_end_w = bottom_rois[b1_width-2] * fw;
		Dtype roi_end_h = bottom_rois[b1_width-1] * fh;
		// clipping
		roi_start_w = std::max(roi_start_w, Dtype(0));
		roi_start_h = std::max(roi_start_h, Dtype(0));
		roi_end_w = std::min(Dtype(fw - 1), roi_end_w);
		roi_end_h = std::min(Dtype(fh - 1), roi_end_h);

		Dtype roi_height = std::max(roi_end_h - roi_start_h + 1, Dtype(1));
		Dtype roi_width = std::max(roi_end_w - roi_start_w + 1, Dtype(1));
		const Dtype bin_size_h = static_cast<Dtype>(roi_height)
			/ static_cast<Dtype>(roiResizedHeight_);
		const Dtype bin_size_w = static_cast<Dtype>(roi_width)
			/ static_cast<Dtype>(roiResizedWidth_);

		const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

		Dtype fX0,fX1,fY0,fY1;
		Dtype fFactorA,fFactorB,fFactorC,fFactorD;
		for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < roiResizedHeight_; ++ph) {
				for (int pw = 0; pw < roiResizedWidth_; ++pw) {
					// Compute pooling region for this output unit:
					//  start (included) = floor(ph * roi_height / pooled_height_)
					//  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
					Dtype hcenter = static_cast<Dtype>(ph + 0.5)* bin_size_h;
					Dtype wcenter = static_cast<Dtype>(pw + 0.5)* bin_size_w;

					hcenter = std::min(std::max(hcenter + roi_start_h, Dtype(0)), Dtype(fh - 1));
					wcenter = std::min(std::max(wcenter + roi_start_w, Dtype(0)), Dtype(fw - 1));

					int hstart = std::min(std::max(hcenter, Dtype(0)), Dtype(fh - 1));
					int wstart = std::min(std::max(wcenter, Dtype(0)), Dtype(fw - 1));
					int hend = std::min(std::max(hstart + 1, 0), fh - 1);
					int wend = std::min(std::max(wstart + 1, 0), fw - 1);

					const int pool_index = ph * roiResizedWidth_ + pw;

					fX0 = wcenter - wstart;
					fX1 = wend - wcenter;
					fY0 = hcenter - hstart;
					fY1 = hend - hcenter;
					fFactorA = fY1 * fX1;
					fFactorB = fY1 * fX0;
					fFactorC = fY0 * fX1;
					fFactorD = fY0 * fX0;

					top_data[pool_index] = batch_data[hstart * fw + wstart] * fFactorA
						+ batch_data[hstart * fw + wend] * fFactorB
						+ batch_data[hend * fw + wstart] * fFactorC
						+ batch_data[hend * fw + wend] * fFactorD;
					argmax_data[4 * pool_index + 0] = hstart * fw + wstart;
					argmax_data[4 * pool_index + 1] = hstart * fw + wend;
					argmax_data[4 * pool_index + 2] = hend * fw + wstart;
					argmax_data[4 * pool_index + 3] = hend * fw + wend;
					w_data[4 * pool_index + 0] = fFactorA;
					w_data[4 * pool_index + 1] = fFactorB;
					w_data[4 * pool_index + 2] = fFactorC;
					w_data[4 * pool_index + 3] = fFactorD;
				}
			}
			// Increment all data pointers by one channel
			batch_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
			argmax_data += idx_.offset(0, 1);
			w_data += coeff_.offset(0, 1);
		}
		// Increment ROI data pointer
		bottom_rois += b1_width;
	}
}

template <typename Dtype>
void RoiAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// if (propagate_down[1]) {
	// 	LOG(FATAL) << this->type()
	// 		<< " Layer cannot backpropagate to roi inputs.";
	// }
	if (!propagate_down[0]) {
		return;
	}
	const int b1_width = bottom[1]->width();
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
	const int* argmax_data = idx_.cpu_data();
	const int num_rois = top[0]->num();
	const Dtype* w_data = coeff_.cpu_data();
	int argmax_index[4];
	Dtype w[4];
	int w_num = 4;

	const int channels = bottom[0]->channels();
	const int fw = bottom[0]->width();
	const int fh = bottom[0]->height();
	// Accumulate gradient over all ROIs
	for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
		int roi_batch_ind = bottom_rois[roi_n * b1_width];
		// Accumulate gradients over each bin in this ROI
		for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < roiResizedHeight_; ++ph) {
				for (int pw = 0; pw < roiResizedWidth_; ++pw) {
					int offset_top = ((roi_n * channels + c) * roiResizedHeight_ + ph)
						* roiResizedWidth_ + pw;
					for (int index = 0; index < w_num; ++index) {
						argmax_index[index] = argmax_data[offset_top * w_num + index];
						w[index] = w_data[offset_top * w_num + index];
					}
					for (int index = 0; index < w_num; ++index) {
						if (argmax_index[index] >= 0) {
							int offset_bottom = (roi_batch_ind * channels + c) * fh
								* fw + argmax_index[index];
							bottom_diff[offset_bottom] += top_diff[offset_top] * w[index];
						}
					}
				}
			}
		}
	}
}


#ifdef CPU_ONLY
	STUB_GPU(RoiAlignLayer);
#endif

INSTANTIATE_CLASS(RoiAlignLayer);
REGISTER_LAYER_CLASS(RoiAlign);

}  // namespace caffe
