#include "caffe/layers/nms_layer.hpp"
#include <vector>

namespace caffe {

template <typename Dtype>
void NmsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const NmsParameter& nms_param = this->layer_param_.nms_param();
	threshold_ = nms_param.threshold();
	num_parts_ = nms_param.num_parts();
	max_peaks_ = nms_param.max_peaks();
}

template <typename Dtype>
void NmsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	std::vector<int> bottom_shape = bottom[0]->shape();
	std::vector<int> top_shape(bottom_shape);
	// x/y/score
	top_shape[3] = 3;
	// 0 -> num of peaks
	// 1:max_peaks_ -> coords and score
	top_shape[2] = max_peaks_+1;
	// each part
	top_shape[1] = num_parts_;

	CHECK_EQ(top_shape[0], 1) << "num() must be 1.";

	top[0]->Reshape(top_shape);
	workspace.Reshape(bottom_shape);
}

template <typename Dtype>
void NmsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->shape(0);
	CHECK_EQ(num, 1) << "num() must be 1.";
	const int channel = bottom[0]->shape(1);
	CHECK_GE(channel, num_parts_) << "channels() must be greater than num_parts.";
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	const int offset2 = oriSpatialHeight * oriSpatialWidth;
	const int offset2_dst = (max_peaks_+1)*3;

	// Do not use this method.
	for(int c = 0; c < num_parts_; c++){
		int peakCount = 0;
		const Dtype* src_pointer = bottom[0]->cpu_data() + c * offset2;
		Dtype* dst_pointer = top[0]->mutable_cpu_data() + c * offset2_dst;
		for (int y = 0; y < oriSpatialHeight; y++) {
			for (int x = 0; x < oriSpatialWidth; x++) {
					if (peakCount >= max_peaks_) continue;
					if (x < 1 || x >= (oriSpatialWidth - 1)) continue;
					if (y < 1 || y >= (oriSpatialHeight - 1)) continue;
			    const Dtype value = src_pointer[y*oriSpatialWidth + x];
			    if(value < threshold_) continue;
					const Dtype top    = src_pointer[(y-1)*oriSpatialWidth + x];
					const Dtype bottom = src_pointer[(y+1)*oriSpatialWidth + x];
					const Dtype left   = src_pointer[y*oriSpatialWidth + (x-1)];
					const Dtype right  = src_pointer[y*oriSpatialWidth + (x+1)];
					const Dtype top_left = src_pointer[(y-1)*oriSpatialWidth + x-1];
					const Dtype top_right = src_pointer[(y-1)*oriSpatialWidth + x+1];
					const Dtype bottom_left = src_pointer[(y+1)*oriSpatialWidth + x-1];
					const Dtype bottom_right = src_pointer[(y+1)*oriSpatialWidth + x+1];
					if(value > top && value > bottom && value > left && value > right && value > top_left
						&& value > bottom_left && value > bottom_right && value > top_right) {
							Dtype x_acc = 0;
							Dtype y_acc = 0;
							Dtype score_acc = 0;
							for (int dy = -2; dy < 3; dy++) {
								if ((y+dy)>=0 && (y+dy)<oriSpatialHeight) {
									for (int dx = -2; dx < 3; dx++) {
										if ((x+dx)>=0 && (x+dx)<oriSpatialWidth) {
											const Dtype score = src_pointer[(y+dy)*oriSpatialWidth + x+dx];
											const Dtype x_c = x+dx;
											const Dtype y_c = y+dy;
											if (score>threshold_) {
												x_acc += x_c*score;
												y_acc += y_c*score;
												score_acc += score;
											}
										}
									}
								}
							}
							const int output_index = (peakCount + 1) * 3;
							dst_pointer[output_index] = x_acc/score_acc;
							dst_pointer[output_index + 1] = y_acc/score_acc;
							dst_pointer[output_index + 2] = value;
							peakCount++;
					}
			}
		}
		dst_pointer[0] = peakCount;
	}
}

#ifdef CPU_ONLY
STUB_GPU(NmsLayer);
#endif

INSTANTIATE_CLASS(NmsLayer);
REGISTER_LAYER_CLASS(Nms);

} // namespace caffe
