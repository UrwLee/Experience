#include "caffe/layers/roi_resize_layer.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void RoiResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const RoiResizeParameter& roi_resize_param = this->layer_param_.roi_resize_param();
	targetSpatialWidth_ = roi_resize_param.target_spatial_width();
	targetSpatialHeight_ = roi_resize_param.target_spatial_height();
}

template <typename Dtype>
void RoiResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), targetSpatialHeight_, targetSpatialWidth_);
}

template <typename Dtype>
void RoiResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bbox = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	// 归一化坐标
	Dtype x1_ = bbox[0];
	Dtype y1_ = bbox[1];
	Dtype x2_ = bbox[2];
	Dtype y2_ = bbox[3];
	// 归一化长宽
	Dtype nw = std::max(x2_-x1_, Dtype(0));
	Dtype nh = std::max(y2_-y1_, Dtype(0));
	// 实际长宽
	Dtype ow = nw * Dtype(oriSpatialWidth);
	Dtype oh = nh * Dtype(oriSpatialHeight);
	// ROI 变换比
	Dtype w_scale = ow / (Dtype)targetSpatialWidth_;
	Dtype h_scale = oh / (Dtype)targetSpatialHeight_;
	// ROI左上角坐标
	Dtype xmin = x1_ * Dtype(oriSpatialWidth);
	Dtype ymin = y1_ * Dtype(oriSpatialHeight);
	for (int n = 0; n < num; n++){
		for (int c = 0; c < channel; c++){
			for (int y = 0; y < targetSpatialHeight_; ++y) {
				for (int x = 0; x < targetSpatialWidth_; ++x) {
					int out_idx = ((n*channel+c)*targetSpatialHeight_+y)*targetSpatialWidth_+x;
					Dtype offset_x = (Dtype)targetSpatialWidth_ / oriSpatialWidth / 2 - 0.5;
			    Dtype offset_y = (Dtype)targetSpatialHeight_ / oriSpatialHeight / 2 - 0.5;
					Dtype x_on_map = (x - offset_x) * w_scale + xmin;
			    Dtype y_on_map = (y - offset_y) * h_scale + ymin;
					if (x_on_map >= 0 && x_on_map < oriSpatialWidth && y_on_map >= 0 && y_on_map < oriSpatialHeight) {
						// LINEAR
						int x_nei[2];
						x_nei[0] = int(x_on_map);
						x_nei[0] = (x_nei[0] < 0) ? 0 : x_nei[0];
						x_nei[1] = (x_nei[0] + 1 >= oriSpatialWidth) ? (oriSpatialWidth - 1) : (x_nei[0] + 1);
						const Dtype dx = x_on_map - x_nei[0];
						int y_nei[2];
						y_nei[0] = int(y_on_map);
						y_nei[0] = (y_nei[0] < 0) ? 0 : y_nei[0];
						y_nei[1] = (y_nei[0] + 1 >= oriSpatialHeight) ? (oriSpatialHeight - 1) : (y_nei[0] + 1);
						const Dtype dy = y_on_map - y_nei[0];
						Dtype vlt = bottom_data[((n*channel+c)*oriSpatialHeight+y_nei[0])*oriSpatialWidth+x_nei[0]];
						Dtype vrt = bottom_data[((n*channel+c)*oriSpatialHeight+y_nei[0])*oriSpatialWidth+x_nei[1]];
						Dtype vlb = bottom_data[((n*channel+c)*oriSpatialHeight+y_nei[1])*oriSpatialWidth+x_nei[0]];
						Dtype vrb = bottom_data[((n*channel+c)*oriSpatialHeight+y_nei[1])*oriSpatialWidth+x_nei[1]];
						top_data[out_idx] = (1-dx)*(1-dy)*vlt + dx*(1-dy)*vrt + (1-dx)*dy*vlb + dx*dy*vrb;
					} else {
						top_data[out_idx] = 0;
					}
				}
			}
		}
	}
}

template <typename Dtype>
void RoiResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}

#ifdef CPU_ONLY
STUB_GPU(RoiResizeLayer);
#endif

INSTANTIATE_CLASS(RoiResizeLayer);
REGISTER_LAYER_CLASS(RoiResize);

}  // namespace caffe
