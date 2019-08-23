#include "caffe/layers/imresize_layer.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void ImResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const ImResizeParameter& imresize_param = this->layer_param_.imresize_param();
	targetSpatialWidth_ = imresize_param.target_spatial_width();
	targetSpatialHeight_ = imresize_param.target_spatial_height();
	start_scale_ = imresize_param.start_scale();
	scale_gap_ = imresize_param.scale_gap();
	factor_ = imresize_param.factor();
}

template <typename Dtype>
void ImResizeLayer<Dtype>::setTargetDimenions(int tw, int th){
	targetSpatialWidth_ = tw;
	targetSpatialHeight_ = th;
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	std::vector<int> bottom_shape = bottom[0]->shape();
	std::vector<int> top_shape(bottom_shape);

	if(factor_ > 0) {
		top_shape[3] = top_shape[3] * factor_;
		top_shape[2] = top_shape[2] * factor_;
		setTargetDimenions(top_shape[3], top_shape[2]);
	} else {
		top_shape[3] = targetSpatialWidth_;
		top_shape[2] = targetSpatialHeight_;
	}
	top_shape[0] = 1;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	//Do not use this method.
	for(int n = 0; n < num; n++){
		for(int c = 0; c < channel; c++){
			// fill src
			cv::Mat src(oriSpatialHeight, oriSpatialWidth, CV_32FC1);
			const int src_offset2 = oriSpatialHeight * oriSpatialWidth;
			const int src_offset3 = src_offset2 * channel;
			const Dtype* src_pointer = bottom[0]->cpu_data();
			for (int y = 0; y < oriSpatialHeight; y++){
				for (int x = 0; x < oriSpatialWidth; x++){
				    src.at<Dtype>(y,x) = src_pointer[n*src_offset3 + c*src_offset2 + y*oriSpatialWidth + x];
				}
			}
			//resize
			cv::Mat dst(targetSpatialHeight_, targetSpatialWidth_, CV_32FC1);
			cv::resize(src, dst, dst.size(), 0, 0, CV_INTER_LINEAR);
			//fill top
			const int dst_offset2 = targetSpatialHeight_ * targetSpatialWidth_;
			const int dst_offset3 = dst_offset2 * channel;
			Dtype* dst_pointer = top[0]->mutable_cpu_data();
			for (int y = 0; y < targetSpatialHeight_; y++){
				for (int x = 0; x < targetSpatialWidth_; x++){
				    dst_pointer[n*dst_offset3 + c*dst_offset2 + y*targetSpatialWidth_ + x] = dst.at<Dtype>(y,x);
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ImResizeLayer);
#endif

INSTANTIATE_CLASS(ImResizeLayer);
REGISTER_LAYER_CLASS(ImResize);

}  // namespace caffe
