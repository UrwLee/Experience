#include "caffe/layers/resize_layer.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void ResizeBlobLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const ResizeBlobParameter& resize_param = this->layer_param_.resize_layer_param();
	targetSpatialWidth_ = resize_param.target_spatial_width();
	targetSpatialHeight_ = resize_param.target_spatial_height();
}

template <typename Dtype>
void ResizeBlobLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), targetSpatialHeight_, targetSpatialWidth_);
  //max_idx_.Reshape(bottom[0]->num(), bottom[0]->channels(), targetSpatialHeight_, targetSpatialWidth_);
}

template <typename Dtype>
void ResizeBlobLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
			cv::resize(src, dst, dst.size(), 0, 0, CV_INTER_CUBIC);
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

template <typename Dtype>
void ResizeBlobLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}

#ifdef CPU_ONLY
STUB_GPU(ResizeBlobLayer);
#endif

INSTANTIATE_CLASS(ResizeBlobLayer);
REGISTER_LAYER_CLASS(ResizeBlob);

}  // namespace caffe
