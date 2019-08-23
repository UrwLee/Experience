#include "caffe/layers/maximum_layer.hpp"
#include <limits>

using namespace std;

namespace caffe {

template <typename Dtype>
void MaximumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(bottom_shape);
	//x, y, and value
	top_shape[3] = 3;
	top_shape[2] = 1;
	// [n,c,1,3]
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MaximumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	const int offset2 = oriSpatialHeight * oriSpatialWidth;
	const int offset3 = offset2 * channel;

	const int offset2_dst = 3;
	const int offset3_dst = offset2_dst * channel;

	for(int n = 0; n < num; n++){
		for(int c = 0; c < channel; c++){
			const Dtype* src_pointer_channel = bottom[0]->cpu_data() + n*offset3 + c*offset2;
			Dtype value = std::numeric_limits<Dtype>::min();
			int arg_x = 0;
			int arg_y = 0;

			for (int y = 0; y < oriSpatialHeight; y++){
				for (int x = 0; x < oriSpatialWidth; x++){
				    const Dtype value_cur = src_pointer_channel[y*oriSpatialWidth + x];
				    if(value_cur > value){
				    	value = value_cur;
				    	arg_x = x;
				    	arg_y = y;
				    }
				}
			}
			Dtype* dst_point_channel = top[0]->mutable_cpu_data() + n*offset3_dst + c*offset2_dst;
			dst_point_channel[0] = arg_x;
			dst_point_channel[1] = arg_y;
			dst_point_channel[2] = value;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(MaximumLayer);
#endif

INSTANTIATE_CLASS(MaximumLayer);
REGISTER_LAYER_CLASS(Maximum);

}  // namespace caffe
