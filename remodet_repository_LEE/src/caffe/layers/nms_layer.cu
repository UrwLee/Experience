#include "caffe/layers/nms_layer.hpp"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

namespace caffe {

template <typename Dtype>
__global__ void nms_register_kernel(const Dtype* src_pointer, int* workspace, const int w, const int h, const Dtype threshold) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x<w) && (y<h)) {
		if(x>0 && x<(w-1) && y>0 && y<(h-1)) {
			const Dtype value = src_pointer[y*w + x];
			if(value > threshold) {
				const Dtype top    = src_pointer[(y-1)*w + x];
				const Dtype bottom = src_pointer[(y+1)*w + x];
				const Dtype left   = src_pointer[y*w + (x-1)];
				const Dtype right  = src_pointer[y*w + (x+1)];
				const Dtype top_left = src_pointer[(y-1)*w + x-1];
				const Dtype top_right = src_pointer[(y-1)*w + x+1];
				const Dtype bottom_left = src_pointer[(y+1)*w + x-1];
				const Dtype bottom_right = src_pointer[(y+1)*w + x+1];
				if(value > top && value > bottom && value > left && value > right && value > top_left
					&& value > bottom_left && value > bottom_right && value > top_right) {
					workspace[y*w + x] = 1;
				} else {
					workspace[y*w + x] = 0;
				}
			} else {
				workspace[y*w + x] = 0;
			}
		}	else {
			workspace[y*w + x] = 0;
		}
	}
}

template <typename Dtype>
__global__ void writeResultKernel(const int nthreads, const int* input, const Dtype* src_pointer, Dtype* output, const int width, const int height, const int max_peaks){
	CUDA_KERNEL_LOOP(i, nthreads) {
		// 比较, 最后一个元素用于统计检测到多少个peaks
		if(i != nthreads - 1) {
			// 检测到跳跃点,说明此处存在一个peak
      if(input[i] != input[i + 1]) {
					// peak编号
          const int peak_index = input[i];
					// 位置索引;w/h
          const int peak_loc = i;
          const int peak_loc_x = peak_loc % width;
          const int peak_loc_y = peak_loc / width;

					// 超过设定max-peaks,不处理,跳过
          if(peak_index < max_peaks) {
							Dtype x_acc = 0;
							Dtype y_acc = 0;
							Dtype score_acc = 0;
							// 统计位置附近7x7区域内的均值
							for (int dy = -3; dy < 4; dy++) {
								if ((peak_loc_y+dy)>=0 && (peak_loc_y+dy)<height) {
									for (int dx = -3; dx < 4; dx++) {
										if ((peak_loc_x+dx)>=0 && (peak_loc_x+dx)<width) {
											const Dtype score = src_pointer[(peak_loc_y+dy)*width + peak_loc_x+dx];
											const Dtype x = peak_loc_x+dx;
											const Dtype y = peak_loc_y+dy;
											if (score>0) {
												x_acc += x*score;
												y_acc += y*score;
												score_acc += score;
											}
										}
									}
								}
							}
							// 输出
							const int output_index = (peak_index + 1) * 3;
							output[output_index] = x_acc/score_acc/width;
              output[output_index + 1] = y_acc/score_acc/height;
              output[output_index + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
         }
      }
    } else {
      	output[0] = input[i];
    }
	}
}

template <typename Dtype>
void NmsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->shape(0);
	CHECK_EQ(num, 1) << "num() should be 1.";
	const int height = bottom[0]->shape(2);
	const int width = bottom[0]->shape(3);
	const int offset = height * width;
	const int offset_dst = (max_peaks_+1) * 3;

	for(int c = 0; c < num_parts_; c++) {
		int* w_pointer1 = workspace.mutable_gpu_data() + c * offset;
		const Dtype* src = bottom[0]->gpu_data() + c * offset;
		Dtype* dst = top[0]->mutable_gpu_data() + c * offset_dst;

		const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
		const dim3 BlocksSize(CAFFE_UPDIV(width, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(height, CAFFE_CUDA_NUM_THREADS_1D));

		nms_register_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(src, w_pointer1, width, height, threshold_);

		thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(w_pointer1);
		thrust::exclusive_scan(dev_ptr, dev_ptr + offset, dev_ptr);

		writeResultKernel<Dtype><<<CAFFE_GET_BLOCKS(offset), CAFFE_CUDA_NUM_THREADS>>>(
			offset, w_pointer1, src, dst, width, height, max_peaks_);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(NmsLayer);

} // namespace caffe
