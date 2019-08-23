#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype* out, const Dtype v0, const Dtype v1, const Dtype v2, const Dtype v3, const Dtype dx) {
    *out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}

template <typename Dtype>
inline __device__ void linear_interpolation(Dtype* out, const Dtype vlt, const Dtype vrt, const Dtype vlb, const Dtype vrb, const Dtype dx, const Dtype dy) {
    *out = (1-dx)*(1-dy)*vlt + dx*(1-dy)*vrt + (1-dx)*dy*vlb + dx*dy*vrb;
}

template <typename Dtype>
__global__ void resize_cubic_kernel(const Dtype* src_ptr, Dtype* dst_ptr, const int src_offset, const int dst_offset,
																		const int num, const int ow, const int oh, const int tw, const int th) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < tw) && (y < th)) {
    Dtype d_temp = 0;
    for(int n = 0; n < num; n++){
      // 数据指针
      const Dtype* src_pointer = src_ptr + n * src_offset;
			Dtype* dst_pointer = dst_ptr + n * dst_offset;
      // 获取在原图上的坐标
      const Dtype offset_x = tw / Dtype(ow) / 2 - 0.5;
      const Dtype offset_y = th / Dtype(oh)/ 2 - 0.5;
      const Dtype x_on_ori = (x - offset_x) * (Dtype(ow) / tw);
      const Dtype y_on_ori = (y - offset_y) * (Dtype(oh) / th);

      // 获取附近的坐标点 [4x4]
      int x_nei[4];
      x_nei[1] = int(x_on_ori + 1e-5);
      x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
      x_nei[0] = ((x_nei[1] < 1) ? x_nei[1] : (x_nei[1] - 1));
      x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
      x_nei[3] = ((x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1));
      const Dtype dx = x_on_ori - x_nei[1];

      int y_nei[4];
      y_nei[1] = int(y_on_ori + 1e-5);
      y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
      y_nei[0] = ((y_nei[1] < 1) ? y_nei[1] : (y_nei[1] - 1));
      y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
      y_nei[3] = ((y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1));
      const Dtype dy = y_on_ori - y_nei[1];
      // 按行插值
      Dtype temp[4];
      for(int i = 0; i < 4; i++){
        cubic_interpolation(&temp[i], src_pointer[y_nei[i]*ow + x_nei[0]],
                                      src_pointer[y_nei[i]*ow + x_nei[1]],
                                      src_pointer[y_nei[i]*ow + x_nei[2]],
                                      src_pointer[y_nei[i]*ow + x_nei[3]], dx);
      }
      // 按列插值
      cubic_interpolation(&d_temp, temp[0], temp[1], temp[2], temp[3], dy);
			dst_pointer[y * tw + x] = d_temp;
    }
  }
}

template <typename Dtype>
__global__ void resize_linear_kernel(const Dtype* src_ptr, Dtype* dst_ptr, const int src_offset,
																		 const int dst_offset, const int num, const int ow, const int oh,
																		 const int tw, const int th){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < tw) && (y < th)) {
    Dtype d_temp = 0;
    for(int n = 0; n < num; n++){
      // 数据指针
      const Dtype* src_pointer = src_ptr + n * src_offset;
			Dtype* dst_pointer = dst_ptr + n * dst_offset;
      // 获取在原图上的坐标
      const Dtype offset_x = tw / Dtype(ow) / 2 - 0.5;
      const Dtype offset_y = th / Dtype(oh)/ 2 - 0.5;
      const Dtype x_on_ori = (x - offset_x) * (Dtype(ow) / tw);
      const Dtype y_on_ori = (y - offset_y) * (Dtype(oh) / th);

      int x_nei[2];
      x_nei[0] = int(x_on_ori + 1e-5);
      x_nei[0] = (x_nei[0] < 0) ? 0 : x_nei[0];
      x_nei[1] = (x_nei[0] + 1 >= ow) ? (ow - 1) : (x_nei[0] + 1);
      const Dtype dx = x_on_ori - x_nei[0];

      int y_nei[2];
      y_nei[0] = int(y_on_ori + 1e-5);
      y_nei[0] = (y_nei[0] < 0) ? 0 : y_nei[0];
      y_nei[1] = (y_nei[0] + 1 >= oh) ? (oh - 1) : (y_nei[0] + 1);
      const Dtype dy = y_on_ori - y_nei[0];
      linear_interpolation(&d_temp,   src_pointer[y_nei[0]*ow + x_nei[0]],
                                      src_pointer[y_nei[0]*ow + x_nei[1]],
                                      src_pointer[y_nei[1]*ow + x_nei[0]],
                                      src_pointer[y_nei[1]*ow + x_nei[1]], dx, dy);
			dst_pointer[y * tw + x] = d_temp;
    }
  }
}

template <typename Dtype>
void ResizeBlobLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* src_pointer = bottom[0]->gpu_data();
	Dtype* dst_pointer = top[0]->mutable_gpu_data();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
	const dim3 BlocksSize(CAFFE_UPDIV(targetSpatialWidth_, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(targetSpatialHeight_, CAFFE_CUDA_NUM_THREADS_1D));
	const int offset_src = oriSpatialHeight * oriSpatialWidth;
	const int offset_dst = targetSpatialWidth_ * targetSpatialHeight_;

	for(int c = 0; c < channel; c++) {
    resize_linear_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      src_pointer + c * offset_src,
      dst_pointer + c * offset_dst,
      channel * offset_src, channel * offset_dst,
      num, oriSpatialWidth, oriSpatialHeight,
      targetSpatialWidth_, targetSpatialHeight_);
	}
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ResizeBlobLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeBlobLayer);

} // namespace caffe
