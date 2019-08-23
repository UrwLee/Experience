#include "caffe/layers/imresize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype* out, const Dtype v0, const Dtype v1, const Dtype v2, const Dtype v3, const Dtype dx) {
    // Dtype a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // Dtype b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // Dtype c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    *out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}

template <typename Dtype>
inline __device__ void linear_interpolation(Dtype* out, const Dtype vlt, const Dtype vrt, const Dtype vlb, const Dtype vrb, const Dtype dx, const Dtype dy) {
    // Dtype out = (1-dx)*(1-dy)*vlt + dx*(1-dy)*vrt + (1-dx)*dy*vlb + dx*dy*vrb
    *out = (1-dx)*(1-dy)*vlt + dx*(1-dy)*vrt + (1-dx)*dy*vlb + dx*dy*vrb;
}

template <typename Dtype>
__global__ void imresize_cubic_kernel(const Dtype* src_ptr, Dtype* dst_pointer, const int src_offset, const int num, const Dtype scale_gap,
									                    const Dtype start_scale, const int oriSpatialWidth, const int oriSpatialHeight, const int tw, const int th) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < tw) && (y < th)) {
    Dtype d_temp = 0;
    Dtype sum = 0;
    // 遍历所有的scale (num)
    for(int n = 0; n < num; n++){
      // 获取pad和原图尺寸
      const int padw = floor((Dtype)oriSpatialWidth * (1-start_scale + n * scale_gap) / 2.);
      const int padh = floor((Dtype)oriSpatialHeight * (1-start_scale + n * scale_gap) / 2.);
      const int ow = oriSpatialWidth - 2 * padw;
      const int oh = oriSpatialHeight - 2 * padh;
      // 输入数据指针
      const Dtype* src_pointer = src_ptr + n * src_offset;

      // 获取在原图上的坐标
      const Dtype offset_x = tw / Dtype(ow) / 2 - 0.5;
      const Dtype offset_y = th / Dtype(oh)/ 2 - 0.5;
      const Dtype x_on_ori = (x - offset_x) * (Dtype(ow) / tw);
      const Dtype y_on_ori = (y - offset_y) * (Dtype(oh) / th);

      // 获取附近的坐标点 [4x4]
      int x_nei[4];
      x_nei[1] = int(x_on_ori + 1e-5);
      x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
      x_nei[0] = ((x_nei[1] < 1) ? x_nei[1] : (x_nei[1] - 1)) + padw;
      x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
      x_nei[3] = ((x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1)) + padw;
      const Dtype dx = x_on_ori - x_nei[1];
      x_nei[1] = x_nei[1] + padw;
      x_nei[2] = x_nei[2] + padw;

      int y_nei[4];
      y_nei[1] = int(y_on_ori + 1e-5);
      y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
      y_nei[0] = ((y_nei[1] < 1) ? y_nei[1] : (y_nei[1] - 1)) + padh;
      y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
      y_nei[3] = ((y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1)) + padh;
      const Dtype dy = y_on_ori - y_nei[1];
      y_nei[1] = y_nei[1] + padh;
      y_nei[2] = y_nei[2] + padh;

      // 按行插值
      Dtype temp[4];
      for(int i = 0; i < 4; i++){
        cubic_interpolation(&temp[i], src_pointer[y_nei[i]*(ow+2*padw) + x_nei[0]],
                                      src_pointer[y_nei[i]*(ow+2*padw) + x_nei[1]],
                                      src_pointer[y_nei[i]*(ow+2*padw) + x_nei[2]],
                                      src_pointer[y_nei[i]*(ow+2*padw) + x_nei[3]], dx);
      }
      // 按列插值
      cubic_interpolation(&d_temp, temp[0], temp[1], temp[2], temp[3], dy);
      sum = sum + d_temp;
    }
    // 输出平均值
    dst_pointer[y*tw+x] = sum / num;
  }
}

template <typename Dtype>
__global__ void imresize_linear_kernel(const Dtype* src_ptr, Dtype* dst_pointer, const int src_offset, const int num, const Dtype scale_gap, const Dtype start_scale, const int oriSpatialWidth, const int oriSpatialHeight, const int tw, const int th){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < tw) && (y < th)) {
    Dtype d_temp = 0;
    Dtype sum = 0;
    // 遍历所有的scale (num)
    for(int n = 0; n < num; n++){
      // 获取pad和原图尺寸
      const int padw = floor((Dtype)oriSpatialWidth * (1-start_scale + n * scale_gap) / 2.);
      const int padh = floor((Dtype)oriSpatialHeight * (1-start_scale + n * scale_gap) / 2.);
      const int ow = oriSpatialWidth - 2 * padw;
      const int oh = oriSpatialHeight - 2 * padh;
      // 输入数据指针
      const Dtype* src_pointer = src_ptr + n * src_offset;

      // 获取在原图上的坐标
      const Dtype offset_x = tw / Dtype(ow) / 2 - 0.5;
      const Dtype offset_y = th / Dtype(oh)/ 2 - 0.5;
      const Dtype x_on_ori = (x - offset_x) * (Dtype(ow) / tw);
      const Dtype y_on_ori = (y - offset_y) * (Dtype(oh) / th);

      // 获取附近的坐标点 [2x2]
      int x_nei[2];
      x_nei[0] = int(x_on_ori + 1e-5);
      x_nei[0] = (x_nei[0] < 0) ? 0 : x_nei[0];
      x_nei[1] = (x_nei[0] + 1 >= ow) ? (ow - 1) : (x_nei[0] + 1);
      const Dtype dx = x_on_ori - x_nei[0];
      x_nei[0] = x_nei[0] + padw;
      x_nei[1] = x_nei[1] + padw;

      int y_nei[2];
      y_nei[0] = int(y_on_ori + 1e-5);
      y_nei[0] = (y_nei[0] < 0) ? 0 : y_nei[0];
      y_nei[1] = (y_nei[0] + 1 >= oh) ? (oh - 1) : (y_nei[0] + 1);
      const Dtype dy = y_on_ori - y_nei[0];
      y_nei[0] = y_nei[0] + padh;
      y_nei[1] = y_nei[1] + padh;

      linear_interpolation(&d_temp,   src_pointer[y_nei[0]*(ow+2*padw) + x_nei[0]],
                                      src_pointer[y_nei[0]*(ow+2*padw) + x_nei[1]],
                                      src_pointer[y_nei[1]*(ow+2*padw) + x_nei[0]],
                                      src_pointer[y_nei[1]*(ow+2*padw) + x_nei[1]], dx, dy);
      sum = sum + d_temp;
    }
    // 输出平均值
    dst_pointer[y*tw+x] = sum / num;
  }
}


template <typename Dtype>
void ImResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
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
		//imresize_cubic_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
    //  src_pointer + c * offset_src,
    //  dst_pointer + c * offset_dst,
		//	channel * offset_src,
    //  num, scale_gap_, start_scale_,
		//	oriSpatialWidth, oriSpatialHeight,
		//	targetSpatialWidth_, targetSpatialHeight_);
    imresize_linear_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      src_pointer + c * offset_src,
      dst_pointer + c * offset_dst,
      channel * offset_src,
      num, scale_gap_, start_scale_,
      oriSpatialWidth, oriSpatialHeight,
      targetSpatialWidth_, targetSpatialHeight_);
	}
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ImResizeLayer);

} // namespace caffe
