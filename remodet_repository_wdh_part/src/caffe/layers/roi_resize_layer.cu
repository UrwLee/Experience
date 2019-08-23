#include "caffe/layers/roi_resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline __device__ Dtype min(Dtype a, Dtype b) {
  return (a < b) ? a : b;
}

template <typename Dtype>
inline __device__ Dtype max(Dtype a, Dtype b) {
  return (a > b) ? a : b;
}

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
__global__ void resize_cubic_kernel(const int nthreads, const Dtype* src_ptr,
                                    const Dtype* bbox, Dtype* dst_ptr,
																		const int num, const int channels,
                                    const int oriSpatialWidth, const int oriSpatialHeight,
                                    const int tarSpatialWidth, const int tarSpatialHeight) {
  CUDA_KERNEL_LOOP(index,nthreads) {
    // get (n,c,y,x)
    int x = index % tarSpatialWidth;
    int y = (index / tarSpatialWidth) % tarSpatialHeight;
    int c = (index / tarSpatialWidth / tarSpatialHeight) % channels;
    int n = (index / tarSpatialWidth / tarSpatialHeight / channels) % num;
    // get box
    Dtype x1_ = bbox[0];
    Dtype y1_ = bbox[1];
    Dtype x2_ = bbox[2];
    Dtype y2_ = bbox[3];
    // get Normalized w&h
    Dtype nw = max(x2_-x1_, Dtype(0));
    Dtype nh = max(y2_-y1_, Dtype(0));
    // get real w&h
    Dtype ow = nw * Dtype(oriSpatialWidth);
    Dtype oh = nh * Dtype(oriSpatialHeight);
    // get transfer ratio
    Dtype w_scale = ow / (Dtype)tarSpatialWidth;
    Dtype h_scale = oh / (Dtype)tarSpatialHeight;
    // get offset
    Dtype offset_x = (Dtype)tarSpatialWidth / oriSpatialWidth / 2 - 0.5;
    Dtype offset_y = (Dtype)tarSpatialHeight / oriSpatialHeight / 2 - 0.5;
    // get ROI top-left point on the fMap
    Dtype xmin = x1_ * Dtype(oriSpatialWidth);
    Dtype ymin = y1_ * Dtype(oriSpatialHeight);
    // get point on fMap
    Dtype x_on_map = (x - offset_x) * w_scale + xmin;
    Dtype y_on_map = (y - offset_y) * h_scale + ymin;
    Dtype v;
    if (x_on_map >= 0 && x_on_map < oriSpatialWidth && y_on_map >= 0 && y_on_map < oriSpatialHeight){
      int x_nei[4];
      x_nei[1] = int(x_on_map);
      x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
      x_nei[0] = ((x_nei[1] < 1) ? x_nei[1] : (x_nei[1] - 1));
      x_nei[2] = (x_nei[1] + 1 >= oriSpatialWidth) ? (oriSpatialWidth - 1) : (x_nei[1] + 1);
      x_nei[3] = ((x_nei[2] + 1 >= oriSpatialWidth) ? (oriSpatialWidth - 1) : (x_nei[2] + 1));
      const Dtype dx = x_on_map - x_nei[1];
      int y_nei[4];
      y_nei[1] = int(y_on_map);
      y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
      y_nei[0] = ((y_nei[1] < 1) ? y_nei[1] : (y_nei[1] - 1));
      y_nei[2] = (y_nei[1] + 1 >= oriSpatialHeight) ? (oriSpatialHeight - 1) : (y_nei[1] + 1);
      y_nei[3] = ((y_nei[2] + 1 >= oriSpatialHeight) ? (oriSpatialHeight - 1) : (y_nei[2] + 1));
      const Dtype dy = y_on_map - y_nei[1];
      Dtype temp[4];
      for(int i = 0; i < 4; i++){
        cubic_interpolation(&temp[i], src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[i])*oriSpatialWidth+x_nei[0]],
                                      src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[i])*oriSpatialWidth+x_nei[1]],
                                      src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[i])*oriSpatialWidth+x_nei[2]],
                                      src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[i])*oriSpatialWidth+x_nei[3]], dx);
      }
      cubic_interpolation(&v, temp[0], temp[1], temp[2], temp[3], dy);
      dst_ptr[index] = v;
    } else {
      dst_ptr[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void resize_linear_kernel(const int nthreads, const Dtype* src_ptr,
                                     const Dtype* bbox, Dtype* dst_ptr,
																		 const int num, const int channels,
                                     const int oriSpatialWidth, const int oriSpatialHeight,
                                     const int tarSpatialWidth, const int tarSpatialHeight){
 CUDA_KERNEL_LOOP(index,nthreads) {
   // get (n,c,y,x)
   int x = index % tarSpatialWidth;
   int y = (index / tarSpatialWidth) % tarSpatialHeight;
   int c = (index / tarSpatialWidth / tarSpatialHeight) % channels;
   int n = (index / tarSpatialWidth / tarSpatialHeight / channels) % num;
   // get box
   Dtype x1_ = bbox[0];
   Dtype y1_ = bbox[1];
   Dtype x2_ = bbox[2];
   Dtype y2_ = bbox[3];
   // get Normalized w&h
   Dtype nw = max(x2_-x1_, Dtype(0));
   Dtype nh = max(y2_-y1_, Dtype(0));
   // get real w&h
   Dtype ow = nw * Dtype(oriSpatialWidth);
   Dtype oh = nh * Dtype(oriSpatialHeight);
   // get transfer ratio
   Dtype w_scale = ow / (Dtype)tarSpatialWidth;
   Dtype h_scale = oh / (Dtype)tarSpatialHeight;
   // get offset
   Dtype offset_x = (Dtype)tarSpatialWidth / oriSpatialWidth / 2 - 0.5;
   Dtype offset_y = (Dtype)tarSpatialHeight / oriSpatialHeight / 2 - 0.5;
   // get ROI top-left point on the fMap
   Dtype xmin = x1_ * Dtype(oriSpatialWidth);
   Dtype ymin = y1_ * Dtype(oriSpatialHeight);
   // get point on fMap
   Dtype x_on_map = (x - offset_x) * w_scale + xmin;
   Dtype y_on_map = (y - offset_y) * h_scale + ymin;
   Dtype v;
   if (x_on_map >= 0 && x_on_map < oriSpatialWidth && y_on_map >= 0 && y_on_map < oriSpatialHeight){
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
     linear_interpolation(&v,
                          src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[0])*oriSpatialWidth+x_nei[0]],
                          src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[0])*oriSpatialWidth+x_nei[1]],
                          src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[1])*oriSpatialWidth+x_nei[0]],
                          src_ptr[((n*channels+c)*oriSpatialHeight+y_nei[1])*oriSpatialWidth+x_nei[1]],
                          dx, dy);
     dst_ptr[index] = v;
   } else {
     dst_ptr[index] = 0;
   }
 }
}

template <typename Dtype>
void RoiResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* src_pointer = bottom[0]->gpu_data();
	Dtype* dst_pointer = top[0]->mutable_gpu_data();
  const Dtype* bbox = bottom[1]->gpu_data();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);
  const int count = num * channel * targetSpatialHeight_ * targetSpatialWidth_;
  resize_linear_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, src_pointer, bbox, dst_pointer, num, channel, oriSpatialWidth, oriSpatialHeight,
        targetSpatialWidth_, targetSpatialHeight_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RoiResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(RoiResizeLayer);

} // namespace caffe
