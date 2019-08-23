#include <string>
#include <vector>
#include <utility>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <csignal>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/visualize_boxpose_layer.hpp"

#define LIMB_COCO {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17}
#define LIMB_COCO_SEQ {30,31, 36,37, 32,33, 34,35, 38,39, 40,41, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 42,43, 44,45, 48,49, 46,47, 50,51};
#define COLOR_MAPS {255,0,0,255,85,0,255,170,0,255,255,0,170,255,0,85,255,0, \
                    0,255,0,0,255,85,0,255,170,0,255,255,0,170,255,0,85,255, \
                    0,0,255,85,0,255,170,0,255,255,0,255,255,0,170,255,0,85}
namespace caffe {

template <typename Dtype>
inline __device__ Dtype min(Dtype a, Dtype b) {
  return (a < b) ? a : b;
}

template <typename Dtype>
inline __device__ Dtype max(Dtype a, Dtype b) {
  return (a > b) ? a : b;
}

// 五段式取色,等间隔
template <typename Dtype>
inline __device__ void getColor(Dtype* c, Dtype v, Dtype vmin, Dtype vmax)
{
   c[0] = c[1] = c[2] = (Dtype)255; // b, g, r, white
   Dtype dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.125 * dv)) {
      // (0-0.125) 小: ->泛蓝色
      c[0] = (Dtype)256 * (0.5 + (v * 4)); //B: 0.5 ~ 1
      c[1] = c[2] = 0;
   } else if (v < (vmin + 0.375 * dv)) {
      // (0.125-0.375) 较小: ->泛青色
      c[0] = 255;
      c[1] = (Dtype)256 * (v - 0.125) * 4; //G: 0 ~ 1
      c[2] = 0;
   } else if (v < (vmin + 0.625 * dv)) {
      // (0.375-0.625) 较大: -> 泛黄色
      c[0] = (Dtype)256 * (-4 * v + 2.5);  //B: 1 ~ 0
      c[1] = 255;
      c[2] = (Dtype)256 * (4 * (v - 0.375)); //R: 0 ~ 1
   } else if (v < (vmin + 0.875 * dv)) {
      // (0.625-0.875) 很大: -> 泛深红色
      c[0] = 0;
      c[1] = (Dtype)256 * (-4 * v + 3.5);  //G: 1 ~ 0
      c[2] = 255;
   } else {
      // (0.875-1.0) 极大: -> 泛鲜红色
      c[0] = 0;
      c[1] = 0;
      c[2] = (Dtype)256 * (-4 * v + 4.5); //R: 1 ~ 0.5
   }
}

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype* out, Dtype v0, Dtype v1, Dtype v2, Dtype v3, Dtype dx) {
    *out = (-0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3) * dx * dx * dx
         + (v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5 * v0 + 0.5 * v2) * dx
         + v1;
}

// 绘制Vec-Maps
template <typename Dtype>
__global__ void render_vecmaps_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* heatmaps) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  __syncthreads();
  const int color[] = COLOR_MAPS;
  const int nColor = sizeof(color)/(3*sizeof(int));
  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
  const int limb_seq[] = LIMB_COCO_SEQ;
  if(x < w && y < h) {
    Dtype b, g, r;
    Dtype h_inv = (Dtype)nh / (Dtype)h;
    Dtype w_inv = (Dtype)nw / (Dtype)w;

    b = image[y * w + x];
    g = image[w * h + y * w + x];
    r = image[2 * w * h + y * w + x];

    Dtype x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
    Dtype y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

    Dtype c[3];
    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    if(x_on_box >= 0 && x_on_box < nw && y_on_box >=0 && y_on_box < nh){
      for (int l = 0; l < nlimb; ++l) {
        int x_nei = int(x_on_box + 1e-5);
        x_nei = (x_nei < 0) ? 0 : x_nei;
        int y_nei = int(y_on_box + 1e-5);
        y_nei = (y_nei < 0) ? 0 : y_nei;
        int offset_src = limb_seq[2*l] * nw * nh;
        int offset_src2 = limb_seq[2*l+1] * nw * nh;
        Dtype value = heatmaps[offset_src + y_nei * nw + x_nei];
        Dtype value2 = heatmaps[offset_src2 + y_nei * nw + x_nei];
        value = value * value + value2 * value2;
        value = max(min(value, Dtype(1)), Dtype(0));
        c[0] += value * color[(l%nColor)*3+0];
        c[1] += value * color[(l%nColor)*3+1];
        c[2] += value * color[(l%nColor)*3+2];
      }
    }
    c[0] = max(min(c[0], Dtype(255)), Dtype(0));
    c[1] = max(min(c[1], Dtype(255)), Dtype(0));
    c[2] = max(min(c[2], Dtype(255)), Dtype(0));
    Dtype alpha = 0.7;
    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];
    image[y * w + x] = b;
    image[w * h + y * w + x] = g;
    image[2 * w * h + y * w + x] = r;
  }
}

// 绘制heatmaps
template <typename Dtype>
__global__ void render_heatmaps_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* heatmaps) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int color[] = COLOR_MAPS;
  const int nColor = sizeof(color)/(3*sizeof(int));
  __syncthreads();
  if(x < w && y < h){
    Dtype b, g, r;
    Dtype c[3];
    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    Dtype value = 0;
    Dtype h_inv = (Dtype)nh / (Dtype)h;
    Dtype w_inv = (Dtype)nw / (Dtype)w;
    b = image[y * w + x];
    g = image[w * h + y * w + x];
    r = image[2 * w * h + y * w + x];
    for (int part = 0; part < 18; part++) {
      Dtype x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
      Dtype y_on_box = h_inv * y + (0.5 * h_inv - 0.5);
      if(x_on_box >= 0 && x_on_box < nw && y_on_box >= 0 && y_on_box < nh){
        int x_nei = int(x_on_box + 1e-5);
        x_nei = (x_nei < 0) ? 0 : x_nei;
        int y_nei = int(y_on_box + 1e-5);
        y_nei = (y_nei < 0) ? 0 : y_nei;
        int offset_src = part * nw * nh;
        value = heatmaps[offset_src + y_nei * nw + x_nei];
        value = max(min(value, Dtype(1)), Dtype(0));
        c[0] += value * color[(part%nColor)*3+0];
        c[1] += value * color[(part%nColor)*3+1];
        c[2] += value * color[(part%nColor)*3+2];
      }
    }
    c[0] = max(min(c[0], Dtype(255)), Dtype(0));
    c[1] = max(min(c[1], Dtype(255)), Dtype(0));
    c[2] = max(min(c[2], Dtype(255)), Dtype(0));
    Dtype alpha = 0.7;
    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];
    image[y * w + x] = b;
    image[w * h + y * w + x] = g;
    image[2 * w * h + y * w + x] = r;
  }
}

// 绘制pose
template <typename Dtype>
__global__ void render_pose_kernel(Dtype* img_ptr, const int w, const int h, const Dtype* proposals,
                                  const Dtype* vec, const int num_people, const Dtype threshold) {
  // 坐标
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  // 图形网格内的id, 由block内部不同的线程对共享内存进行操作
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;
  // 每个block内部保留poses坐标的副本
  __shared__ float shared_poses[48*18*3];
  __shared__ float2 shared_mins[48];
  __shared__ float2 shared_maxs[48];
  __shared__ float2 shared_scalef[48];
  // 获取副本
  if (global_idx < num_people) {
    int p = global_idx;
    shared_mins[p].x = w;
    shared_mins[p].y = h;
    shared_maxs[p].x = 0;
    shared_maxs[p].y = 0;
    for (int part = 0; part < 18; part++) {
      float x = proposals[p*54+part*3];
      float y = proposals[p*54+part*3+1];
      float z = proposals[p*54+part*3+2];
      x *= w;
      y *= h;
      shared_poses[p*54 + part*3] = x;
      shared_poses[p*54 + part*3+1] = y;
      shared_poses[p*54 + part*3+2] = z;
      if (z>threshold) {
        if (x<shared_mins[p].x) shared_mins[p].x = x;
        if (x>shared_maxs[p].x) shared_maxs[p].x = x;
        if (y<shared_mins[p].y) shared_mins[p].y = y;
        if (y>shared_maxs[p].y) shared_maxs[p].y = y;
      }
    }
    shared_scalef[p].x = shared_maxs[p].x-shared_mins[p].x;
    shared_scalef[p].y = shared_maxs[p].y-shared_mins[p].y;
    shared_scalef[p].x = (shared_scalef[p].x+shared_scalef[p].y)/2.0;
    // 按照200像素等效
    if (shared_scalef[p].x < 200) {
      shared_scalef[p].x = shared_scalef[p].x / 200.;
      if (shared_scalef[p].x < 0.33) shared_scalef[p].x = 0.33;
    } else {
      shared_scalef[p].x = 1.0;
    }
    // 将最大最小分别移除50像素
    shared_maxs[p].x += 50;
    shared_maxs[p].y += 50;
    shared_mins[p].x -= 50;
    shared_mins[p].y -= 50;
  }
  __syncthreads();
  // limb绘制列表
  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
  const int color[] = COLOR_MAPS;
  const int nColor = sizeof(color)/(3*sizeof(int));
  // limb的椭圆宽度定义
  Dtype stickwidth = (Dtype)h / 120.;
  // 绘制每个像素点
  if(x < w && y < h){
    Dtype b, g, r;
    b = img_ptr[y * w + x];
    g = img_ptr[w * h + y * w + x];
    r = img_ptr[2 * w * h + y * w + x];
    // 绘制所有的person
    for(int p = 0; p < num_people; p++) {
      if (x > shared_maxs[p].x || x < shared_mins[p].x
          || y > shared_maxs[p].y || y < shared_mins[p].y) {
        continue;
      }
      for(int l = 0; l < nlimb; l++){
        float b_sqrt = shared_scalef[p].x * shared_scalef[p].x * stickwidth * stickwidth;
        float x_a = (shared_poses[p*54 + limb[2*l]*3]);
        float x_b = (shared_poses[p*54 + limb[2*l+1]*3]);
        float y_a = (shared_poses[p*54 + limb[2*l]*3 + 1]);
        float y_b = (shared_poses[p*54 + limb[2*l+1]*3 + 1]);
        float value_a = shared_poses[p*54 + limb[2*l]*3 + 2];
        float value_b = shared_poses[p*54 + limb[2*l+1]*3 + 2];
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);
          float dist_1 = (x - x_p) * vec[p*nlimb*2 + 2*l] + (y - y_p) * vec[p*nlimb*2 + 2*l+1];
          float dist_2 = (x - x_p) * vec[p*nlimb*2 + 2*l+1] - (y - y_p) * vec[p*nlimb*2 + 2*l];
          if(dist_1 * dist_1 / a_sqrt +  dist_2 * dist_2 / b_sqrt < 1) {
            float3 co;
            co.x = color[(l%nColor)*3+0];
            co.y = color[(l%nColor)*3+1];
            co.z = color[(l%nColor)*3+2];
            b = 0.5 * b + 0.5 * co.z;
            g = 0.5 * g + 0.5 * co.y;
            r = 0.5 * r + 0.5 * co.x;
          }
        }
      }
    }
    img_ptr[y * w + x] = b;
    img_ptr[w * h + y * w + x] = g;
    img_ptr[2 * w * h + y * w + x] = r;
  }
}

template <typename Dtype>
__global__ void render_points_kernel(Dtype* img_ptr, const int w, const int h, const Dtype* proposals,
                                     const int num_people, const Dtype threshold) {
  // 坐标
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int color[] = COLOR_MAPS;
  const int nColor = sizeof(color)/(3*sizeof(int));
  __syncthreads();
  if(x < w && y < h){
    Dtype b, g, r;
    b = img_ptr[y * w + x];
    g = img_ptr[w * h + y * w + x];
    r = img_ptr[2 * w * h + y * w + x];
    float radius = (float)(2 * h) / 200.;
    for(int p = 0; p < num_people; p++) {
      // get scale of this person
      float xmin = w;
      float xmax = 0;
      float ymin = h;
      float ymax = 0;
      for (int part = 0; part < 18; part++) {
        float x = proposals[p*54+part*3] * w;
        float y = proposals[p*54+part*3+1] * h;
        float v = proposals[p*54+part*3+2];
        if (v>threshold) {
          if (x<xmin) xmin = x;
          if (x>xmax) xmax = x;
          if (y<ymin) ymin = y;
          if (y>ymax) ymax = y;
        }
      }
      float x_scale = xmax - xmin;
      float y_scale = ymax - ymin;
      x_scale = (x_scale + y_scale) / 2;
      if (x_scale < 200) {
        x_scale = x_scale / 200.;
        if (x_scale < 0.33) x_scale = 0.33;
      } else {
        x_scale = 1.0;
      }
      // draw this person
      for(int i = 0; i < 18; i++) {
        float local_x = proposals[p*54 + i*3] * w;
        float local_y = proposals[p*54 + i*3 + 1] * h;
        float value = proposals[p*54 + i*3 + 2];
        if(value > threshold) {
          float dist2 = (x - local_x) * (x - local_x) + (y - local_y) * (y - local_y);
          float maxr2 = x_scale * x_scale * radius * radius;
          if(dist2 < maxr2){
            float3 co;
            co.x = color[(i%nColor)*3+0];
            co.y = color[(i%nColor)*3+1];
            co.z = color[(i%nColor)*3+2];
            b = 0.4 * b + 0.6 * co.z;
            g = 0.4 * g + 0.6 * co.y;
            r = 0.4 * r + 0.6 * co.x;
          }
        }
      }//end
    }
    img_ptr[y * w + x] = b;
    img_ptr[w * h + y * w + x] = g;
    img_ptr[2 * w * h + y * w + x] = r;
  }
}

template <typename Dtype>
__global__ void cv_inv_kernel(const int n, const Dtype* image, const int w, const int h, unsigned char* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index % w;
    const int y = (index / w) % h;
    const int c = (index / w / h) % 3;
    int value = (int)(image[index] + 0.5);
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    out[3*(y*w+x)+c] = (unsigned char)value;
  }
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::cv_inv(const Dtype* image, const int w, const int h, unsigned char* out) {
  const int count = 3 * w * h;
  cv_inv_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, image, w, h, out);
}

// 绘制pose
template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* proposals,
                                                  const Dtype* vec, const int num_people, const Dtype threshold) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  if (num_people > 0) {
    render_pose_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      image, w, h, proposals, vec, num_people, threshold);
  }
}

// 绘制points
template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::render_points_gpu(Dtype* image, const int w, const int h, const Dtype* proposals,
                                                     const int num_people, const Dtype threshold) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  if (num_people > 0) {
    render_points_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      image, w, h, proposals, num_people, threshold);
  }
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::render_heatmaps_gpu(Dtype* image, const int w, const int h,
                                          const Dtype* heatmaps, const int nw, const int nh) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  render_heatmaps_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(image, w, h, nw, nh, heatmaps);
}

// 绘制vecmaps
template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::render_vecmaps_gpu(Dtype* image, const int w, const int h,
                                          const Dtype* heatmaps, const int nw, const int nh) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  render_vecmaps_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(image, w, h, nw, nh, heatmaps);
}

template <typename Dtype>
void VisualizeBoxposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int w = bottom[0]->width();
  const int h = bottom[0]->height();
  const int nw = bottom[1]->width();
  const int nh = bottom[1]->height();
  int num_people = bottom[2]->height();
  Dtype* image = bottom[0]->mutable_gpu_data();
  const Dtype* heatmaps = bottom[1]->gpu_data();
  const Dtype* proposals = bottom[2]->gpu_data();
  const Dtype* proposals_cpu = bottom[2]->cpu_data();
  if (proposals_cpu[0] < 0) num_people = 0;
  // draw skeleton
  if (drawtype_ == VisualizeBoxposeParameter_BPDrawType_POSE ||
      drawtype_ == VisualizeBoxposeParameter_BPDrawType_POSE_BOX) {
    if (num_people > 0) {
      int pose_people = 0;
      for (int i = 0; i < num_people; ++i) {
        int n_points = proposals_cpu[i*61 + 58];
        if (n_points >= 3) {
          pose_people++;
        }
      }
      if (pose_people > 0) {
        const int limb[] = LIMB_COCO;
        const int nlimb = sizeof(limb)/(2*sizeof(int));
        Blob<Dtype> vec_blob(1,1,pose_people,nlimb*2);
        Dtype* vec_ptr = vec_blob.mutable_cpu_data();
        Blob<Dtype> proposals_blob(1,1,pose_people,54);
        Dtype* proposals_ptr = proposals_blob.mutable_cpu_data();
        int idx = 0;
        for (int p = 0; p < num_people; ++p) {
          if (int(proposals_cpu[p*61 + 58]) < 3) continue;
          // proposals_blob
          for (int k = 0; k < 18; ++k) {
            float xk = proposals_cpu[p*61 + 4 + 3*k];
            float yk = proposals_cpu[p*61 + 5 + 3*k];
            float vk = proposals_cpu[p*61 + 6 + 3*k];
            proposals_ptr[idx*54+3*k] = xk;
            proposals_ptr[idx*54+3*k+1] = yk;
            proposals_ptr[idx*54+3*k+2] = vk;
          }
          // vec
          for (int l = 0; l < nlimb; ++l) {
            float a_x = proposals_cpu[p*61 + 4 + 3*limb[2*l]] * w;
            float a_y = proposals_cpu[p*61 + 5 + 3*limb[2*l]] * h;
            float a_v = proposals_cpu[p*61 + 6 + 3*limb[2*l]];
            float b_x = proposals_cpu[p*61 + 4 + 3*limb[2*l+1]] * w;
            float b_y = proposals_cpu[p*61 + 5 + 3*limb[2*l+1]] * h;
            float b_v = proposals_cpu[p*61 + 6 + 3*limb[2*l+1]];
            if (a_v > pose_threshold_ && b_v > pose_threshold_) {
              float dx = b_x - a_x;
              float dy = b_y - a_y;
              float norm = sqrt(dx * dx + dy * dy);
              dx = dx /norm;
              dy = dy /norm;
              vec_ptr[idx*nlimb*2 + 2*l] = dx;
              vec_ptr[idx*nlimb*2 + 2*l + 1] = dy;
            } else {
              vec_ptr[idx*nlimb*2 + 2*l] = 0;
              vec_ptr[idx*nlimb*2 + 2*l + 1] = 0;
            }
          }
          idx++;
        }
        const Dtype* vec_cptr_gpu = vec_blob.gpu_data();
        const Dtype* proposals_cptr_gpu = proposals_blob.gpu_data();
        render_pose_gpu(image,w,h,proposals_cptr_gpu,vec_cptr_gpu,pose_people,pose_threshold_);
        render_points_gpu(image,w,h,proposals_cptr_gpu,pose_people,pose_threshold_);
      }
    }
  } else if (drawtype_ == VisualizeBoxposeParameter_BPDrawType_HEATMAP ||
             drawtype_ == VisualizeBoxposeParameter_BPDrawType_HEATMAP_BOX) {
    // draw heatmaps
    render_heatmaps_gpu(image, w, h, heatmaps, nw, nh);
  } else if (drawtype_ == VisualizeBoxposeParameter_BPDrawType_VECMAP ||
             drawtype_ == VisualizeBoxposeParameter_BPDrawType_VECMAP_BOX) {
    // draw vecmaps
    render_vecmaps_gpu(image, w, h, heatmaps, nw, nh);
  } else {
    // other draw type.
    // do nothing
  }
  // wrap by GPU
  unsigned char* wrap;
  unsigned char wrap_image[h*w*3];
  cudaMalloc(&wrap, w*h*3*sizeof(unsigned char));
  cv_inv(bottom[0]->gpu_data(), w, h, wrap);
  cudaMemcpy(wrap_image, wrap, w*h*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(wrap);
  // display
  cv::Mat display_image(h, w, CV_8UC3, wrap_image);
  // draw box
  if (drawtype_ == VisualizeBoxposeParameter_BPDrawType_BOX ||
      drawtype_ == VisualizeBoxposeParameter_BPDrawType_POSE_BOX ||
      drawtype_ == VisualizeBoxposeParameter_BPDrawType_HEATMAP_BOX ||
      drawtype_ == VisualizeBoxposeParameter_BPDrawType_VECMAP_BOX) {
    if (num_people > 0) {
      for (int i = 0; i < num_people; ++i) {
        int xmin = static_cast<int>(proposals_cpu[i*61] * w);
        int xmax = static_cast<int>(proposals_cpu[i*61+2] * w);
        int ymin = static_cast<int>(proposals_cpu[i*61+1] * h);
        int ymax = static_cast<int>(proposals_cpu[i*61+3] * h);
        xmin = std::min(std::max(xmin,0),w-1);
        ymin = std::min(std::max(ymin,0),h-1);
        xmax = std::min(std::max(xmax,0),w-1);
        ymax = std::min(std::max(ymax,0),h-1);
        cv::Point top_left_pt(xmin,ymin);
        cv::Point bottom_right_pt(xmax,ymax);
        cv::rectangle(display_image, top_left_pt, bottom_right_pt, cv::Scalar(0,255,0), 3);
      }
    }
  }
  static int counter = 1;
  static double last_time = get_wall_time();
  static double this_time = last_time;
  static float fps = 1.0;
  // save
  if (write_frames_) {
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(98);
    char fname[256];
    sprintf(fname, "%s/frame%06d.jpg", output_directory_.c_str(), counter);
    cv::imwrite(fname, display_image, compression_params);
  }
  // show & score
  if (visualize_) {
    if (print_score_) {
      if (num_people > 0) {
        for (int i = 0; i < num_people; ++i) {
          float conf = proposals_cpu[i*61 + 59];
          std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                    << "Score: " << conf << std::endl;
        }
      }
    }
    cv::imshow("remo", display_image);
  }
  // FPS
  counter++;
  if (counter % 30 == 0) {
    this_time = get_wall_time();
    fps = (float)30 / (float)(this_time - last_time);
    last_time = this_time;
    std::cout << "Frame ID: " << counter << std::endl;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
              << "FPS: " << fps << std::endl;
  }
  // wait for key-process
  if (cv::waitKey(1) == 27) {
    raise(SIGINT);
  }
  // output top
  top[0]->mutable_cpu_data()[0] = 0;
}

INSTANTIATE_LAYER_GPU_FUNCS(VisualizeBoxposeLayer);

}
