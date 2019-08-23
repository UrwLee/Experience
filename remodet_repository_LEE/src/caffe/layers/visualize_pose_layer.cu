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

#include "caffe/layers/visualize_pose_layer.hpp"

#define LIMB_COCO {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17}
#define LIMB_COCO_SEQ {30,31, 36,37, 32,33, 34,35, 38,39, 40,41, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 42,43, 44,45, 48,49, 46,47, 50,51};
namespace caffe {

template <typename Dtype>
inline __device__ Dtype min(Dtype a, Dtype b) {
  return (a < b) ? a : b;
}

template <typename Dtype>
inline __device__ Dtype max(Dtype a, Dtype b) {
  return (a > b) ? a : b;
}

// 第一种取色方法: 五段式取色,等间隔
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

// 第二种取色方法: 七段式取色,不等间隔
template <typename Dtype>
inline __device__ void getColor2(Dtype* c, Dtype v, Dtype vmin, Dtype vmax)
{
   c[0] = c[1] = c[2] = (Dtype)255; // b, g, r, white

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;

   v = (Dtype)55 * v;
   const int RY = 15;
   const int YG = 6;
   const int GC = 4;
   const int CB = 11;
   const int BM = 13;
   const int MR = 6;

   if (v < RY) {
    //  第一段: 蓝色 -> 青色
      c[0] = 255;
      c[1] = (Dtype)255 * (v / RY);  // G: 0-1
      c[2] = 0;
   } else if (v < RY+YG) {
    //  第二段: 青色 -> 绿色
      c[0] = 255 - (Dtype)255 * ((v-RY) / YG);
      c[1] = 255;
      c[2] = 0;
   } else if (v < RY+YG+GC) {
    // 第三段: 绿色 -> 黄色
      c[0] = 0;
      c[1] = 255;
      c[2] = (Dtype)255 * ((v-RY-YG) / GC);
   } else if (v < RY+YG+GC+CB) {
    // 第四段: 黄色 -> 红色
      c[0] = 0;
      c[1] = 255 - (Dtype)255 * ((v-RY-YG-GC) / CB);
      c[2] = 255;
   } else if (v < RY+YG+GC+CB+BM) {
    //  第五段: 红色 -> 粉色
      c[0] = (Dtype)255 * ((v-RY-YG-GC-CB) / BM);
      c[1] = 0;
      c[2] = 255;
   } else if (v < RY+YG+GC+CB+BM+MR) {
    //  第六段: 粉色 -> 蓝色
      c[0] = 255;
      c[1] = 0;
      c[2] = 255 - (Dtype)255 * ((v-RY-YG-GC-CB-BM) / MR);
   } else {
    //  第七段: 蓝色
     c[0] = 255;
     c[1] = 0;
     c[2] = 0;
   }
}

// 根据方向取色
template <typename Dtype>
inline __device__ void getColorXY(Dtype* c, Dtype x, Dtype y) {
  float rad = sqrt( x*x + y*y );
  float a = atan2(-y,-x)/M_PI;
  float fk = (a+1)/2.0; // 0 to 1
  if (::isnan(fk)) fk = 0;
  if (rad>1) rad = 1;
  getColor2(c, fk, 0, 1);
  c[0] = 255*(rad*(c[0]/255));
  c[1] = 255*(rad*(c[1]/255));
  c[2] = 255*(rad*(c[2]/255));
}

// cubic 插值法
template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype* out, Dtype v0, Dtype v1, Dtype v2, Dtype v3, Dtype dx) {
    *out = (-0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3) * dx * dx * dx
         + (v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5 * v0 + 0.5 * v2) * dx
         + v1;
}

// 绘制Vec-Map
template <typename Dtype>
__global__ void render_vecmap_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* vecmaps, const int vec_channel) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  __syncthreads();
  // const int limb[] = LIMB_COCO;
  // const int nlimb = sizeof(limb)/(2*sizeof(int));
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

    Dtype value = 0;
    Dtype value2 = 0;
    Dtype val = 0;
    if(x_on_box >= 0 && x_on_box < nw && y_on_box >=0 && y_on_box < nh){
      Dtype value_this;
      int x_nei[4];
      x_nei[1] = int(x_on_box + 1e-5);
      x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
      x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
      x_nei[2] = (x_nei[1] + 1 >= nw) ? (nw - 1) : (x_nei[1] + 1);
      x_nei[3] = (x_nei[2] + 1 >= nw) ? (nw - 1) : (x_nei[2] + 1);
      Dtype dx = x_on_box - x_nei[1];

      int y_nei[4];
      y_nei[1] = int(y_on_box + 1e-5);
      y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
      y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
      y_nei[2] = (y_nei[1] + 1 >= nh) ? (nh - 1) : (y_nei[1] + 1);
      y_nei[3] = (y_nei[2] + 1 >= nh) ? (nh - 1) : (y_nei[2] + 1);
      Dtype dy = y_on_box - y_nei[1];

      Dtype temp[4];

      int offset_src = limb_seq[2*vec_channel] * nw * nh;
      int offset_src2 = limb_seq[2*vec_channel+1] * nw * nh;
      for(int i = 0; i < 4; i++){
        cubic_interpolation<Dtype>(&temp[i], vecmaps[offset_src + y_nei[i]*nw + x_nei[0]],
                                             vecmaps[offset_src + y_nei[i]*nw + x_nei[1]],
                                             vecmaps[offset_src + y_nei[i]*nw + x_nei[2]],
                                             vecmaps[offset_src + y_nei[i]*nw + x_nei[3]], dx);
      }
      cubic_interpolation<Dtype>(&value_this, temp[0], temp[1], temp[2], temp[3], dy);
      value = value_this;
      for(int i = 0; i < 4; i++){
        cubic_interpolation<Dtype>(&temp[i], vecmaps[offset_src2 + y_nei[i]*nw + x_nei[0]],
                                             vecmaps[offset_src2 + y_nei[i]*nw + x_nei[1]],
                                             vecmaps[offset_src2 + y_nei[i]*nw + x_nei[2]],
                                             vecmaps[offset_src2 + y_nei[i]*nw + x_nei[3]], dx);
      }
      cubic_interpolation<Dtype>(&value_this, temp[0], temp[1], temp[2], temp[3], dy);
      value2 = value_this;
      val = value * value + value2 * value2;
    }

    Dtype c[3];
    Dtype alpha = 0.7;
    getColor(c, val, (Dtype)0, (Dtype)1);

    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];

    image[y * w + x] = b;
    image[w * h + y * w + x] = g;
    image[2 * w * h + y * w + x] = r;
  }
}

// 绘制Vec-Maps
template <typename Dtype>
__global__ void render_vecmap_from_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* vecmaps, const int from_channel) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  __syncthreads();
  const int color[] = {
    255,     0,     0,
    255,    85,     0,
    255,   170,     0,
    255,   255,     0,
    170,   255,     0,
     85,   255,     0,
      0,   255,    0,
      0,   255,    85,
      0,   255,   170,
      0,   255,   255,
      0,   170,   255,
      0,    85,   255,
      0,     0,   255,
     85,     0,   255,
    170,     0,   255,
    255,     0,   255,
    255,     0,   170,
    255,     0,    85,
    85,    85,    170};
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
      for (int l = from_channel; l < nlimb; ++l) {
        int x_nei = int(x_on_box + 1e-5);
        x_nei = (x_nei < 0) ? 0 : x_nei;
        int y_nei = int(y_on_box + 1e-5);
        y_nei = (y_nei < 0) ? 0 : y_nei;
        int offset_src = limb_seq[2*l] * nw * nh;
        int offset_src2 = limb_seq[2*l+1] * nw * nh;
        Dtype value = vecmaps[offset_src + y_nei * nw + x_nei];
        Dtype value2 = vecmaps[offset_src2 + y_nei * nw + x_nei];
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

// 绘制Heatmap
template <typename Dtype>
__global__ void render_heatmap_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* heatmaps, const int num_parts, const int part) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  __syncthreads();
  if(x < w && y < h) {
    Dtype b, g, r;
    Dtype h_inv = (Dtype)nh / (Dtype)h;
    Dtype w_inv = (Dtype)nw / (Dtype)w;

    b = image[y * w + x];
    g = image[w * h + y * w + x];
    r = image[2 * w * h + y * w + x];

    Dtype x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
    Dtype y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

    Dtype value = 0;

    if(x_on_box >= 0 && x_on_box < nw && y_on_box >=0 && y_on_box < nh){
      Dtype value_this;
      int x_nei[4];
      x_nei[1] = int(x_on_box + 1e-5);
      x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
      x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
      x_nei[2] = (x_nei[1] + 1 >= nw) ? (nw - 1) : (x_nei[1] + 1);
      x_nei[3] = (x_nei[2] + 1 >= nw) ? (nw - 1) : (x_nei[2] + 1);
      Dtype dx = x_on_box - x_nei[1];

      int y_nei[4];
      y_nei[1] = int(y_on_box + 1e-5);
      y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
      y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
      y_nei[2] = (y_nei[1] + 1 >= nh) ? (nh - 1) : (y_nei[1] + 1);
      y_nei[3] = (y_nei[2] + 1 >= nh) ? (nh - 1) : (y_nei[2] + 1);
      Dtype dy = y_on_box - y_nei[1];

      Dtype temp[4];
      int offset_src = part * nw * nh;
      for(int i = 0; i < 4; i++){
        cubic_interpolation<Dtype>(&temp[i], heatmaps[offset_src + y_nei[i]*nw + x_nei[0]],
                                             heatmaps[offset_src + y_nei[i]*nw + x_nei[1]],
                                             heatmaps[offset_src + y_nei[i]*nw + x_nei[2]],
                                             heatmaps[offset_src + y_nei[i]*nw + x_nei[3]], dx);
      }
      cubic_interpolation<Dtype>(&value_this, temp[0], temp[1], temp[2], temp[3], dy);
      value = value_this;
    }

    Dtype c[3];
    Dtype alpha = 0.7;
    getColor(c, value, (Dtype)0, (Dtype)1);

    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];

    image[y * w + x] = b;
    image[w * h + y * w + x] = g;
    image[2 * w * h + y * w + x] = r;
  }
}

// 绘制from_part -> num_parts个map
template <typename Dtype>
__global__ void render_heatmap_from_kernel(Dtype* image, const int w, const int h, const int nw,
                                    const int nh, const Dtype* heatmaps, const int num_parts, const int from_part) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int color[] = {
    255,     0,     0,
    255,    85,     0,
    255,   170,     0,
    255,   255,     0,
    170,   255,     0,
     85,   255,     0,
      0,   255,    0,
      0,   255,    85,
      0,   255,   170,
      0,   255,   255,
      0,   170,   255,
      0,    85,   255,
      0,     0,   255,
     85,     0,   255,
    170,     0,   255,
    255,     0,   255,
    255,     0,   170,
    255,     0,    85,
    85,    85,    170};
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
    for (int part = from_part; part < num_parts; part++) {
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
__global__ void render_pose_kernel(Dtype* img_ptr, const int w, const int h, const Dtype* poses,
                                  const Dtype* vec, const int num_people, const Dtype threshold, const int num_parts) {
  // 坐标
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  // 图形网格内的id, 由block内部不同的线程对共享内存进行操作
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  // 每个block内部保留poses坐标的副本
  __shared__ float shared_poses[96*18*3];
  __shared__ float2 shared_mins[96];
  __shared__ float2 shared_maxs[96];
  __shared__ float2 shared_scalef[96];
  // 由这些线程对共享内存进行读写
  // 获取副本
  if (global_idx < num_people) {
    int p = global_idx;
    shared_mins[p].x = w;
    shared_mins[p].y = h;
    shared_maxs[p].x = 0;
    shared_maxs[p].y = 0;
    for (int part = 0; part < num_parts; part++) {
      float x = poses[p*(num_parts+1)*3 + part*3];
      float y = poses[p*(num_parts+1)*3 + part*3+1];
      float z = poses[p*(num_parts+1)*3 + part*3+2];
      x *= w;
      y *= h;
      shared_poses[p*num_parts*3 + part*3] = x;
      shared_poses[p*num_parts*3 + part*3+1] = y;
      shared_poses[p*num_parts*3 + part*3+2] = z;
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

  // 等待所有线程在此同步
  __syncthreads();

  // limb绘制列表
  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
  // 19 colors
  const int color[] = {
     255,     0,     0,
     255,    85,     0,
     255,   170,     0,
     255,   255,     0,
     170,   255,     0,
      85,   255,     0,
       0,   255,    0,
       0,   255,    85,
       0,   255,   170,
       0,   255,   255,
       0,   170,   255,
       0,    85,   255,
       0,     0,   255,
      85,     0,   255,
     170,     0,   255,
     255,     0,   255,
     255,     0,   170,
     255,     0,    85,
      85,    85,    170};
  const int nColor = sizeof(color)/(3*sizeof(int));

  // 关节点的绘制半径
  Dtype radius = (Dtype)(2*h) / 200.;
  // limb的椭圆宽度定义
  Dtype stickwidth = (Dtype)h / 120.;

  // 绘制每个像素点
  if(x < w && y < h){
    Dtype b, g, r;
    b = img_ptr[y * w + x];
    g = img_ptr[w * h + y * w + x];
    r = img_ptr[2 * w * h + y * w + x];
    // 绘制所有的person
    for(int p = 0; p < num_people; p++){
      // 该像素是否在该person范围内,不在直接返回
      if (x > shared_maxs[p].x || x < shared_mins[p].x
          || y > shared_maxs[p].y || y < shared_mins[p].y) {
        continue;
      }
      // 检查每个线段
      for(int l = 0; l < nlimb; l++){
        // 椭圆宽度范围定义
        Dtype b_sqrt = shared_scalef[p].x * shared_scalef[p].x * stickwidth * stickwidth;
        // 颜色占比
        Dtype alpha = 0.5;
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        // 获取两个点的x和y坐标
        float x_a = (shared_poses[p*num_parts*3 + part_a*3]);
        float x_b = (shared_poses[p*num_parts*3 + part_b*3]);
        float y_a = (shared_poses[p*num_parts*3 + part_a*3 + 1]); // * ratio_to_origin + offset;
        float y_b = (shared_poses[p*num_parts*3 + part_b*3 + 1]); // * ratio_to_origin + offset;
        float value_a = shared_poses[p*num_parts*3 + part_a*3 + 2];
        float value_b = shared_poses[p*num_parts*3 + part_b*3 + 2];
        // 如果该线段有效, 则进一步判断该像素是否在该limb的作用范围内
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          // float angle = atan2f(y_b - y_a, x_b - x_a);
          // float sine = sinf(angle);
          // float cosine = cosf(angle);
          // // 中点到起点的平方距离
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);
          //
          // // A: 连接方向上的分量
          // // B: 连接法线方向上的投影分量
          // float A = cosine * (x - x_p) + sine * (y - y_p);
          // float B = sine * (x - x_p) - cosine * (y - y_p);
          float dx = x - x_p;
          float dy = y - y_p;
          float dist_1 = dx * vec[p*nlimb*2 + 2*l] + dy * vec[p*nlimb*2 + 2*l+1];
          float dist_2 = dx * vec[p*nlimb*2 + 2*l+1] - dy * vec[p*nlimb*2 + 2*l];
          float judge = dist_1 * dist_1 / a_sqrt +  dist_2 * dist_2 / b_sqrt;
          // 椭圆作用范围判断
          // a_sqrt -> 连接线段的长度之半 (平方)
          // b_sqrt -> 连接法线方向上的半径 (平方)
          // float judge = A * A / a_sqrt + B * B / b_sqrt;
          float maxV = 1;
          float3 co;
          co.x = color[(l%nColor)*3+0];
          co.y = color[(l%nColor)*3+1];
          co.z = color[(l%nColor)*3+2];
          // 在范围内,则修改该像素
          if(judge < maxV) {
            b = (1-alpha) * b + alpha * co.z;
            g = (1-alpha) * g + alpha * co.y;
            r = (1-alpha) * r + alpha * co.x;
          }
        }
      }// limb绘制完毕
      // 绘制关节点
      for(int i = 0; i < num_parts; i++) {
        float local_x = shared_poses[p*num_parts*3 + i*3];
        float local_y = shared_poses[p*num_parts*3 + i*3 + 1];
        float value = shared_poses[p*num_parts*3 + i*3 + 2];
        // 判断该关节点是否有效
        if(value > threshold) {
          float dist2 = (x - local_x) * (x - local_x) + (y - local_y) * (y - local_y);
          float maxr2 = shared_scalef[p].x * shared_scalef[p].x * radius * radius;
          float alpha = 0.6;
          float3 co;
          co.x = color[(i%nColor)*3+0];
          co.y = color[(i%nColor)*3+1];
          co.z = color[(i%nColor)*3+2];
          // 该像素在该关节点作用范围. 修改像素
          if(dist2 < maxr2){
            b = (1-alpha) * b + alpha * co.z;
            g = (1-alpha) * g + alpha * co.y;
            r = (1-alpha) * r + alpha * co.x;
          }
        }
      }
      // 关节点绘制完毕
    } // 所有person绘制完毕
    // 最后修改图像像素
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
void VisualizeposeLayer<Dtype>::cv_inv(const Dtype* image, const int w, const int h, unsigned char* out) {
  const int count = 3 * w * h;
  cv_inv_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, image, w, h, out);
}

// 绘制pose
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* poses,
                                            const Dtype* vec, const int num_people, const Dtype threshold,
                                            const int num_parts) {
  // image -> 1 x 3 x h x w
  // poses -> 1 x NP x num_parts x 3
  // num_people -> NP
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  if (num_people > 0) {
    render_pose_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      image, w, h, poses, vec, num_people, threshold, num_parts);
  }
}

// 绘制某个part的heatmap
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_heatmap_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                                            const int nw, const int nh, const int num_parts, const int part) {
  // image -> 1 x 3 x h x w
  // heatmaps -> 1 x num_parts x nh x nw
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  int real_part = part;
  if (part < 0) real_part = 0;
  if (part >= num_parts) real_part = num_parts - 1;
  render_heatmap_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
    image, w, h, nw, nh, heatmaps, num_parts, real_part);
}

// 绘制从from_part -> num_parts - 1 个heatmap
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_heatmaps_from_id_gpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                                            const int nw, const int nh, const int num_parts, const int from_part) {
  // image -> 1 x 3 x h x w
  // heatmaps -> 1 x num_parts x nh x nw
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  int part_id = from_part;
  if (from_part < 0) part_id = 0;
  if (from_part >= num_parts) part_id = num_parts - 1;
  render_heatmap_from_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
    image, w, h, nw, nh, heatmaps, num_parts, part_id);
}

// 绘制某个part的vecmap
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_vecmap_gpu(Dtype* image, const int w, const int h, const Dtype* vecmap,
                                            const int nw, const int nh, const int num_limbs, const int channel) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  int real_channel = channel;
  if (channel < 0) real_channel = 0;
  if (channel >= num_limbs) real_channel = num_limbs - 1;
  render_vecmap_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
    image, w, h, nw, nh, vecmap, real_channel);
}

// 绘制vecmaps
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_vecmaps_from_id_gpu(Dtype* image, const int w, const int h, const Dtype* vecmaps,
                                            const int nw, const int nh, const int channel_from) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  int real_channel = channel_from;
  if (channel_from < 0) real_channel = 0;
  render_vecmap_from_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
    image, w, h, nw, nh, vecmaps, real_channel);
}

// 显示
template <typename Dtype>
void VisualizeposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const int w = bottom[0]->width();
  const int h = bottom[0]->height();
  const int nw = bottom[1]->width();
  const int nh = bottom[1]->height();
  int num_people = bottom[2]->channels();
  Dtype* image = bottom[0]->mutable_gpu_data();
  const Dtype* heatmaps = bottom[1]->gpu_data();
  const Dtype* joints = bottom[2]->gpu_data();
  const Dtype* joints_cpu = bottom[2]->cpu_data();
  // if x < 0 -> none person is detected.
  if (joints_cpu[0] < 0) num_people = 0;
  // draw poses / heatmaps
  if (drawtype_ == VisualizeposeParameter_DrawType_POSE) {
    // print pose INFO
    if (num_people > 0 && print_score_) {
      for (int i = 0; i < num_people; ++i) {
        int idx = i*(num_parts_+1)*3 + num_parts_*3;
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                  << "Person " << i+1 << ": keypoints -> " << (int)joints_cpu[idx]
                  << ", score -> " << joints_cpu[idx+1] << ", avg_score -> "
                  << joints_cpu[idx+2] << std::endl;
      }
    }
    if (draw_skeleton_) {
      if (num_people > 0) {
        // cal vec of joints
        const int limb[] = LIMB_COCO;
        const int nlimb = sizeof(limb)/(2*sizeof(int));
        Blob<Dtype> vec_blob(1,1,num_people,nlimb*2);
        Dtype* vec_ptr = vec_blob.mutable_cpu_data();
        for (int p = 0; p < num_people; ++p){
          for (int l = 0; l < nlimb; ++l) {
            // we cal the vec of limbs
            float a_x = joints_cpu[p*(num_parts_+1)*3 + 3*limb[2*l]] * w;
            float a_y = joints_cpu[p*(num_parts_+1)*3 + 3*limb[2*l] + 1] * h;
            float b_x = joints_cpu[p*(num_parts_+1)*3 + 3*limb[2*l+1]] * w;
            float b_y = joints_cpu[p*(num_parts_+1)*3 + 3*limb[2*l+1] + 1] * h;
            float dx = b_x - a_x;
            float dy = b_y - a_y;
            float norm = sqrt(dx * dx + dy * dy);
            dx = dx /norm;
            dy = dy /norm;
            vec_ptr[p*nlimb*2 + 2*l] = dx;
            vec_ptr[p*nlimb*2 + 2*l + 1] = dy;
          }
        }
        const Dtype* vec_cptr_gpu = vec_blob.gpu_data();
        render_pose_gpu(image, w, h, joints, vec_cptr_gpu, num_people, pose_threshold_, num_parts_);
      }
    }
  } else if (drawtype_ == VisualizeposeParameter_DrawType_HEATMAP_ID) {
    render_heatmap_gpu(image, w, h, heatmaps, nw, nh, num_parts_, part_id_);
  } else if (drawtype_ == VisualizeposeParameter_DrawType_HEATMAP_FROM) {
    render_heatmaps_from_id_gpu(image, w, h, heatmaps, nw, nh, num_parts_, from_part_);
  } else if (drawtype_ == VisualizeposeParameter_DrawType_VECMAP_ID) {
    render_vecmap_gpu(image, w, h, heatmaps, nw, nh, num_limbs_, vec_id_);
  } else if (drawtype_ == VisualizeposeParameter_DrawType_VECMAP_FROM) {
    render_vecmaps_from_id_gpu(image, w, h, heatmaps, nw, nh, from_vec_);
  } else {
    LOG(FATAL) << "Unknown drawn type.";
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
  static int counter = 1;
  static double last_time = get_wall_time();
  static double this_time = last_time;
  static float fps = 1.0;
  // char tmp_str[256];
  // write FPS
  // snprintf(tmp_str, 256, "%4.1f fps", fps);
  // cv::putText(display_image, tmp_str, cv::Point(25,35),
  //     cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);
  // write num_persons
  // snprintf(tmp_str, 256, "%4d", num_people);
  // cv::putText(display_image, tmp_str, cv::Point(w-100+2, 35+2),
  //     cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
  // cv::putText(display_image, tmp_str, cv::Point(w-100, 35),
  //     cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
  // save images
  if (write_frames_) {
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(98);
    char fname[256];
    sprintf(fname, "%s/frame%06d.jpg", output_directory_.c_str(), counter);
    cv::imwrite(fname, display_image, compression_params);
  }
  // show the image
  if (visualize_) {
    cv:: Mat img_show;
    cv::resize(display_image,img_show,cv::Size(768,432));
    cv::imshow("remo", img_show);
    //cv::imshow("remo", display_image);
  }
  // incremental and compute FPS
  counter++;
  if (counter % 30 == 0) {
    std::cout << "Frame ID: " << counter << std::endl;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
              << "FPS: " << fps << std::endl;
    this_time = get_wall_time();
    fps = (float)30 / (float)(this_time - last_time);
    last_time = this_time;
  }
  // wait for key-process
  if (cv::waitKey(1) == 27) {
    raise(SIGINT);
  }
  // output top
  top[0]->mutable_cpu_data()[0] = 0;
}

INSTANTIATE_LAYER_GPU_FUNCS(VisualizeposeLayer);

} // namespace caffe
