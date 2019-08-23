#include "caffe/remo/visualizer.hpp"
#include <string>
#include <vector>
#include <utility>
#include <stdio.h>

namespace caffe {

#define LIMB_COCO {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17}
#define LIMB_COCO_SEQ {30,31, 36,37, 32,33, 34,35, 38,39, 40,41, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 42,43, 44,45, 48,49, 46,47, 50,51};
#define COLOR_MAPS {255,0,0,255,85,0,255,170,0,255,255,0,170,255,0,85,255,0, \
                   0,255,0,0,255,85,0,255,170,0,255,255,0,170,255,0,85,255, \
                   0,0,255,85,0,255,170,0,255,255,0,255,255,0,170,255,0,85}
// Methods
template <typename Dtype>
inline __device__ Dtype min(Dtype a, Dtype b) {
  return (a < b) ? a : b;
}

template <typename Dtype>
inline __device__ Dtype max(Dtype a, Dtype b) {
  return (a > b) ? a : b;
}

template <typename Dtype>
inline __device__ void getColor(Dtype* c, Dtype v, Dtype vmin, Dtype vmax) {
   c[0] = c[1] = c[2] = (Dtype)255;
   Dtype dv;
   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;
   if (v < (vmin + 0.125 * dv)) {
      c[0] = (Dtype)256 * (0.5 + (v * 4));
      c[1] = c[2] = 0;
   } else if (v < (vmin + 0.375 * dv)) {
      c[0] = 255;
      c[1] = (Dtype)256 * (v - 0.125) * 4;
      c[2] = 0;
   } else if (v < (vmin + 0.625 * dv)) {

      c[0] = (Dtype)256 * (-4 * v + 2.5);
      c[1] = 255;
      c[2] = (Dtype)256 * (4 * (v - 0.375));
   } else if (v < (vmin + 0.875 * dv)) {
      c[0] = 0;
      c[1] = (Dtype)256 * (-4 * v + 3.5);
      c[2] = 255;
   } else {
      c[0] = 0;
      c[1] = 0;
      c[2] = (Dtype)256 * (-4 * v + 4.5);
   }
}

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype* out, Dtype v0, Dtype v1, Dtype v2, Dtype v3, Dtype dx) {
    *out = (-0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3) * dx * dx * dx
         + (v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5 * v0 + 0.5 * v2) * dx
         + v1;
}

// 绘制pose
/**
 * img_ptr -> Blob 图像数据指针
 * proposals -> n*(18*3) -> person的关键点数据，normalized.
 * vec -> 连接法线向量
 * num_people -> num of persons
 * threshold -> 关键点置信度阈值
 */
template <typename Dtype>
__global__ void render_pose_kernel(Dtype* img_ptr, const int w, const int h, const Dtype* proposals,
                                  const Dtype* vec, const int num_people, const Dtype threshold) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ float shared_poses[48*18*3];
  __shared__ float2 shared_mins[48];
  __shared__ float2 shared_maxs[48];
  __shared__ float2 shared_scalef[48];
  __shared__ float max_scalef[48];
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
    max_scalef[p] = max(shared_scalef[p].x,shared_scalef[p].y);
    if (max_scalef[p] < 200) {
      max_scalef[p] = max_scalef[p] / 200.;
      if (max_scalef[p] < 0.3) max_scalef[p] = 0.3;
    } else {
      max_scalef[p] = 1.0;
    }
    shared_maxs[p].x += 50;
    shared_maxs[p].y += 50;
    shared_mins[p].x -= 50;
    shared_mins[p].y -= 50;
  }
  __syncthreads();
  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
  const int color[] = COLOR_MAPS;
  const int nColor = sizeof(color)/(3*sizeof(int));
  Dtype stickwidth = (Dtype)h / 500.;
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
        float b_sqrt = max_scalef[p] * max_scalef[p] * stickwidth * stickwidth;
        // float b_sqrt = 0.01;
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

/**
 * 绘制points
 */
template <typename Dtype>
__global__ void render_points_kernel(Dtype* img_ptr, const int w, const int h, const Dtype* proposals,
                                     const int num_people, const Dtype threshold) {
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
    float radius = (float)(2 * h) / 500.;
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
      }
    }
    img_ptr[y * w + x] = b;
    img_ptr[w * h + y * w + x] = g;
    img_ptr[2 * w * h + y * w + x] = r;
  }
}

// 绘制heatmap
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

// 绘制vecmaps
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

// 转换为cv::Mat
template <typename Dtype>
__global__ void blob_cv_kernel(const int n, const Dtype* image, const int w, const int h, unsigned char* cv_img) {
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index % w;
    const int y = (index / w) % h;
    const int c = (index / w / h) % 3;
    int value = (int)(image[index] + 0.5);
    value = value < 0 ? 0 : (value > 255 ? 255 : value);
    cv_img[3*(y*w+x)+c] = (unsigned char)value;
  }
}

// 将GPU中的数据复制到CPU中的Mat格式
template <typename Dtype>
void Visualizer<Dtype>::blob_to_cv(const Dtype* image, const int w, const int h, cv::Mat* cv_img) {
  const int count = 3 * w * h;
  unsigned char wrap_image[count];
  unsigned char* wrap;
  cudaMalloc(&wrap, count*sizeof(unsigned char));
  blob_cv_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, image, w, h, wrap);
  cudaMemcpy(wrap_image, wrap, count*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(wrap);
  cv::Mat image_pro(h, w, CV_8UC3, wrap_image);
  *cv_img = image_pro;
}
template void Visualizer<float>::blob_to_cv(const float* image, const int w, const int h, cv::Mat* cv_img);
template void Visualizer<double>::blob_to_cv(const double* image, const int w, const int h, cv::Mat* cv_img);

// 将Mat格式转换为Blob格式，准备进行GPU处理
template <typename Dtype>
void Visualizer<Dtype>::cv_to_blob(const cv::Mat& image, Dtype* data) {
  for(int h = 0; h < image.rows; ++h) {
    const uchar* ptr = image.ptr<uchar>(h);
    int img_index = 0;
    for(int w = 0; w < image.cols; ++w){
      for(int c = 0; c < image.channels(); ++c){
        int data_index = (c * image.rows + h) * image.cols + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        data[data_index] = pixel;
      }
    }
  }
}
template void Visualizer<float>::cv_to_blob(const cv::Mat& image, float* data);
template void Visualizer<double>::cv_to_blob(const cv::Mat& image, double* data);

// 绘制pose
template <typename Dtype>
void Visualizer<Dtype>::render_pose_gpu(Dtype* image, const int w, const int h, const Dtype* proposals,
                                        const Dtype* vec, const int num_people, const Dtype threshold) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  if (num_people > 0) {
    render_pose_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      image, w, h, proposals, vec, num_people, threshold);
  }
}
template void Visualizer<float>::render_pose_gpu(float* image, const int w, const int h, const float* proposals,
                                        const float* vec, const int num_people, const float threshold);
template void Visualizer<double>::render_pose_gpu(double* image, const int w, const int h, const double* proposals,
                                        const double* vec, const int num_people, const double threshold);

// 绘制points
template <typename Dtype>
void Visualizer<Dtype>::render_points_gpu(Dtype* image, const int w, const int h, const Dtype* proposals,
                                                     const int num_people, const Dtype threshold) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  if (num_people > 0) {
    render_points_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(
      image, w, h, proposals, num_people, threshold);
  }
}
template void Visualizer<float>::render_points_gpu(float* image, const int w, const int h, const float* proposals,
                                                    const int num_people, const float threshold);
template void Visualizer<double>::render_points_gpu(double* image, const int w, const int h, const double* proposals,
                                                    const int num_people, const double threshold);

// 绘制heatmaps
template <typename Dtype>
void Visualizer<Dtype>::render_heatmaps_gpu(Dtype* image, const int w, const int h,
                                          const Dtype* heatmaps, const int nw, const int nh) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  render_heatmaps_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(image, w, h, nw, nh, heatmaps);
}
template void Visualizer<float>::render_heatmaps_gpu(float* image, const int w, const int h, const float* heatmaps, const int nw, const int nh);
template void Visualizer<double>::render_heatmaps_gpu(double* image, const int w, const int h, const double* heatmaps, const int nw, const int nh);

// 绘制vecmaps
template <typename Dtype>
void Visualizer<Dtype>::render_vecmaps_gpu(Dtype* image, const int w, const int h,
                                          const Dtype* heatmaps, const int nw, const int nh) {
  const dim3 ThreadsPerBlock(CAFFE_CUDA_NUM_THREADS_1D, CAFFE_CUDA_NUM_THREADS_1D);
  const dim3 BlocksSize(CAFFE_UPDIV(w, CAFFE_CUDA_NUM_THREADS_1D), CAFFE_UPDIV(h, CAFFE_CUDA_NUM_THREADS_1D));
  render_vecmaps_kernel<Dtype><<<BlocksSize, ThreadsPerBlock>>>(image, w, h, nw, nh, heatmaps);
}
template void Visualizer<float>::render_vecmaps_gpu(float* image, const int w, const int h, const float* heatmaps, const int nw, const int nh);
template void Visualizer<double>::render_vecmaps_gpu(double* image, const int w, const int h, const double* heatmaps, const int nw, const int nh);

// draw skeleton
template <typename Dtype>
void Visualizer<Dtype>::draw_skeleton(const vector<PMeta<Dtype> >& meta) {
  draw_skeleton(meta, &image_);
}
template void Visualizer<float>::draw_skeleton(const vector<PMeta<float> >& meta);
template void Visualizer<double>::draw_skeleton(const vector<PMeta<double> >& meta);

// draw skeleton
template <typename Dtype>
void Visualizer<Dtype>::draw_skeleton(const vector<PMeta<Dtype> >& meta, cv::Mat* out_image) {
  if (meta.size() == 0) {
    *out_image = image_;
    return;
  }
  int pose_people = 0;
  for (int i = 0; i < meta.size(); ++i) {
    if (meta[i].num_points >= 3) {
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
    for (int p = 0; p < meta.size(); ++p) {
      if (meta[p].num_points < 3) continue;
      for (int k = 0; k < 18; ++k) {
        float xk = meta[p].kps[k].x;
        float yk = meta[p].kps[k].y;
        float vk = meta[p].kps[k].v;
        proposals_ptr[idx*54+3*k] = xk;
        proposals_ptr[idx*54+3*k+1] = yk;
        proposals_ptr[idx*54+3*k+2] = vk;
      }
      // vec
      for (int l = 0; l < nlimb; ++l) {
        meta[p].kps[limb[2*l]].x * image_.cols;
        float a_x = meta[p].kps[limb[2*l]].x * image_.cols;
        float a_y = meta[p].kps[limb[2*l]].y * image_.rows;
        float a_v = meta[p].kps[limb[2*l]].v;
        float b_x = meta[p].kps[limb[2*l+1]].x * image_.cols;
        float b_y = meta[p].kps[limb[2*l+1]].y * image_.rows;
        float b_v = meta[p].kps[limb[2*l+1]].v;
        if (a_v > 0.05 && b_v > 0.05) {
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
    // image -> gpu
    Blob<Dtype> image_blob(1,3,image_.rows,image_.cols);
    cv_to_blob(image_,image_blob.mutable_cpu_data());
    Dtype* image_ptr = image_blob.mutable_gpu_data();
    render_pose_gpu(image_ptr,image_.cols,image_.rows,proposals_cptr_gpu,vec_cptr_gpu,pose_people,0.05);
    render_points_gpu(image_ptr,image_.cols,image_.rows,proposals_cptr_gpu,pose_people,0.05);
    blob_to_cv(image_blob.gpu_data(), image_.cols, image_.rows, out_image);
  } else {
    *out_image = image_;
    return;
  }
}
template void Visualizer<float>::draw_skeleton(const vector<PMeta<float> >& meta, cv::Mat* out_image);
template void Visualizer<double>::draw_skeleton(const vector<PMeta<double> >& meta, cv::Mat* out_image);

template <typename Dtype>
void Visualizer<Dtype>::draw_vecmap(const Dtype* heatmaps, const int map_width, const int map_height) {
  draw_vecmap(heatmaps,map_width,map_height,&image_);
}
template void Visualizer<float>::draw_vecmap(const float* heatmaps, const int map_width, const int map_height);
template void Visualizer<double>::draw_vecmap(const double* heatmaps, const int map_width, const int map_height);

template <typename Dtype>
void Visualizer<Dtype>::draw_vecmap(const Dtype* heatmaps, const int map_width,
                                    const int map_height, cv::Mat* out_image) {
  Blob<Dtype> image_blob(1,3,image_.rows,image_.cols);
  cv_to_blob(image_,image_blob.mutable_cpu_data());
  render_vecmaps_gpu(image_blob.mutable_gpu_data(), image_.cols, image_.rows,
                      heatmaps, map_width, map_height);
  blob_to_cv(image_blob.gpu_data(), image_.cols, image_.rows, out_image);
}
template void Visualizer<float>::draw_vecmap(const float* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);
template void Visualizer<double>::draw_vecmap(const double* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);

template <typename Dtype>
void Visualizer<Dtype>::draw_heatmap(const Dtype* heatmaps, const int map_width,
                                    const int map_height) {
  draw_heatmap(heatmaps,map_width,map_height,&image_);
}
template void Visualizer<float>::draw_heatmap(const float* heatmaps, const int map_width, const int map_height);
template void Visualizer<double>::draw_heatmap(const double* heatmaps, const int map_width, const int map_height);

template <typename Dtype>
void Visualizer<Dtype>::draw_heatmap(const Dtype* heatmaps, const int map_width,
                                    const int map_height, cv::Mat* out_image) {
  Blob<Dtype> image_blob(1,3,image_.rows,image_.cols);
  cv_to_blob(image_,image_blob.mutable_cpu_data());
  render_heatmaps_gpu(image_blob.mutable_gpu_data(), image_.cols, image_.rows,
                      heatmaps, map_width, map_height);
  blob_to_cv(image_blob.gpu_data(), image_.cols, image_.rows, out_image);
}
template void Visualizer<float>::draw_heatmap(const float* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);
template void Visualizer<double>::draw_heatmap(const double* heatmaps, const int map_width, const int map_height, cv::Mat* out_image);

}
