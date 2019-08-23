#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/reid/color_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define COLOR_LIMB {0,1, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 11,12, 12,13}
#define COLOR_TORSO {2,5,8,11}

// 统计Limbs中每个像素的类别: 0-511 (8,8,8)
// template <typename Dtype>
// __global__ void LimbsColorStatsKernel(const int nthreads,
//     const Dtype* color_data, const int nLimbs, const int nSplits,
//     const int xoffs, const int yoffs, const int image_w, const int image_h,
//     const Dtype* proposals, Dtype* top_data) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//     // 位置
//     const int dx = index % xoffs;
//     const int dy = (index / xoffs) % yoffs;
//     const int lm = (index / xoffs / yoffs) % nSplits;
//     const int l = index / xoffs / yoffs / nSplits;
//     // 表
//     const int color_limb[] = COLOR_LIMB;
//     // 节点位置
//     const Dtype x1_l = proposals[4 + color_limb[2*l]*3] * image_w;
//     const Dtype y1_l = proposals[4 + color_limb[2*l]*3+1] * image_h;
//     const Dtype v1_l = proposals[4 + color_limb[2*l]*3+2];
//     const Dtype x2_l = proposals[4 + color_limb[2*l+1]*3] * image_w;
//     const Dtype y2_l = proposals[4 + color_limb[2*l+1]*3+1] * image_h;
//     const Dtype v2_l = proposals[4 + color_limb[2*l+1]*3+2];
//     if ((v1_l > (Dtype)0.05) && (v2_l > (Dtype)0.05)) {
//       Dtype d_x = x2_l - x1_l;
//       Dtype d_y = y2_l - y1_l;
//       int my = round(y1_l + lm * d_y / nSplits);
//       int mx = round(x1_l + lm * d_x / nSplits);
//       int sy = dy - yoffs / 2;
//       int sx = dx - xoffs / 2;
//       int py = my + sy;
//       int px = mx + sx;
//       if (px >=0 && px < image_w && py >=0 && py < image_h) {
//         int r = color_data[py*image_w+px] + 104;
//         int g = color_data[image_h*image_w+py*image_w+px] + 117;
//         int b = color_data[2*image_h*image_w+py*image_w+px] + 123;
//         r = r < 0 ? 0 : (r > 255 ? 255 : r);
//         g = g < 0 ? 0 : (g > 255 ? 255 : g);
//         b = b < 0 ? 0 : (b > 255 ? 255 : b);
//         int r_idx = r / 32;
//         int g_idx = g / 32;
//         int b_idx = b / 32;
//         top_data[index] = r_idx*64+g_idx*8+b_idx;
//       } else {
//         // 不属于任何一类
//         top_data[index] = -1;
//       }
//     } else {
//       // 全部为-1，不属于任何一类
//       top_data[index] = -1;
//     }
//   }
// }
//
// template <typename Dtype>
// void ColorFeatureLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   const int N = bottom[1]->shape(2);
//   const int image_w = bottom[0]->width();
//   const int image_h = bottom[0]->height();
//
//   const Dtype* proposal = bottom[1]->gpu_data();
//   const Dtype* proposal_cpu = bottom[1]->cpu_data();
//   const Dtype* color_data = bottom[0]->gpu_data();
//   const Dtype* color_data_cpu = bottom[0]->cpu_data();
//
//   const int offs_color = image_h * image_w;
//
//   // if no proposal
//   if ((N == 1) && ((int)proposal_cpu[60] < 0)) {
//     top[0]->Reshape(1,1,11,512);
//     caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
//     return;
//   }
//
//   // normal operation
//   top[0]->Reshape(1,N,11,512);
//   const int offs_item = 11*512;
//   const int offs_map = 512;
//   Dtype* top_data = top[0]->mutable_cpu_data();
//   const int color_limb[] = COLOR_LIMB;
//   const int color_torso[] = COLOR_TORSO;
//   CHECK_EQ(sizeof(color_limb)/(2*sizeof(int)), 9);
//   CHECK_EQ(sizeof(color_torso)/sizeof(int), 4);
//
//   // ###########################################################################
//   // Limbs统计
//   // ###########################################################################
//   // limbs
//   // 参数
//   const int nSplits = 10;
//   const int xoffs = 3;
//   const int yoffs = 3;
//   const int count = 9 * nSplits * yoffs * xoffs;
//   const int limb_points = nSplits * yoffs * xoffs;
//   const int nLimbs = sizeof(color_limb)/(2*sizeof(int));
//   Blob<Dtype> limbs_bins(1,1,N,count);
//   // 统计每个像素点的类别
//   for (int i = 0; i < N; ++i) {
//     LimbsColorStatsKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//       count, color_data, nLimbs, nSplits, xoffs, yoffs, image_w, image_h,
//       proposal+i*61, limbs_bins.mutable_gpu_data()+i*count);
//   }
//   // 统计
//   const Dtype* cls_cptr = limbs_bins.cpu_data();
//   for (int i = 0; i < N; ++i) {
//     for (int l = 0; l < 9; ++l) {
//       const int data_idx = i * offs_item + l * offs_map;
//       // 检查
//       const Dtype v1_l = proposal_cpu[61*i + 4 + color_limb[2*l]*3+2];
//       const Dtype v2_l = proposal_cpu[61*i + 4 + color_limb[2*l+1]*3+2];
//       if ((v1_l < (Dtype)0.05) || (v2_l < (Dtype)0.05)) {
//         caffe_set(top[0]->width(), Dtype(-1), top[0]->mutable_cpu_data()+data_idx);
//         continue;
//       }
//       //initialization
//       caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+data_idx);
//       // 统计
//       for (int k = 0; k < limb_points; ++k) {
//         int cls = cls_cptr[i*count+l*limb_points+k];
//         if (cls >= 0 && cls < top[0]->width()) {
//           top_data[data_idx+cls]++;
//         }
//       }
//       // 标准化
//       Dtype sum = 0.001;
//       for (int p = 0; p < top[0]->width(); ++p) {
//         sum += top_data[data_idx+p] * top_data[data_idx+p];
//       }
//       sum = sqrt(sum);
//       for (int p = 0; p < top[0]->width(); ++p) {
//         top_data[data_idx+p] /= sum;
//       }
//     }
//   }
//   // ========================================
//   // Torso & BBox -> 使用CPU计算
//   // ========================================
//   for (int i = 0; i < N; ++i) {
//     // ###########################################################################
//     // Torso统计
//     // ###########################################################################
//     const int torso_data_idx = i*offs_item+9*offs_map;
//     Dtype torso_xmin = FLT_MAX, torso_ymin = FLT_MAX, torso_xmax = -1, torso_ymax = -1;
//     int vis_torso = 0;
//     for (int t = 0; t < 4; ++t) {
//       const Dtype xt = proposal_cpu[61*i + 4 + color_torso[t]*3] * image_w;
//       const Dtype yt = proposal_cpu[61*i + 4 + color_torso[t]*3 + 1] * image_h;
//       const Dtype vt = proposal_cpu[61*i + 4 + color_torso[t]*3 + 2];
//       if (vt > (Dtype)0.05) {
//         vis_torso++;
//         torso_xmin = xt < torso_xmin ? xt : torso_xmin;
//         torso_ymin = yt < torso_ymin ? yt : torso_ymin;
//         torso_xmax = xt > torso_xmax ? xt : torso_xmax;
//         torso_ymax = yt > torso_ymax ? yt : torso_ymax;
//       }
//     }
//     if (vis_torso >= 3) {
//       int xmin = torso_xmin < 0 ? 0 : (torso_xmin > (image_w - 1) ? (image_w - 1) : (int)torso_xmin);
//       int ymin = torso_ymin < 0 ? 0 : (torso_ymin > (image_h - 1) ? (image_h - 1) : (int)torso_ymin);
//       int xmax = torso_xmax < 0 ? 0 : (torso_xmax > (image_w - 1) ? (image_w - 1) : (int)torso_xmax);
//       int ymax = torso_ymax < 0 ? 0 : (torso_ymax > (image_h - 1) ? (image_h - 1) : (int)torso_ymax);
//       CHECK_GT(xmax, xmin);
//       CHECK_GT(ymax, ymin);
//       // 初始化
//       caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+torso_data_idx);
//       // 统计
//       for (int py = ymin; py < ymax; ++py) {
//         for (int px = xmin; px < xmax; ++px) {
//           int r = color_data_cpu[py*image_w+px] + 104;
//           int g = color_data_cpu[offs_color+py*image_w+px] + 117;
//           int b = color_data_cpu[2*offs_color+py*image_w+px] + 123;
//           r = r < 0 ? 0 : (r > 255 ? 255 : r);
//           g = g < 0 ? 0 : (g > 255 ? 255 : g);
//           b = b < 0 ? 0 : (b > 255 ? 255 : b);
//           int r_idx = r / 32;
//           int g_idx = g / 32;
//           int b_idx = b / 32;
//           top_data[torso_data_idx+r_idx*64+g_idx*8+b_idx]++;
//         }
//       }
//       // 标准化
//       Dtype sum = 0.001;
//       for (int p = 0; p < top[0]->width(); ++p) {
//         sum += top_data[torso_data_idx+p] * top_data[torso_data_idx+p];
//       }
//       sum = sqrt(sum);
//       for (int p = 0; p < top[0]->width(); ++p) {
//         top_data[torso_data_idx+p] /= sum;
//       }
//     } else {
//       caffe_set(top[0]->width(), Dtype(-1), top[0]->mutable_cpu_data()+torso_data_idx);
//     }
//
//     // ##########################################################################
//     // 计算BBOX的直方图
//     // ##########################################################################
//     const int bbox_data_idx = i*offs_item+10*offs_map;
//     // 初始化
//     caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+bbox_data_idx);
//     // 获取bbox
//     int xmin = proposal_cpu[61*i] * image_w;
//     int ymin = proposal_cpu[61*i+1] * image_h;
//     int xmax = proposal_cpu[61*i+2] * image_w;
//     int ymax = proposal_cpu[61*i+3] * image_h;
//     xmin = xmin < 0 ? 0 : (xmin > (image_w - 1) ? (image_w - 1) : (int)xmin);
//     ymin = ymin < 0 ? 0 : (ymin > (image_h - 1) ? (image_h - 1) : (int)ymin);
//     xmax = xmax < 0 ? 0 : (xmax > (image_w - 1) ? (image_w - 1) : (int)xmax);
//     ymax = ymax < 0 ? 0 : (ymax > (image_h - 1) ? (image_h - 1) : (int)ymax);
//     // 统计
//     CHECK_GT(xmax, xmin);
//     CHECK_GT(ymax, ymin);
//     for (int py = ymin; py < ymax; ++py) {
//       for (int px = xmin; px < xmax; ++px) {
//         int r = color_data_cpu[py*image_w+px] + 104;
//         int g = color_data_cpu[offs_color+py*image_w+px] + 117;
//         int b = color_data_cpu[2*offs_color+py*image_w+px] + 123;
//         r = r < 0 ? 0 : (r > 255 ? 255 : r);
//         g = g < 0 ? 0 : (g > 255 ? 255 : g);
//         b = b < 0 ? 0 : (b > 255 ? 255 : b);
//         int r_idx = r / 32;
//         int g_idx = g / 32;
//         int b_idx = b / 32;
//         top_data[bbox_data_idx+r_idx*64+g_idx*8+b_idx]++;
//       }
//     }
//     // 标准化
//     Dtype sum = 0.001;
//     for (int p = 0; p < top[0]->width(); ++p) {
//       sum += top_data[bbox_data_idx+p] * top_data[bbox_data_idx+p];
//     }
//     sum = sqrt(sum);
//     for (int p = 0; p < top[0]->width(); ++p) {
//       top_data[bbox_data_idx+p] /= sum;
//     }
//   }
// }
//
// INSTANTIATE_LAYER_GPU_FUNCS(ColorFeatureLayer);

}  // namespace caffe
