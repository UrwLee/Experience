#include <vector>
#include <float.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/reid/color_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// 9 Limbs
/**
*    <Neck-Throat>: 0-1
*    <RS-RE> <RE-RW>: 2-3 3-4
*    <LS-LE> <LE-LW>: 5-6 6-7
*    <RH-RK> <RK-RA>: 8-9 9-10
*    <LH-LK> <LK-LA>: 11-12 12-13
 */
#define COLOR_LIMB {0,1, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 11,12, 12,13}
// Torso
/**
 * RS-LS-RH-LH -> 2/5/8/11
 */
#define COLOR_TORSO {2,5,8,11}

template <typename Dtype>
void ColorFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // NOTE: params has not been defined.
}

template <typename Dtype>
void ColorFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // each proposal should have a width of 61.
  // [1,1,N,61]
  CHECK_EQ(bottom[1]->shape(3), 61);
  // input data layer should have a channel of 3.
  // [1,3,ih,iw]
  CHECK_EQ(bottom[0]->shape(1), 3);
  // output blob
  // [1,N,11,512]
  vector<int> shape(4,1);
  shape[1] = bottom[1]->shape(2);
  shape[2] = 11;
  shape[3] = 512;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ColorFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int N = bottom[1]->shape(2);
  const int image_w = bottom[0]->width();
  const int image_h = bottom[0]->height();

  // proposals
  const Dtype* proposal = bottom[1]->cpu_data();
  // color data
  const Dtype* color_data = bottom[0]->cpu_data();
  const int offs_color = image_h * image_w;
  // no proposals: all output is -1.
  if ((N == 1) && ((int)proposal[59] < 0)) {
    top[0]->Reshape(1,1,11,512);
    caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
    return;
  }

  // normal output.
  /**
   * 0-3: bbox
   * 4-57: kps <x,y,v>
   * 58: num_vis
   * 59: score
   * 60: id (-1)
   */
  top[0]->Reshape(1,N,11,512);
  const int offs_item = 11*512;
  const int offs_map = 512;
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int color_limb[] = COLOR_LIMB;
  const int color_torso[] = COLOR_TORSO;
  CHECK_EQ(sizeof(color_limb)/(2*sizeof(int)), 9);
  CHECK_EQ(sizeof(color_torso)/sizeof(int), 4);
  for (int i = 0; i < N; ++i) {
    // ##########################################################################
    // 计算Limb的直方图：９
    // ##########################################################################
    for (int l = 0; l < 9; ++l) {
      // 计算坐标
      const Dtype x1_l = proposal[61*i + 4 + color_limb[2*l]*3] * image_w;
      const Dtype y1_l = proposal[61*i + 4 + color_limb[2*l]*3+1] * image_h;
      const Dtype v1_l = proposal[61*i + 4 + color_limb[2*l]*3+2];
      const Dtype x2_l = proposal[61*i + 4 + color_limb[2*l+1]*3] * image_w;
      const Dtype y2_l = proposal[61*i + 4 + color_limb[2*l+1]*3+1] * image_h;
      const Dtype v2_l = proposal[61*i + 4 + color_limb[2*l+1]*3+2];
      // this limb is visible
      if ((v1_l > (Dtype)0.05) && (v2_l > (Dtype)0.05)) {
        // initial the output as 0.
        caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+i*offs_item+l*offs_map);
        // 计算参数
        const int data_idx = i*offs_item+l*offs_map;
        const int split_nums_limb = 10;
        const int area_pixels = 1;  // [-1, +1]
        Dtype dx = x2_l - x1_l;
        Dtype dy = y2_l - y1_l;
        // 统计线段上的N个点, 统计每个点的周围若干个点
        for (int lm = 0; lm < split_nums_limb; ++lm) {
          int mx = round(x1_l + lm * dx / split_nums_limb);
          int my = round(y1_l + lm * dy / split_nums_limb);
          for (int sy = -area_pixels; sy < area_pixels; ++sy) {
            for (int sx = -area_pixels; sx < area_pixels; ++sx) {
              int py = my + sy;
              int px = mx + sx;
              if (px >=0 && px < image_w && py >=0 && py < image_h) {
                int r = color_data[py*image_w+px] + 104;
                int g = color_data[offs_color+py*image_w+px] + 117;
                int b = color_data[2*offs_color+py*image_w+px] + 123;
                r = r < 0 ? 0 : (r > 255 ? 255 : r);
                g = g < 0 ? 0 : (g > 255 ? 255 : g);
                b = b < 0 ? 0 : (b > 255 ? 255 : b);
                int r_idx = r / 32;
                int g_idx = g / 32;
                int b_idx = b / 32;
                top_data[data_idx+r_idx*64+g_idx*8+b_idx]++;
              }
            }
          }
        }
        // 标准化
        Dtype sum = 0.001;
        for (int p = 0; p < top[0]->width(); ++p) {
          sum += top_data[data_idx+p] * top_data[data_idx+p];
        }
        sum = sqrt(sum);
        for (int p = 0; p < top[0]->width(); ++p) {
          top_data[data_idx+p] /= sum;
        }
      } else {
        // this limb is invisible
        caffe_set(top[0]->width(), Dtype(-1), top[0]->mutable_cpu_data()+i*offs_item+l*offs_map);
      }
    }
    // ##########################################################################
    // 计算Torso的直方图
    // ##########################################################################
    const int torso_data_idx = i*offs_item+9*offs_map;
    // 计算torso的bbox
    Dtype torso_xmin = FLT_MAX, torso_ymin = FLT_MAX, torso_xmax = -1, torso_ymax = -1;
    int vis_torso = 0;
    for (int t = 0; t < 4; ++t) {
      const Dtype xt = proposal[61*i + 4 + color_torso[t]*3] * image_w;
      const Dtype yt = proposal[61*i + 4 + color_torso[t]*3 + 1] * image_h;
      const Dtype vt = proposal[61*i + 4 + color_torso[t]*3 + 2];
      if (vt > (Dtype)0.05) {
        vis_torso++;
        torso_xmin = xt < torso_xmin ? xt : torso_xmin;
        torso_ymin = yt < torso_ymin ? yt : torso_ymin;
        torso_xmax = xt > torso_xmax ? xt : torso_xmax;
        torso_ymax = yt > torso_ymax ? yt : torso_ymax;
      }
    }
    // normal
    if (vis_torso >= 3) {
      int xmin = torso_xmin < 0 ? 0 : (torso_xmin > (image_w - 1) ? (image_w - 1) : (int)torso_xmin);
      int ymin = torso_ymin < 0 ? 0 : (torso_ymin > (image_h - 1) ? (image_h - 1) : (int)torso_ymin);
      int xmax = torso_xmax < 0 ? 0 : (torso_xmax > (image_w - 1) ? (image_w - 1) : (int)torso_xmax);
      int ymax = torso_ymax < 0 ? 0 : (torso_ymax > (image_h - 1) ? (image_h - 1) : (int)torso_ymax);
      CHECK_GT(xmax, xmin);
      CHECK_GT(ymax, ymin);
      // 初始化
      caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+torso_data_idx);
      // 统计该区域内的所有像素
      for (int py = ymin; py < ymax; ++py) {
        for (int px = xmin; px < xmax; ++px) {
          int r = color_data[py*image_w+px] + 104;
          int g = color_data[offs_color+py*image_w+px] + 117;
          int b = color_data[2*offs_color+py*image_w+px] + 123;
          r = r < 0 ? 0 : (r > 255 ? 255 : r);
          g = g < 0 ? 0 : (g > 255 ? 255 : g);
          b = b < 0 ? 0 : (b > 255 ? 255 : b);
          int r_idx = r / 32;
          int g_idx = g / 32;
          int b_idx = b / 32;
          top_data[torso_data_idx+r_idx*64+g_idx*8+b_idx]++;
        }
      }
      // 标准化
      Dtype sum = 0.001;
      for (int p = 0; p < top[0]->width(); ++p) {
        sum += top_data[torso_data_idx+p] * top_data[torso_data_idx+p];
      }
      sum = sqrt(sum);
      for (int p = 0; p < top[0]->width(); ++p) {
        top_data[torso_data_idx+p] /= sum;
      }
    } else {
    // abnormal -> invisible
      caffe_set(top[0]->width(), Dtype(-1), top[0]->mutable_cpu_data()+torso_data_idx);
    }
    // ##########################################################################
    // 计算BBOX的直方图
    // ##########################################################################
    const int bbox_data_idx = i*offs_item+10*offs_map;
    // 初始化
    caffe_set(top[0]->width(), (Dtype)0, top[0]->mutable_cpu_data()+bbox_data_idx);
    // 获取bbox
    int xmin = proposal[61*i] * image_w;
    int ymin = proposal[61*i+1] * image_h;
    int xmax = proposal[61*i+2] * image_w;
    int ymax = proposal[61*i+3] * image_h;
    xmin = xmin < 0 ? 0 : (xmin > (image_w - 1) ? (image_w - 1) : (int)xmin);
    ymin = ymin < 0 ? 0 : (ymin > (image_h - 1) ? (image_h - 1) : (int)ymin);
    xmax = xmax < 0 ? 0 : (xmax > (image_w - 1) ? (image_w - 1) : (int)xmax);
    ymax = ymax < 0 ? 0 : (ymax > (image_h - 1) ? (image_h - 1) : (int)ymax);
    // 统计
    CHECK_GT(xmax, xmin);
    CHECK_GT(ymax, ymin);
    for (int py = ymin; py < ymax; ++py) {
      for (int px = xmin; px < xmax; ++px) {
        int r = color_data[py*image_w+px] + 104;
        int g = color_data[offs_color+py*image_w+px] + 117;
        int b = color_data[2*offs_color+py*image_w+px] + 123;
        r = r < 0 ? 0 : (r > 255 ? 255 : r);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);
        int r_idx = r / 32;
        int g_idx = g / 32;
        int b_idx = b / 32;
        top_data[bbox_data_idx+r_idx*64+g_idx*8+b_idx]++;
      }
    }
    // 标准化
    Dtype sum = 0.001;
    for (int p = 0; p < top[0]->width(); ++p) {
      sum += top_data[bbox_data_idx+p] * top_data[bbox_data_idx+p];
    }
    sum = sqrt(sum);
    for (int p = 0; p < top[0]->width(); ++p) {
      top_data[bbox_data_idx+p] /= sum;
    }
    // ##########################################################################
    // END of Proposal Color-Features
  }
}

#ifdef CPU_ONLY
STUB_GPU(ColorFeatureLayer);
#endif

INSTANTIATE_CLASS(ColorFeatureLayer);
REGISTER_LAYER_CLASS(ColorFeature);

}  // namespace caffe
