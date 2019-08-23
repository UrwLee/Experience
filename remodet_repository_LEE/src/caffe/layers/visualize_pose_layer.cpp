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

namespace caffe {

template <typename Dtype>
double VisualizeposeLayer<Dtype>::get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time,NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

template <typename Dtype>
inline void getColors(Dtype* c, Dtype v, Dtype vmin, Dtype vmax) {
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

// 绘制某个part的heatmap
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_heatmap_cpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                                            const int nw, const int nh, const int num_parts, const int part) {
  // image -> 1 x 3 x h x w
  // heatmaps -> 1 x num_parts x nh x nw
  int real_part = part;
  if (part < 0) real_part = 0;
  if (part >= num_parts) real_part = num_parts - 1;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
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
        int x_nei = int(x_on_box + 1e-5);
        x_nei = (x_nei < 0) ? 0 : x_nei;
        int y_nei = int(y_on_box + 1e-5);
        y_nei = (y_nei < 0) ? 0 : y_nei;
        int offset_src = real_part * nw * nh;
        value = heatmaps[offset_src + y_nei * nw + x_nei];
      }

      Dtype c[3];
      Dtype alpha = 0.7;
      getColors(c, value, (Dtype)0, (Dtype)1);

      b = (1-alpha) * b + alpha * c[2];
      g = (1-alpha) * g + alpha * c[1];
      r = (1-alpha) * r + alpha * c[0];

      image[y * w + x] = b;
      image[w * h + y * w + x] = g;
      image[2 * w * h + y * w + x] = r;
    }
  }
}

// 绘制从from_part -> num_parts - 1 个heatmap
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_heatmaps_from_id_cpu(Dtype* image, const int w, const int h, const Dtype* heatmaps,
                                            const int nw, const int nh, const int num_parts, const int from_part) {
  // image -> 1 x 3 x h x w
  // heatmaps -> 1 x num_parts x nh x nw
  int part_id = from_part;
  if (from_part < 0) part_id = 0;
  if (from_part >= num_parts) part_id = num_parts - 1;
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
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
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
      for (int part = part_id; part < num_parts; part++) {
        Dtype x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
        Dtype y_on_box = h_inv * y + (0.5 * h_inv - 0.5);
        if(x_on_box >= 0 && x_on_box < nw && y_on_box >= 0 && y_on_box < nh){
          int x_nei = int(x_on_box + 1e-5);
          x_nei = (x_nei < 0) ? 0 : x_nei;
          int y_nei = int(y_on_box + 1e-5);
          y_nei = (y_nei < 0) ? 0 : y_nei;
          int offset_src = part * nw * nh;
          value = heatmaps[offset_src + y_nei * nw + x_nei];
          value = std::max(std::min(value, Dtype(1)), Dtype(0));
          c[0] += value * color[(part%nColor)*3+0];
          c[1] += value * color[(part%nColor)*3+1];
          c[2] += value * color[(part%nColor)*3+2];
        }
      }
      c[0] = std::max(std::min(c[0], Dtype(255)), Dtype(0));
      c[1] = std::max(std::min(c[1], Dtype(255)), Dtype(0));
      c[2] = std::max(std::min(c[2], Dtype(255)), Dtype(0));
      Dtype alpha = 0.7;
      b = (1-alpha) * b + alpha * c[2];
      g = (1-alpha) * g + alpha * c[1];
      r = (1-alpha) * r + alpha * c[0];
      image[y * w + x] = b;
      image[w * h + y * w + x] = g;
      image[2 * w * h + y * w + x] = r;
    }
  }
}

// 绘制pose
template <typename Dtype>
void VisualizeposeLayer<Dtype>::render_pose_cpu(Dtype* image, const int w, const int h, const Dtype* poses,
                                            const int num_people, const Dtype threshold, const int num_parts) {
  // image -> 1 x 3 x h x w
  // poses -> 1 x NP x num_parts x 3
  // num_people -> NP
  Dtype shared_poses[num_people*num_parts*3];
  Dtype shared_mins_x[num_people];
  Dtype shared_maxs_x[num_people];
  Dtype shared_mins_y[num_people];
  Dtype shared_maxs_y[num_people];
  Dtype shared_scalef_x[num_people];
  Dtype shared_scalef_y[num_people];
  // copy
  for (int p = 0; p < num_people; ++p) {
    shared_mins_x[p] = w;
    shared_mins_y[p] = h;
    shared_maxs_x[p] = 0;
    shared_maxs_y[p] = 0;
    for (int part = 0; part < num_parts; ++part) {
      Dtype x = poses[p*(num_parts+1)*3 + part*3];
      Dtype y = poses[p*(num_parts+1)*3 + part*3+1];
      Dtype z = poses[p*(num_parts+1)*3 + part*3+2];
      x *= w;
      y *= h;
      shared_poses[p*num_parts*3 + part*3] = x;
      shared_poses[p*num_parts*3 + part*3+1] = y;
      shared_poses[p*num_parts*3 + part*3+2] = z;
      if (z > threshold) {
        if (x<shared_mins_x[p]) shared_mins_x[p] = x;
        if (x>shared_maxs_x[p]) shared_maxs_x[p] = x;
        if (y<shared_mins_y[p]) shared_mins_y[p] = y;
        if (y>shared_maxs_y[p]) shared_maxs_y[p] = y;
      }
    }
    shared_scalef_x[p] = shared_maxs_x[p] - shared_mins_x[p];
    shared_scalef_y[p] = shared_maxs_y[p] - shared_mins_y[p];
    shared_scalef_x[p] = (shared_scalef_x[p] + shared_scalef_y[p]) / 2.0;
    // 按照200像素等效
    if (shared_scalef_x[p] < 200) {
      shared_scalef_x[p] = shared_scalef_x[p] / 200.;
      if (shared_scalef_x[p] < 0.33) shared_scalef_x[p] = 0.33;
    } else {
      shared_scalef_x[p] = 1.0;
    }
    // 将最大最小分别移除50像素
    shared_maxs_x[p] += 50;
    shared_maxs_y[p] += 50;
    shared_mins_x[p] -= 50;
    shared_mins_y[p] -= 50;
  }
  // limb绘制列表
  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
  // 绘制颜色
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
  Dtype radius = (Dtype)(2*h) / 200.;
  Dtype stickwidth = (Dtype)h / 120.;
  // 逐点绘制
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      Dtype b, g, r;
      b = image[y * w + x];
      g = image[w * h + y * w + x];
      r = image[2 * w * h + y * w + x];
      // 绘制所有的person
      for(int p = 0; p < num_people; p++){
        // 该像素是否在该person范围内,不在直接返回
        if (x > shared_maxs_x[p] || x < shared_mins_x[p]
            || y > shared_maxs_y[p] || y < shared_mins_y[p]) {
          continue;
        }
        // 绘制线段
        for(int l = 0; l < nlimb; l++){
          Dtype b_sqrt = shared_scalef_x[p] * shared_scalef_x[p] * stickwidth * stickwidth;
          Dtype alpha = 0.5;
          int part_a = limb[2*l];
          int part_b = limb[2*l+1];
          float x_a = (shared_poses[p*num_parts*3 + part_a*3]);
          float x_b = (shared_poses[p*num_parts*3 + part_b*3]);
          float y_a = (shared_poses[p*num_parts*3 + part_a*3 + 1]);
          float y_b = (shared_poses[p*num_parts*3 + part_b*3 + 1]);
          float value_a = shared_poses[p*num_parts*3 + part_a*3 + 2];
          float value_b = shared_poses[p*num_parts*3 + part_b*3 + 2];
          if(value_a > threshold && value_b > threshold){
            float x_p = (x_a + x_b) / 2;
            float y_p = (y_a + y_b) / 2;
            float angle = atan2f(y_b - y_a, x_b - x_a);
            float sine = sinf(angle);
            float cosine = cosf(angle);
            float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);
            float A = cosine * (x - x_p) + sine * (y - y_p);
            float B = sine * (x - x_p) - cosine * (y - y_p);
            float judge = A * A / a_sqrt + B * B / b_sqrt;
            float maxV = 1;

            float cx, cy, cz;
            cx = color[(l%nColor)*3+0];
            cy = color[(l%nColor)*3+1];
            cz = color[(l%nColor)*3+2];
            if(judge < maxV) {
              b = (1-alpha) * b + alpha * cz;
              g = (1-alpha) * g + alpha * cy;
              r = (1-alpha) * r + alpha * cx;
            }
          }
        }// limb绘制完毕
        // 绘制关节点
        for(int i = 0; i < num_parts; i++) {
          float local_x = shared_poses[p*num_parts*3 + i*3];
          float local_y = shared_poses[p*num_parts*3 + i*3 + 1];
          float value = shared_poses[p*num_parts*3 + i*3 + 2];
          if(value > threshold) {
            float dist2 = (x - local_x) * (x - local_x) + (y - local_y) * (y - local_y);
            float maxr2 = shared_scalef_x[p] * shared_scalef_x[p] * radius * radius;
            float alpha = 0.6;

            float cx,cy,cz;
            cx = color[(i%nColor)*3+0];
            cy = color[(i%nColor)*3+1];
            cz = color[(i%nColor)*3+2];
            if(dist2 < maxr2){
              b = (1-alpha) * b + alpha * cz;
              g = (1-alpha) * g + alpha * cy;
              r = (1-alpha) * r + alpha * cx;
            }
          }
        }
      }
      image[y * w + x] = b;
      image[w * h + y * w + x] = g;
      image[2 * w * h + y * w + x] = r;
    }
  }
}

// template <typename Dtype>
// void VisualizeposeLayer<Dtype>::render_pose_vers(const Dtype* joints, const int w, const int h, const int num_people,
//                                                  const Dtype pose_threshold, Dtype* vers) {
//   const int limb[] = LIMB_COCO;
//   for (int i = 0; i < num_people; ++i) {
//     for (int j = 0; j < num_limbs_; ++j) {
//       const int nA = limb[2*j];
//       const int nB = limb[2*j+1];
//       Dtype s_x = joints[i*num_parts_*3 + nA*3] * w;
//       Dtype s_y = joints[i*num_parts_*3 + nA*3 + 1] * h;
//       Dtype d_x = joints[i*num_parts_*3 + nB*3] * w - s_x;
//       Dtype d_y = joints[i*num_parts_*3 + nB*3 + 1] * h - s_y;
//       Dtype norm_vec = sqrt(d_x*d_x + d_y*d_y);
//       vers[i*num_limbs_+j*4] = d_x / norm_vec;
//       vers[i*num_limbs_+j*4+1] = d_y / norm_vec;
//       vers[i*num_limbs_+j*4+2] = -d_y / norm_vec;
//       vers[i*num_limbs_+j*4+3] = d_x / norm_vec;
//       vers[i*num_limbs_+j*4+4] = norm_vec;
//     }
//   }
// }

template <typename Dtype>
void VisualizeposeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const VisualizeposeParameter &visualize_pose_param =
      this->layer_param_.visualize_pose_param();
  is_type_coco_ = visualize_pose_param.is_type_coco();
  num_parts_ = 18;
  num_limbs_ = 17;
  // draw type
  drawtype_ = visualize_pose_param.type();
  // if drawtype_ == HEATMAP_ID -> which part to draw?
  part_id_ = visualize_pose_param.part_id();
  // if drawtype_ == HEATMAP_FROM -> which part to draw from?
  from_part_ = visualize_pose_param.from_part();

  vec_id_ = visualize_pose_param.vec_id();
  from_vec_ = visualize_pose_param.from_vec();

  // pose_threshold_ for pose-display
  pose_threshold_ = visualize_pose_param.pose_threshold();
  // write_frames_
  write_frames_ = visualize_pose_param.write_frames();
  output_directory_ = visualize_pose_param.output_directory();

  // visualize
  visualize_ = visualize_pose_param.visualize();
  draw_skeleton_ = visualize_pose_param.draw_skeleton();
  print_score_ = visualize_pose_param.print_score();
}

template <typename Dtype>
void VisualizeposeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // bottom[0]: image
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 3);
  // bottom[1]: heatmaps + PAs
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), num_parts_ + 2 * num_limbs_);
  // bottom[2]: joints
  CHECK_EQ(bottom[2]->num(), 1);
  CHECK_EQ(bottom[2]->height(), num_parts_+1);
  CHECK_EQ(bottom[2]->width(), 3);

  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void VisualizeposeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    const int w = bottom[0]->width();
    const int h = bottom[0]->height();
    const int nw = bottom[1]->width();
    const int nh = bottom[1]->height();
    int num_people = bottom[2]->channels();
    Dtype* image = bottom[0]->mutable_cpu_data();
    const Dtype* heatmaps = bottom[1]->cpu_data();
    const Dtype* joints = bottom[2]->cpu_data();
    // person
    if (joints[0] < 0) num_people = 0;
    // draw poses / heatmaps
    if (drawtype_ == VisualizeposeParameter_DrawType_POSE) {
      if (draw_skeleton_) {
        if (num_people > 0) {
          render_pose_cpu(image, w, h, joints, num_people, pose_threshold_, num_parts_);
        }
      }
    } else if (drawtype_ == VisualizeposeParameter_DrawType_HEATMAP_ID) {
      render_heatmap_cpu(image, w, h, heatmaps, nw, nh, num_parts_, part_id_);
    } else if (drawtype_ == VisualizeposeParameter_DrawType_HEATMAP_FROM) {
      render_heatmaps_from_id_cpu(image, w, h, heatmaps, nw, nh, num_parts_, from_part_);
    } else {
      LOG(FATAL) << "Unknown drawn type.";
    }
    // wrap
    const Dtype* image_cptr = bottom[0]->cpu_data();
    unsigned char wrap_image[h*w*3];
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          int value = int(image_cptr[c*h*w + i*w + j] + 0.5);
          value = value < 0 ? 0 : (value > 255 ? 255 : value);
          wrap_image[3*(i*w+j)+c] = (unsigned char)value;
        }
      }
    }
    // display
    cv::Mat display_image(h, w, CV_8UC3, wrap_image);
    static int counter = 0;
    static double last_time = get_wall_time();
    static double this_time = last_time;
    static float fps = 1.0;
    char tmp_str[256];
    // write FPS
    snprintf(tmp_str, 256, "%4.1f fps", fps);
    cv::putText(display_image, tmp_str, cv::Point(25,35),
        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);
    // write num_persons
    snprintf(tmp_str, 256, "%4d", num_people);
    cv::putText(display_image, tmp_str, cv::Point(w-100+2, 35+2),
        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(display_image, tmp_str, cv::Point(w-100, 35),
        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
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
      cv::imshow("remo", display_image);
    }
    // incremental and compute FPS
    counter++;
    if (counter % 30 == 0) {
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

#ifdef CPU_ONLY
STUB_GPU_FORWARD(VisualizeposeLayer, Forward);
#endif

INSTANTIATE_CLASS(VisualizeposeLayer);
REGISTER_LAYER_CLASS(Visualizepose);

} // namespace caffe
