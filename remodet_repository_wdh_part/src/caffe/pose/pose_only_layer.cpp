#include <string>
#include <vector>
#include <map>
#include <utility>
#include <stdio.h>

#include "caffe/pose/pose_only_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void PoseOnlyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  num_parts_ = 18;
}

template <typename Dtype>
void PoseOnlyLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  // bottom[0]: pose_proposal
  // [1, num_people, num_parts+1, 3]
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->height(), num_parts_+1);
  CHECK_EQ(bottom[0]->width(), 3);
  top[0]->Reshape(1,1,1,7+num_parts_*3);
}

template <typename Dtype>
void PoseOnlyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype* pose_data = bottom[0]->cpu_data();
  //----------------------------------------------------------------------------
  vector<NormalizedBBox> pose_bboxes;
  vector<vector<Dtype> > pose_kps;
  int num_pose_person = bottom[0]->channels();
  if (pose_data[0] < 0 && num_pose_person == 1) {
    num_pose_person = 0;
  }
  for (int i = 0; i < num_pose_person; ++i) {
    NormalizedBBox box;
    vector<Dtype> kps(3*num_parts_+3,Dtype(0));
    Dtype xmin = FLT_MAX, ymin = FLT_MAX, xmax = -FLT_MAX, ymax = -FLT_MAX;
    int num_keypoints = 0;
    for (int k = 0; k < num_parts_; ++k) {
      Dtype x = pose_data[i*(num_parts_+1)*3 + k*3];
      Dtype y = pose_data[i*(num_parts_+1)*3 + k*3 + 1];
      Dtype v = pose_data[i*(num_parts_+1)*3 + k*3 + 2];
      kps[3*k] = x;
      kps[3*k+1] = y;
      kps[3*k+2] = v;
      if (v > 0.05) {
        num_keypoints++;
        xmin = (x < xmin) ? x : xmin;
        ymin = (y < ymin) ? y : ymin;
        xmax = (x > xmax) ? x : xmax;
        ymax = (y > ymax) ? y : ymax;
      }
    }
    // count of parts
    kps[3*num_parts_] = pose_data[i*(num_parts_+1)*3 + num_parts_*3];
    // sum of score
    kps[3*num_parts_+1] = pose_data[i*(num_parts_+1)*3 + num_parts_*3 + 1];
    // avg_score
    kps[3*num_parts_+2] = pose_data[i*(num_parts_+1)*3 + num_parts_*3 + 2];
    // push到pose_bboxes
    box.set_xmin(xmin);
    box.set_ymin(ymin);
    box.set_xmax(xmax);
    box.set_ymax(ymax);
    pose_bboxes.push_back(box);
    pose_kps.push_back(kps);
  }
  //----------------------------------------------------------------------------
  // 遍历pose-boxes
  vector<vector<Dtype> > matched_vecs;
  for (int g = 0; g < pose_bboxes.size(); ++g) {
    // 匹配失败, 使用关键点估计box
    vector<Dtype> out_vec(7+3*num_parts_,0);
    // output box
    float xmin_pose = pose_bboxes[g].xmin();
    float xmax_pose = pose_bboxes[g].xmax();
    float ymin_pose = pose_bboxes[g].ymin();
    float ymax_pose = pose_bboxes[g].ymax();
    xmin_pose -= (xmax_pose - xmin_pose) * 0.1;
    xmin_pose = std::min(std::max(xmin_pose, (float)0),(float)1);
    xmax_pose += (xmax_pose - xmin_pose) * 0.1;
    xmax_pose = std::min(std::max(xmax_pose, (float)0),(float)1);
    ymin_pose -= (ymax_pose - ymin_pose) * 0.1;
    ymin_pose = std::min(std::max(ymin_pose, (float)0),(float)1);
    ymax_pose += (ymax_pose - ymin_pose) * 0.1;
    ymax_pose = std::min(std::max(ymax_pose, (float)0),(float)1);
    out_vec[0] = xmin_pose;
    out_vec[1] = ymin_pose;
    out_vec[2] = xmax_pose;
    out_vec[3] = ymax_pose;
    int n_points = 0;
    for (int k = 0; k < num_parts_; ++k) {
      float xk = pose_kps[g][3*k];
      float yk = pose_kps[g][3*k+1];
      float vk = pose_kps[g][3*k+2];
      // detected
      if (vk > 0.05) {
        n_points++;
        out_vec[4+3*k] = xk;
        out_vec[4+3*k+1] = yk;
        out_vec[4+3*k+2] = vk;
      }
    }
    out_vec[4+3*num_parts_] = n_points;
    out_vec[5+3*num_parts_] = pose_kps[g][3*num_parts_+2];
    out_vec[6+3*num_parts_] = -1;
    matched_vecs.push_back(out_vec);
  }
  //----------------------------------------------------------------------------
  int num_p = matched_vecs.size();
  if (num_p == 0) {
    top[0]->Reshape(1,1,1,7+num_parts_*3);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(1,1,num_p,7+num_parts_*3);
    Dtype* top_data = top[0]->mutable_cpu_data();
    int idx = 0;
    for (int i = 0; i < num_p; ++i) {
      // 输出box信息
      top_data[idx++] = matched_vecs[i][0];
      top_data[idx++] = matched_vecs[i][1];
      top_data[idx++] = matched_vecs[i][2];
      top_data[idx++] = matched_vecs[i][3];
      // 输出kps信息
      for (int k = 0; k < num_parts_; ++k) {
        top_data[idx++] = matched_vecs[i][4+3*k];
        top_data[idx++] = matched_vecs[i][4+3*k+1];
        top_data[idx++] = matched_vecs[i][4+3*k+2];
      }
      // num_points
      top_data[idx++] = matched_vecs[i][4+3*num_parts_];
      // 输出评分信息
      top_data[idx++] = matched_vecs[i][5+3*num_parts_];
      // ID
      top_data[idx++] = matched_vecs[i][6+3*num_parts_];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(PoseOnlyLayer, Forward);
#endif

INSTANTIATE_CLASS(PoseOnlyLayer);
REGISTER_LAYER_CLASS(PoseOnly);

} // namespace caffe
