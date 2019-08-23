#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/pose_eval_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void PoseEvalLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 获取该层的参数
  const PoseEvalParameter& pose_eval_param = this->layer_param_.pose_eval_param();
  stride_ = pose_eval_param.stride();
  area_thre_ = pose_eval_param.area_thre();
  for (int i = 0; i < pose_eval_param.oks_thre_size(); ++i) {
    oks_thre_.push_back(pose_eval_param.oks_thre(i));
  }
}

template <typename Dtype>
void PoseEvalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0] -> [1,p,np,3]
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->width(), 3);
  // bottom[1] -> [1,4,h,w]
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 4);
  top[0]->Reshape(1,1,1,1+3*oks_thre_.size());
}

template <typename Dtype>
void PoseEvalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  // ===========================================================================
  // Vars for oks
  // GT
  vector<vector<Point2f> > gt_kps;
  vector<vector<int> > gt_vis;
  vector<float> gt_area;
  const int num_parts = bottom[0]->height()-1;
  const int num_person = int(gt_data[0]);
  const int channelOffset = bottom[1]->height() * bottom[1]->width();
  CHECK_GT(num_person, 0);
  for (int i = 0; i < num_person; ++i) {
    int c = i/10;
    int idx = c * channelOffset + 1 + (num_parts*3 + 1) * (i % 10);
    float area = gt_data[idx++];
    if (area < area_thre_) continue;
    gt_area.push_back(area);
    vector<Point2f> kps;
    vector<int> vis;
    // 获取每个person的关节点和可见信息
    for (int k = 0; k < num_parts; ++k) {
      Point2f point;
      point.x = gt_data[idx++];
      point.y = gt_data[idx++];
      int v = gt_data[idx++];
      kps.push_back(point);
      vis.push_back(v);
    }
    gt_kps.push_back(kps);
    gt_vis.push_back(vis);
  }
  // ===========================================================================
  // Pred
  vector<vector<Point2f> > pred_kps;
  vector<vector<float> > pred_conf;
  // bottom[0] -> [1,p,np,3] (x,y,s) while x/y is normalized value
  const int offs = num_parts * 3;
  const int img_w = bottom[1]->width() * stride_;
  const int img_h = bottom[1]->height() * stride_;
  int people_preds = bottom[0]->channels();
  if ((pred_data[0] < 0) && (people_preds == 1)) people_preds = 0;
  if (people_preds > 0) {
    for (int i = 0; i < people_preds; ++i) {
      int idx = i * offs;
      vector<Point2f> kps;
      vector<float> conf;
      for (int k = 0; k < num_parts; ++k) {
        Point2f point;
        point.x = pred_data[idx++] * img_w;
        point.y = pred_data[idx++] * img_h;
        float s = pred_data[idx++];
        kps.push_back(point);
        conf.push_back(s);
      }
      pred_kps.push_back(kps);
      pred_conf.push_back(conf);
    }
  }
  CHECK_EQ(pred_kps.size(), people_preds);
  CHECK_EQ(pred_conf.size(), people_preds);
  // ===========================================================================
  // 计算GT与PRED的OKS
  // 获取Vars
  vector<float> vars;
  if (num_parts == 18) {
    float sigma[18] = {0.26,0.79,0.79,0.72,0.62,0.79,0.72,0.62,1.07,0.87,0.89,1.07,0.87,0.89,0.25,0.25,0.35,0.35};
    for (int i = 0; i < num_parts; ++i) {
      float var = pow(2*sigma[i], 2);
      vars.push_back(var);
    }
  } else if (num_parts == 15) {
    float sigma[15] = {0.75,0.79,0.79,0.72,0.62,0.79,0.72,0.62,1.07,0.87,0.89,1.07,0.87,0.89,0.79};
    for (int i = 0; i < num_parts; ++i) {
      float var = pow(2*sigma[i], 2);
      vars.push_back(var);
    }
  } else {
    LOG(FATAL) << "Unsupport num_parts, only 18 for COCO and 15 for MPI.";
  }
  CHECK_EQ(vars.size(), num_parts);
  // 获取Tps/Fps
  vector<int> tps, fps;
  int gts = gt_kps.size();
  tps.resize(oks_thre_.size());
  fps.resize(oks_thre_.size());
  if (pred_kps.size() == 0) {
    for (int i = 0; i < tps.size(); ++i) {
      tps[i] = 0;
      fps[i] = 0;
    }
  } else {
    // normal operation
    vector<bool> has_found_pred(pred_kps.size(), false);
    vector<bool> has_found_gt(gt_kps.size(), false);
    vector<pair<float, pair<int, int> > > oks_pairs;
    // 遍历每一个GT
    for (int i = 0; i < gt_kps.size(); ++i) {
      // 搜索每一个Pred,检查其OKS是否合法
      for (int j = 0; j < pred_kps.size(); ++j) {
        int num_vis = 0;
        float sum = 0;
        // 检查所有parts
        for (int k = 0; k < num_parts; ++k) {
          if (gt_vis[i][k] <= 1) {
            num_vis++;
            float dx = gt_kps[i][k].x - pred_kps[j][k].x;
            float dy = gt_kps[i][k].y - pred_kps[j][k].y;
            float e = (dx*dx+dy*dy)/2.0/gt_area[i]/vars[k];
            sum += exp(-e);
          }
        }
        float oks = (num_vis > 0) ? sum / num_vis : 0;
        oks_pairs.push_back(std::make_pair(oks, std::make_pair(i,j)));
      }
    }
    // 使用oks排序
    std::stable_sort(oks_pairs.begin(), oks_pairs.end(), SortScorePairDescend<std::pair<int,int> >);
    // 根据oks排序结果获得匹配
    for (int i = 0; i < oks_pairs.size(); ++i) {
      float oks = oks_pairs[i].first;
      int gt_idx = oks_pairs[i].second.first;
      int pred_idx = oks_pairs[i].second.second;
      if (has_found_gt[gt_idx]) continue;
      if (has_found_pred[pred_idx]) continue;
      for (int l = 0; l < oks_thre_.size(); ++l) {
        if (oks >= oks_thre_[l]) tps[l]++;
      }
      has_found_gt[gt_idx] = true;
      has_found_pred[pred_idx] = true;
    }
    for (int i = 0; i < oks_thre_.size(); ++i) {
      fps[i] = pred_kps.size() - tps[i];
    }
  }

  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = gts;
  for(int i = 0; i < oks_thre_.size(); ++i) {
    top_data[1+3*i] = tps[i];
    top_data[2+3*i] = fps[i];
    top_data[3+3*i] = oks_thre_[i];
  }
}

INSTANTIATE_CLASS(PoseEvalLayer);
REGISTER_LAYER_CLASS(PoseEval);

}  // namespace caffe
