#include "caffe/remo/res_frame.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
ResFrame<Dtype>::ResFrame(const cv::Mat& image, const int id, const int max_dis_size,
                          const vector<PMeta<Dtype> >& meta) {
  image_ = image;
  id_ = id;
  max_dis_size_ = max_dis_size;
  meta_ = meta;
}

template <typename Dtype>
void ResFrame<Dtype>::get_meta(std::vector<std::vector<Dtype> >* meta) {
  meta->clear();
  if (meta_.size() == 0) return;
  for (int i = 0; i < meta_.size(); ++i) {
    std::vector<Dtype> pmeta;
    // ID
    pmeta.push_back(meta_[i].id);
    // similarity
    // pmeta.push_back(meta_[i].similarity);
    // score
    pmeta.push_back(meta_[i].score);
    // bbox
    pmeta.push_back(meta_[i].bbox.x1_);
    pmeta.push_back(meta_[i].bbox.y1_);
    pmeta.push_back(meta_[i].bbox.x2_);
    pmeta.push_back(meta_[i].bbox.y2_);
    // num of points
    pmeta.push_back(meta_[i].num_points);
    // kps
    for (int k = 0; k < 18; ++k) {
      pmeta.push_back(meta_[i].kps[k].x);
      pmeta.push_back(meta_[i].kps[k].y);
      pmeta.push_back(meta_[i].kps[k].v);
    }
    // push
    meta->push_back(pmeta);
  }
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_vecmap(const Dtype* heatmaps, const int width, const int height,
                                           const bool show_bbox, const bool show_id) {
  Visualizer<Dtype> visual(image_, max_dis_size_);
  cv::Mat drawn;
  visual.draw_vecmap(heatmaps,width,height,&drawn);
  if (show_bbox) {
    if (!show_id) {
      for (int i = 0; i < meta_.size(); ++i) {
        meta_[i].id = -1;  // not drawn
      }
    }
    cv::Mat drawn_box;
    visual.draw_bbox(meta_,drawn,&drawn_box);
    return drawn_box;
  }
  return drawn;
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_vecmap(const Dtype* heatmaps, const int width, const int height) {
  return get_drawn_vecmap(heatmaps,width,height,false,false);
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_heatmap(const Dtype* heatmaps, const int width, const int height,
                           const bool show_bbox, const bool show_id) {
  Visualizer<Dtype> visual(image_, max_dis_size_);
  cv::Mat drawn;
  visual.draw_heatmap(heatmaps,width,height,&drawn);
  if (show_bbox) {
    if (!show_id) {
      for (int i = 0; i < meta_.size(); ++i) {
        meta_[i].id = -1;  // not drawn
      }
    }
    cv::Mat drawn_box;
    visual.draw_bbox(meta_,drawn,&drawn_box);
    return drawn_box;
  }
  return drawn;
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_heatmap(const Dtype* heatmaps, const int width, const int height) {
  return get_drawn_heatmap(heatmaps,width,height,false,false);
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_bbox(const bool show_id) {
  Visualizer<Dtype> visual(image_, max_dis_size_);
  if (!show_id) {
    for (int i = 0; i < meta_.size(); ++i) {
      meta_[i].id = -1;
    }
  }
  cv::Mat drawn;
  visual.draw_bbox(meta_,&drawn);
  return drawn;
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_skeleton(const bool show_bbox, const bool show_id) {
  Visualizer<Dtype> visual(image_, max_dis_size_);
  cv::Mat drawn;
  visual.draw_skeleton(meta_,&drawn);
  if (show_bbox) {
    if (!show_id) {
      for (int i = 0; i < meta_.size(); ++i) {
        meta_[i].id = -1;
      }
    }
    cv::Mat drawn_box;
    visual.draw_bbox(meta_,drawn,&drawn_box);
    return drawn_box;
  }
  return drawn;
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn_skeleton() {
  return get_drawn_skeleton(false,false);
}

template <typename Dtype>
cv::Mat ResFrame<Dtype>::get_drawn() {
  return get_drawn_skeleton(true, true);
}

INSTANTIATE_CLASS(ResFrame);
}
