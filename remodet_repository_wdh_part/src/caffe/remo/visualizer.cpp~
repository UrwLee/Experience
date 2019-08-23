#include "caffe/remo/visualizer.hpp"

namespace caffe {

static bool DRAW_HANDS = false;

template <typename Dtype>
Visualizer<Dtype>::Visualizer(cv::Mat& image, int max_display_size) {
  const int width = image.cols;
  const int height = image.rows;
  const int maxsize = (width > height) ? width : height;
  const Dtype ratio = (Dtype)max_display_size / maxsize;
  const int display_width = static_cast<int>(width * ratio);
  const int display_height = static_cast<int>(height * ratio);
  cv::resize(image, image_, cv::Size(display_width,display_height), cv::INTER_LINEAR);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(const BoundingBox<Dtype>& bbox, const int id) {
  draw_bbox(0,255,0,bbox,id);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(int r, int g, int b, const BoundingBox<Dtype>& bbox,
                                  const int id) {
  cv::Point top_left_pt(static_cast<int>(bbox.x1_ * image_.cols),
                        static_cast<int>(bbox.y1_ * image_.rows));
  cv::Point bottom_right_pt(static_cast<int>(bbox.x2_ * image_.cols),
                        static_cast<int>(bbox.y2_ * image_.rows));
  cv::rectangle(image_, top_left_pt, bottom_right_pt, cv::Scalar(b,g,r), 3);
  // draw person id
  if (id >= 1) {
    cv::Point bottom_left_pt1(static_cast<int>(bbox.x1_ * image_.cols + 5),
                             static_cast<int>(bbox.y2_ * image_.rows - 5));
    cv::Point bottom_left_pt2(static_cast<int>(bbox.x1_ * image_.cols + 3),
                            static_cast<int>(bbox.y2_ * image_.rows - 3));
    char buffer[50];
    snprintf(buffer, sizeof(buffer), "%d", id);
    cv::putText(image_, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(image_, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
  }
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(const BoundingBox<Dtype>& bbox, const int id, cv::Mat* out_image) {
  draw_bbox(0,255,0,bbox,id,out_image);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(int r, int g, int b, const BoundingBox<Dtype>& bbox,
                                  const int id, cv::Mat* out_image) {
  cv::Mat image;
  image_.copyTo(image);
  cv::Point top_left_pt(static_cast<int>(bbox.x1_ * image.cols),
                        static_cast<int>(bbox.y1_ * image.rows));
  cv::Point bottom_right_pt(static_cast<int>(bbox.x2_ * image.cols),
                        static_cast<int>(bbox.y2_ * image.rows));
  cv::rectangle(image, top_left_pt, bottom_right_pt, cv::Scalar(b,g,r), 3);
  // draw person id
  if (id >= 1) {
    cv::Point bottom_left_pt1(static_cast<int>(bbox.x1_ * image.cols + 5),
                             static_cast<int>(bbox.y2_ * image.rows - 5));
    cv::Point bottom_left_pt2(static_cast<int>(bbox.x1_ * image.cols + 3),
                            static_cast<int>(bbox.y2_ * image.rows - 3));
    char buffer[50];
    snprintf(buffer, sizeof(buffer), "%d", id);
    cv::putText(image, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
    cv::putText(image, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
  }
  // output
  *out_image = image;
}

template <typename Dtype>
void Visualizer<Dtype>::draw_hand(const PMeta<Dtype>& meta, cv::Mat* image) {
  int re = 3, rw = 4;
  // int le = 6, lw = 7;
  Dtype scale_hand = 1;
  // RIGHT
  if (meta.kps[re].v > 0.05 && meta.kps[rw].v > 0.05) {
    // draw right hand
    // four corners
    float xe = meta.kps[re].x;
    float ye = meta.kps[re].y;
    float xw = meta.kps[rw].x;
    float yw = meta.kps[rw].y;
    float dx = xw - xe;
    float dy = yw - ye;
    float norm = sqrt(dx*dx+dy*dy);
    dx /= norm;
    dy /= norm;
    cv::Point2f p1,p2,p3,p4;
    p1.x = xw + norm*scale_hand*dy*0.5*9/16;
    p1.y = yw - norm*scale_hand*dx*0.5*9/16;
    p2.x = xw - norm*scale_hand*dy*0.5*9/16;
    p2.y = yw + norm*scale_hand*dx*0.5*9/16;
    p3.x = p1.x + norm*scale_hand*dx;
    p3.y = p1.y + norm*scale_hand*dy;
    p4.x = p2.x + norm*scale_hand*dx;
    p4.y = p2.y + norm*scale_hand*dy;
    // get bbox
    float xmin,ymin,xmax,ymax;
    xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    // get min square box include <...>
    float cx = (xmin+xmax)/2;
    float cy = (ymin+ymax)/2;
    float wh = std::max(xmax-xmin,(ymax-ymin)*9/16);
    xmin = cx - wh/2;
    xmax = cx + wh/2;
    ymin = cy - wh*16/9/2;
    ymax = cy + wh*16/9/2;
    xmin = std::min(std::max(xmin,float(0.)),float(1.));
    ymin = std::min(std::max(ymin,float(0.)),float(1.));
    xmax = std::min(std::max(xmax,float(0.)),float(1.));
    ymax = std::min(std::max(ymax,float(0.)),float(1.));
    cv::Point top_left_pt(xmin*image->cols,ymin*image->rows);
    cv::Point bottom_right_pt(xmax*image->cols,ymax*image->rows);
    cv::rectangle(*image, top_left_pt, bottom_right_pt, cv::Scalar(0,0,255), 3);
  }
  // LEFT
  // if (meta.kps[le].v > 0.05 && meta.kps[lw].v > 0.05) {
  //   // draw left hand
  //   float xe = meta.kps[le].x;
  //   float ye = meta.kps[le].y;
  //   float xw = meta.kps[lw].x;
  //   float yw = meta.kps[lw].y;
  //   float dx = xw - xe;
  //   float dy = yw - ye;
  //   float norm = sqrt(dx*dx+dy*dy);
  //   dx /= norm;
  //   dy /= norm;
  //   cv::Point2f p1,p2,p3,p4;
  //   p1.x = xw + norm*scale_hand*dy*0.5*9/16;
  //   p1.y = yw - norm*scale_hand*dx*0.5*9/16;
  //   p2.x = xw - norm*scale_hand*dy*0.5*9/16;
  //   p2.y = yw + norm*scale_hand*dx*0.5*9/16;
  //   p3.x = p1.x + norm*scale_hand*dx;
  //   p3.y = p1.y + norm*scale_hand*dy;
  //   p4.x = p2.x + norm*scale_hand*dx;
  //   p4.y = p2.y + norm*scale_hand*dy;
  //   // get bbox
  //   float xmin,ymin,xmax,ymax;
  //   xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
  //   xmin = std::min(std::max(xmin,float(0.)),float(1.));
  //   ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
  //   ymin = std::min(std::max(ymin,float(0.)),float(1.));
  //   xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
  //   xmax = std::min(std::max(xmax,float(0.)),float(1.));
  //   ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
  //   ymax = std::min(std::max(ymax,float(0.)),float(1.));
  //   // get min square box include <...>
  //   float cx = (xmin+xmax)/2;
  //   float cy = (ymin+ymax)/2;
  //   float wh = std::max(xmax-xmin,(ymax-ymin)*9/16);
  //   xmin = cx - wh/2;
  //   xmax = cx + wh/2;
  //   ymin = cy - wh*16/9/2;
  //   ymax = cy + wh*16/9/2;
  //   xmin = std::min(std::max(xmin,float(0.)),float(1.));
  //   ymin = std::min(std::max(ymin,float(0.)),float(1.));
  //   xmax = std::min(std::max(xmax,float(0.)),float(1.));
  //   ymax = std::min(std::max(ymax,float(0.)),float(1.));
  //   cv::Point top_left_pt(xmin*image->cols,ymin*image->rows);
  //   cv::Point bottom_right_pt(xmax*image->cols,ymax*image->rows);
  //   cv::rectangle(*image, top_left_pt, bottom_right_pt, cv::Scalar(0,0,255), 3);
  // }
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(const vector<PMeta<Dtype> >& meta) {
  draw_bbox(0,255,0,DRAW_HANDS,meta);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta) {
  if (meta.size() == 0) return;
  for (int i = 0; i < meta.size(); ++i) {
    const BoundingBox<Dtype>& bbox = meta[i].bbox;
    const int id = meta[i].id;
    const Dtype similarity = meta[i].similarity;
    const Dtype max_back_similarity = meta[i].max_back_similarity;
    int xmin = (int)(bbox.x1_*image_.cols);
    int xmax = (int)(bbox.x2_*image_.cols);
    int ymin = (int)(bbox.y1_*image_.rows);
    int ymax = (int)(bbox.y2_*image_.rows);
    xmin = std::max(std::min(xmin,image_.cols-1),0);
    xmax = std::max(std::min(xmax,image_.cols-1),0);
    ymin = std::max(std::min(ymin,image_.rows-1),0);
    ymax = std::max(std::min(ymax,image_.rows-1),0);
    cv::Point top_left_pt(xmin,ymin);
    cv::Point bottom_right_pt(xmax,ymax);
    cv::rectangle(image_, top_left_pt, bottom_right_pt, cv::Scalar(b,g,r), 3);
    //--------------------------------------------------------------------------
    if (draw_hands) {
      draw_hand(meta[i],&image_);
    }
    //--------------------------------------------------------------------------
    if (id >= 1) {
      // id
      cv::Point bottom_left_pt1(xmin+5,ymax-5);
      cv::Point bottom_left_pt2(xmin+3,ymax-3);
      char buffer[50];
      snprintf(buffer, sizeof(buffer), "%d", id);
      cv::putText(image_, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
      cv::putText(image_, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
      // similarity
      cv::Point bottom_right_pt1(xmax-150,ymax-5);
      cv::Point bottom_right_pt2(xmax-148,ymax-3);
      char sm_buffer[50];
      snprintf(sm_buffer, sizeof(sm_buffer), "%.2f/%.2f", (float)similarity, (float)max_back_similarity);
      cv::putText(image_, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
      cv::putText(image_, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
    }
  }
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(const vector<PMeta<Dtype> >& meta,
                                  cv::Mat* out_image) {
  draw_bbox(0,255,0,DRAW_HANDS,meta,out_image);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta,
                                  cv::Mat* out_image) {
  cv::Mat image;
  image_.copyTo(image);
  if (meta.size() > 0) {
    for (int i = 0; i < meta.size(); ++i) {
      const BoundingBox<Dtype>& bbox = meta[i].bbox;
      const int id = meta[i].id;
      const Dtype similarity = meta[i].similarity;
      const Dtype max_back_similarity = meta[i].max_back_similarity;
      int xmin = (int)(bbox.x1_*image.cols);
      int xmax = (int)(bbox.x2_*image.cols);
      int ymin = (int)(bbox.y1_*image.rows);
      int ymax = (int)(bbox.y2_*image.rows);
      xmin = std::max(std::min(xmin,image.cols-1),0);
      xmax = std::max(std::min(xmax,image.cols-1),0);
      ymin = std::max(std::min(ymin,image.rows-1),0);
      ymax = std::max(std::min(ymax,image.rows-1),0);
      cv::Point top_left_pt(xmin,ymin);
      cv::Point bottom_right_pt(xmax,ymax);
      cv::rectangle(image, top_left_pt, bottom_right_pt, cv::Scalar(b,g,r), 3);
      //------------------------------------------------------------------------
      // draw hands
      if (draw_hands) {
        draw_hand(meta[i],&image);
      }
      //------------------------------------------------------------------------
      if (id >= 1) {
        cv::Point bottom_left_pt1(xmin+5,ymax-5);
        cv::Point bottom_left_pt2(xmin+3,ymax-3);
        char buffer[50];
        snprintf(buffer, sizeof(buffer), "%d", id);
        cv::putText(image, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(image, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
        // similarity
        cv::Point bottom_right_pt1(xmax-150,ymax-5);
        cv::Point bottom_right_pt2(xmax-148,ymax-3);
        char sm_buffer[50];
        snprintf(sm_buffer, sizeof(sm_buffer), "%.2f/%.2f", (float)similarity, (float)max_back_similarity);
        cv::putText(image, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(image, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
      }
    }
    *out_image = image;
  } else {
    *out_image = image;
  }
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(const vector<PMeta<Dtype> >& meta,
                                  const cv::Mat& src_image, cv::Mat* out_image) {
  draw_bbox(0,255,0,DRAW_HANDS,meta,src_image,out_image);
}

template <typename Dtype>
void Visualizer<Dtype>::draw_bbox(int r, int g, int b, bool draw_hands, const vector<PMeta<Dtype> >& meta,
                                  const cv::Mat& src_image, cv::Mat* out_image) {
  cv::Mat image;
  src_image.copyTo(image);
  if (meta.size() > 0) {
    for (int i = 0; i < meta.size(); ++i) {
      const BoundingBox<Dtype>& bbox = meta[i].bbox;
      const int id = meta[i].id;
      const Dtype similarity = meta[i].similarity;
      const Dtype max_back_similarity = meta[i].max_back_similarity;
      int xmin = (int)(bbox.x1_*image.cols);
      int xmax = (int)(bbox.x2_*image.cols);
      int ymin = (int)(bbox.y1_*image.rows);
      int ymax = (int)(bbox.y2_*image.rows);
      xmin = std::max(std::min(xmin,image.cols-1),0);
      xmax = std::max(std::min(xmax,image.cols-1),0);
      ymin = std::max(std::min(ymin,image.rows-1),0);
      ymax = std::max(std::min(ymax,image.rows-1),0);
      cv::Point top_left_pt(xmin,ymin);
      cv::Point bottom_right_pt(xmax,ymax);
      cv::rectangle(image, top_left_pt, bottom_right_pt, cv::Scalar(b,g,r), 3);
      // -----------------------------------------------------------------------
      // draw hands
      if (draw_hands) {
        draw_hand(meta[i],&image);
      }
      // -----------------------------------------------------------------------
      if (id >= 1) {
        cv::Point bottom_left_pt1(xmin+5,ymax-5);
        cv::Point bottom_left_pt2(xmin+3,ymax-3);
        char buffer[50];
        snprintf(buffer, sizeof(buffer), "%d", id);
        cv::putText(image, buffer, bottom_left_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(image, buffer, bottom_left_pt2, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
        // similarity
        cv::Point bottom_right_pt1(xmax-150,ymax-5);
        cv::Point bottom_right_pt2(xmax-148,ymax-3);
        char sm_buffer[50];
        snprintf(sm_buffer, sizeof(sm_buffer), "%.2f/%.2f", (float)similarity, (float)max_back_similarity);
        cv::putText(image, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(image, sm_buffer, bottom_right_pt1, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
      }
    }
    *out_image = image;
  } else {
    *out_image = image;
  }
}

template <typename Dtype>
void Visualizer<Dtype>::save(const cv::Mat& image, const std::string& save_dir, const int image_id) {
  char imagename [256];
  sprintf(imagename, "%s/%10d.jpg", save_dir.c_str(), image_id);
  cv::imwrite(imagename, image);
}

template <typename Dtype>
void Visualizer<Dtype>::save(const std::string& save_dir, const int image_id) {
  char imagename [256];
  sprintf(imagename, "%s/%10d.jpg", save_dir.c_str(), image_id);
  cv::imwrite(imagename, image_);
}

INSTANTIATE_CLASS(Visualizer);
}
