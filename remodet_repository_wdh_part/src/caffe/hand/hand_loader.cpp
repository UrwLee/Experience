#include "caffe/hand/hand_loader.hpp"

namespace caffe {

using std::vector;
using std::string;

template <typename Dtype>
void HandLoader<Dtype>::LoadImage(const int image_num,
                                  cv::Mat* image) {
  const MData<Dtype>& meta = annotations_[image_num];
  const string& image_file = meta.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
}

template <typename Dtype>
void HandLoader<Dtype>::LoadAnnotation(const int image_num,
                                       cv::Mat* image,
                                       MData<Dtype>* meta) {
  const MData<Dtype>& anno = annotations_[image_num];
  const string& image_file = anno.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
  *meta = anno;
}

template <typename Dtype>
void HandLoader<Dtype>::ShowHand() {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    MData<Dtype> meta;
    LoadAnnotation(i, &image, &meta);
    for (int j = 0; j < meta.ins.size(); ++j) {
      InsData<Dtype>& instance =  meta.ins[j];
      vector<BoundingBox<Dtype> > bbox;
      int num_hand;
      get_hand_bbox(instance,meta.img_width,meta.img_height,&bbox,&num_hand);
      if (num_hand > 0) {
        for (int p = 0; p < num_hand; ++p) {
          // draw bbox
          BoundingBox<Dtype>& box = bbox[p];
          cv::Point top_left_pt((int)box.x1_,(int)box.y1_);
          cv::Point bottom_right_pt((int)box.x2_,(int)box.y2_);
          cv::Mat image_copy = image.clone();
          cv::rectangle(image_copy, top_left_pt, bottom_right_pt, cv::Scalar(0,0,255), 2);
          cv::namedWindow("Handshow", cv::WINDOW_AUTOSIZE);
          cv::imshow("Handshow", image_copy);
          cv::waitKey(0);
        }
      }
    }
  }
}

template <typename Dtype>
void HandLoader<Dtype>::Saving(const std::string& output_folder, const std::string& prefix) {
  static int count = 0;
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    MData<Dtype> meta;
    LoadAnnotation(i, &image, &meta);
    for (int j = 0; j < meta.ins.size(); ++j) {
      InsData<Dtype>& instance =  meta.ins[j];
      vector<BoundingBox<Dtype> > bbox;
      int num_hand;
      get_hand_bbox(instance,meta.img_width,meta.img_height,&bbox,&num_hand);
      if (num_hand > 0) {
        for (int p = 0; p < num_hand; ++p) {
          BoundingBox<Dtype>& box = bbox[p];
          // crop bbox
          cv::Mat image_copy = image.clone();
          cv::Rect roi((int)box.x1_,(int)box.y1_,(int)box.get_width(),(int)box.get_height());
          cv::Mat image_roi = image_copy(roi);
          // resize
          // cv::Mat resized_roi;
          // cv::resize(image_roi, resized_roi, cv::Size(resized_width,resized_height), CV_INTER_CUBIC);
          // saving
          char buf[256];
          sprintf(buf, "%s/hand_%s_%08d.jpg", output_folder.c_str(),prefix.c_str(),count);
          LOG(INFO) << "saving image: " << buf;
          imwrite(buf, image_roi);
          ++count;
        }
      }
    }
  }
}

template <typename Dtype>
void HandLoader<Dtype>::merge_from(const HandLoader<Dtype>* dst) {
  const std::vector<MData<Dtype> >& dst_annos = dst->get_annotations();
  if (dst_annos.size() == 0) return;
  for (int i = 0; i < dst_annos.size(); ++i) {
    annotations_.push_back(dst_annos[i]);
  }
  LOG(INFO) << "Add " << dst_annos.size() << " Images.";
}

template <typename Dtype>
void HandLoader<Dtype>::get_hand_bbox(const InsData<Dtype>& ins, const int width, const int height,
                        vector<BoundingBox<Dtype> >* box, int* num_hands) {
  const Dtype as_thre = 1.2;
  const int size_thre = 32*32;
  *num_hands = 0;
  box->clear();
  // get boxes
  int re = 8, rw = 10;
  int le = 7, lw = 9;
  Dtype scale_hand = 1;
  if (ins.joint.isVisible[re] <= 1 && ins.joint.isVisible[rw] <= 1) {
    BoundingBox<Dtype> t_box;
    // draw right hand
    // four corners
    float xe = ins.joint.joints[re].x;
    float ye = ins.joint.joints[re].y;
    float xw = ins.joint.joints[rw].x;
    float yw = ins.joint.joints[rw].y;
    float dx = xw - xe;
    float dy = yw - ye;
    float norm = sqrt(dx*dx+dy*dy);
    dx /= norm;
    dy /= norm;
    cv::Point2f p1,p2,p3,p4;
    p1.x = xw + norm*scale_hand*dy*0.5;
    p1.y = yw - norm*scale_hand*dx*0.5;
    p2.x = xw - norm*scale_hand*dy*0.5;
    p2.y = yw + norm*scale_hand*dx*0.5;
    p3.x = p1.x + norm*scale_hand*dx;
    p3.y = p1.y + norm*scale_hand*dy;
    p4.x = p2.x + norm*scale_hand*dx;
    p4.y = p2.y + norm*scale_hand*dy;
    // get bbox
    float xmin,ymin,xmax,ymax;
    xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
    xmin = std::min(std::max(xmin,float(0.)),float(width-1));
    ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
    ymin = std::min(std::max(ymin,float(0.)),float(height-1));
    xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
    xmax = std::min(std::max(xmax,float(0.)),float(width-1));
    ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
    ymax = std::min(std::max(ymax,float(0.)),float(height-1));
    // get min square box include <...>
    float cx = (xmin+xmax)/2;
    float cy = (ymin+ymax)/2;
    float wh = std::max(xmax-xmin,ymax-ymin);
    xmin = cx - wh/2;
    xmax = cx + wh/2;
    ymin = cy - wh/2;
    ymax = cy + wh/2;
    xmin = std::min(std::max(xmin,float(0.)),float(width-1));
    ymin = std::min(std::max(ymin,float(0.)),float(height-1));
    xmax = std::min(std::max(xmax,float(0.)),float(width-1));
    ymax = std::min(std::max(ymax,float(0.)),float(height-1));
    float width = xmax - xmin;
    float height = ymax - ymin;
    float as = width / height;
    int area = (int)width * (int)height;
    if (as > Dtype(1)/as_thre && as < as_thre && area > size_thre) {
      t_box.x1_ = xmin;
      t_box.y1_ = ymin;
      t_box.x2_ = xmax;
      t_box.y2_ = ymax;
      box->push_back(t_box);
      (*num_hands)++;
    }
  }
  if (ins.joint.isVisible[le] <= 1 && ins.joint.isVisible[lw] <= 1) {
    BoundingBox<Dtype> t_box;
    // draw left hand
    // four corners
    float xe = ins.joint.joints[le].x;
    float ye = ins.joint.joints[le].y;
    float xw = ins.joint.joints[lw].x;
    float yw = ins.joint.joints[lw].y;
    float dx = xw - xe;
    float dy = yw - ye;
    float norm = sqrt(dx*dx+dy*dy);
    dx /= norm;
    dy /= norm;
    cv::Point2f p1,p2,p3,p4;
    p1.x = xw + norm*scale_hand*dy*0.5;
    p1.y = yw - norm*scale_hand*dx*0.5;
    p2.x = xw - norm*scale_hand*dy*0.5;
    p2.y = yw + norm*scale_hand*dx*0.5;
    p3.x = p1.x + norm*scale_hand*dx;
    p3.y = p1.y + norm*scale_hand*dy;
    p4.x = p2.x + norm*scale_hand*dx;
    p4.y = p2.y + norm*scale_hand*dy;
    // get bbox
    float xmin,ymin,xmax,ymax;
    xmin = std::min(std::min(std::min(p1.x,p2.x),p3.x),p4.x);
    xmin = std::min(std::max(xmin,float(0.)),float(width-1));
    ymin = std::min(std::min(std::min(p1.y,p2.y),p3.y),p4.y);
    ymin = std::min(std::max(ymin,float(0.)),float(height-1));
    xmax = std::max(std::max(std::max(p1.x,p2.x),p3.x),p4.x);
    xmax = std::min(std::max(xmax,float(0.)),float(width-1));
    ymax = std::max(std::max(std::max(p1.y,p2.y),p3.y),p4.y);
    ymax = std::min(std::max(ymax,float(0.)),float(height-1));
    // get min square box include <...>
    float cx = (xmin+xmax)/2;
    float cy = (ymin+ymax)/2;
    float wh = std::max(xmax-xmin,ymax-ymin);
    xmin = cx - wh/2;
    xmax = cx + wh/2;
    ymin = cy - wh/2;
    ymax = cy + wh/2;
    xmin = std::min(std::max(xmin,float(0.)),float(width-1));
    ymin = std::min(std::max(ymin,float(0.)),float(height-1));
    xmax = std::min(std::max(xmax,float(0.)),float(width-1));
    ymax = std::min(std::max(ymax,float(0.)),float(height-1));
    float width = xmax - xmin;
    float height = ymax - ymin;
    float as = width / height;
    int area = (int)width * (int)height;
    if (as > Dtype(1)/as_thre && as < as_thre && area > size_thre) {
      t_box.x1_ = xmin;
      t_box.y1_ = ymin;
      t_box.x2_ = xmax;
      t_box.y2_ = ymax;
      box->push_back(t_box);
      (*num_hands)++;
    }
  }
}

INSTANTIATE_CLASS(HandLoader);
}
