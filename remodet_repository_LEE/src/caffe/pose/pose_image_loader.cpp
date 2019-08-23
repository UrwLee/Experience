#include "caffe/tracker/bounding_box.hpp"
#include "caffe/pose/pose_image_loader.hpp"

namespace caffe {

using std::vector;
using std::string;

template <typename Dtype>
void PoseImageLoader<Dtype>::LoadImage(const int image_num,
                                  cv::Mat* image) {
  const MetaData<Dtype>& annotation = annotations_[image_num];
  const string& image_file = annotation.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::LoadAnnotation(const int image_num,
                                       cv::Mat* image,
                                       MetaData<Dtype>* meta) {
  const MetaData<Dtype>& annotation = annotations_[image_num];
  const string& image_file = annotation.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
  *meta = annotation;
}

template <typename Dtype>
void PoseImageLoader<Dtype>::ShowImages() {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    LoadImage(i, &image);
    cv::namedWindow("Imageshow", cv::WINDOW_AUTOSIZE);
    cv::imshow("Imageshow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::ShowAnnotations(const bool show_bbox) {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    drawAnnotations(i, show_bbox, &image);
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);
    cv::imshow("ImageShow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::ShowAnnotationsRand(const bool show_bbox) {
  while (true) {
    const int image_num = rand() % annotations_.size();
    cv::Mat image;
    drawAnnotations(image_num, show_bbox, &image);
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);
    cv::imshow("ImageShow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::Saving(const std::string& output_folder, const bool show_bbox) {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    drawAnnotations(i, show_bbox, &image);
    // save
    int delim_pos = annotations_[i].img_path.find_last_of("/");
    const string& file_name = annotations_[i].img_path.substr(delim_pos+1, annotations_[i].img_path.length());
    const string& output_file = output_folder + "/" + file_name;
    LOG(INFO) << "saving image: " << file_name;
    imwrite(output_file, image);
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::merge_from(const PoseImageLoader<Dtype>* dst) {
  const std::vector<MetaData<Dtype> >& dst_annos = dst->get_annotations();
  if (dst_annos.size() == 0) return;
  for (int i = 0; i < dst_annos.size(); ++i) {
    annotations_.push_back(dst_annos[i]);
  }
  LOG(INFO) << "Add " << dst_annos.size() << " Images.";
}

template <typename Dtype>
void PoseImageLoader<Dtype>::drawAnnotations(const int image_num, const bool show_bbox, cv::Mat* dst_image) {
  cv::Mat image;
  MetaData<Dtype> meta;
  LoadAnnotation(image_num, &image, &meta);
  *dst_image = image.clone();
  // 绘制
  if (show_bbox) {
    // 绿色ｂｏｘ
    drawbox(meta, dst_image);
  }
  // 红色关节点
  drawkps(meta, dst_image);
}

template <typename Dtype>
void PoseImageLoader<Dtype>::drawbox(const MetaData<Dtype>& meta, cv::Mat* image_out) {
  // bbox
  const BoundingBox<Dtype>& bbox = meta.bbox;
  bbox.Draw(0,255,0,image_out);
  // bbox of others
  for (int i = 0; i < meta.bbox_others.size(); ++i) {
    const BoundingBox<Dtype>& bbox_op = meta.bbox_others[i];
    bbox_op.Draw(0,255,0,image_out);
  }
}

template <typename Dtype>
void PoseImageLoader<Dtype>::drawkps(const MetaData<Dtype>& meta, cv::Mat* image_out) {
  const int num_kps = meta.joint_self.joints.size();
  // draw self
  for(int i = 0; i < num_kps; i++) {
    if(meta.joint_self.isVisible[i] <= 1)
      circle(*image_out, meta.joint_self.joints[i], 5, CV_RGB(255,0,0), -1);
  }
  // joints of others
  for(int p = 0; p < meta.numOtherPeople; p++) {
    for(int i = 0; i < num_kps; i++) {
      if(meta.joint_others[p].isVisible[i] <= 1)
        circle(*image_out, meta.joint_others[p].joints[i], 5, CV_RGB(255,0,0), -1);
    }
  }
}

INSTANTIATE_CLASS(PoseImageLoader);
}
