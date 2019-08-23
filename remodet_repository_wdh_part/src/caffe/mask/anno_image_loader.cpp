#include "caffe/tracker/bounding_box.hpp"
#include "caffe/mask/anno_image_loader.hpp"
 
namespace caffe {

using std::vector;
using std::string;

// colors for mask [6 colors]
static const int COLOR_MAPS[18] = {255,85,0,255,170,130,70,255,0,170,0,130,85,255,0,0,255,85};

template <typename Dtype>
void AnnoImageLoader<Dtype>::LoadImage(const int image_num,
                                       cv::Mat* image) {
  const AnnoData<Dtype>& anno = annotations_[image_num];
  const string& image_file = anno.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::LoadAnnotation(const int image_num,
                                            cv::Mat* image,
                                            AnnoData<Dtype>* anno) {
  const AnnoData<Dtype>& annotation = annotations_[image_num];
  const string& image_file = annotation.img_path;
  *image = cv::imread(image_file.c_str());
  if (!image->data) {
    LOG(FATAL) << "Could not open or find image: " << image_file;
    return;
  }
  *anno = annotation;
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::ShowImages() {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    LoadImage(i, &image);
    cv::namedWindow("Imageshow", cv::WINDOW_AUTOSIZE);
    cv::imshow("Imageshow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::ShowAnnotations() {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    drawAnnotations(i, &image);
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);
    cv::imshow("ImageShow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::ShowAnnotationsRand() {
  while (true) {
    const int image_num = rand() % annotations_.size();
    cv::Mat image;
    drawAnnotations(image_num, &image);
    cv::namedWindow("ImageShow", cv::WINDOW_AUTOSIZE);
    cv::imshow("ImageShow", image);
    cv::waitKey(0);
  }
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::Saving(const std::string& output_folder) {
  for (int i = 0; i < annotations_.size(); ++i) {
    cv::Mat image;
    drawAnnotations(i, &image);
    // save
    int delim_pos = annotations_[i].img_path.find_last_of("/");
    const string& file_name = annotations_[i].img_path.substr(delim_pos+1, annotations_[i].img_path.length());
    const string& output_file = output_folder + "/" + file_name;
    LOG(INFO) << "saving image: " << file_name;
    imwrite(output_file, image);
  }
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::merge_from(const AnnoImageLoader<Dtype>* dst) {
  const std::vector<AnnoData<Dtype> >& dst_annos = dst->get_annotations();
  if (dst_annos.size() == 0) return;
  for (int i = 0; i < dst_annos.size(); ++i) {
    annotations_.push_back(dst_annos[i]);
  }
  LOG(INFO) << "Add " << dst_annos.size() << " Images.";
}

template <typename Dtype>
void AnnoImageLoader<Dtype>::drawAnnotations(const int image_num, cv::Mat* dst_image) {
  cv::Mat image;
  AnnoData<Dtype> anno;
  LoadAnnotation(image_num, &image, &anno);
  *dst_image = image.clone();
  if (anno.instances.size() == 0) return;
  for (int i = 0; i < anno.instances.size(); ++i) {
    const int r = COLOR_MAPS[3*(i % 6)];
    const int g = COLOR_MAPS[3*(i % 6) + 1];
    const int b = COLOR_MAPS[3*(i % 6) + 2];
    // draw boxes
    Instance<Dtype>& ins = anno.instances[i];
    const BoundingBox<Dtype>& bbox = ins.bbox;
    if (ins.iscrowd) {
      bbox.Draw(0,0,0,dst_image);
      continue;
    } else {
      bbox.Draw(r,g,b,dst_image);
    }
    // kps
    if (ins.kps_included) {
      const int num = ins.joint.joints.size();
      for(int k = 0; k < num; k++) {
        if(ins.joint.isVisible[k] <= 1)
          circle(*dst_image, ins.joint.joints[k], 3, CV_RGB(r,g,b), -1);
      }
    }
    // mask
    if (ins.mask_included) {
      const std::string& mask_path = ins.mask_path;
      cv::Mat mask = cv::imread(mask_path.c_str(),0);
      if (!mask.data) {
        LOG(FATAL) << "Could not open or find mask_image: " << mask_path;
        return;
      }
      CHECK_EQ(mask.cols, dst_image->cols);
      CHECK_EQ(mask.rows, dst_image->rows);
      float alpha = 0.5;
      for (int y = 0; y < dst_image->rows; ++y) {
        for (int x = 0; x < dst_image->cols; ++x) {
          cv::Vec3b& rgb = dst_image->at<cv::Vec3b>(y, x);
          int mask_val = mask.at<uchar>(y, x);
          if (mask_val > 128) {
            rgb[0] = (1-alpha)*rgb[0] + alpha*b;
            rgb[1] = (1-alpha)*rgb[1] + alpha*g;
            rgb[2] = (1-alpha)*rgb[2] + alpha*r;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(AnnoImageLoader);
}
