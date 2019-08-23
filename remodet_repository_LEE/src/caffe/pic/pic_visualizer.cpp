#include "caffe/pic/pic_visualizer.hpp"
#include "caffe/tracker/basic.hpp"

namespace caffe {

template <typename Dtype>
PicVisualizer<Dtype>::PicVisualizer(
                const PicData<Dtype>& meta_pic,
                const std::string& image_dir,
                const std::string& output_dir) {
  meta_pic_ = meta_pic;
  output_dir_ = output_dir;
  image_dir_ = image_dir;
}

template <typename Dtype>
void PicVisualizer<Dtype>::Save() {
  // open the image & get roi_patch
  const std::string image_file = image_dir_ + '/' + meta_pic_.image_path;
  cv::Mat raw_image = cv::imread(image_file.c_str());
  cv::Rect box_ROI((int)meta_pic_.bbox.x1_, (int)meta_pic_.bbox.y1_, (int)meta_pic_.bbox.get_width(), (int)meta_pic_.bbox.get_height());
  cv::Mat roi_image = raw_image(box_ROI);
  // get lines & colors
  cv::Scalar three_div(0,255,0);
  cv::Scalar two_div(0,0,255);
  cv::Point horiz_l1_pt1, horiz_l1_pt2;
  horiz_l1_pt1.x = 0; horiz_l1_pt1.y = meta_pic_.bgh / 3;
  horiz_l1_pt2.x = meta_pic_.bgw; horiz_l1_pt2.y = meta_pic_.bgh / 3;
  cv::Point horiz_l2_pt1, horiz_l2_pt2;
  horiz_l2_pt1.x = 0; horiz_l2_pt1.y = meta_pic_.bgh / 2;
  horiz_l2_pt2.x = meta_pic_.bgw; horiz_l2_pt2.y = meta_pic_.bgh / 2;
  cv::Point horiz_l3_pt1, horiz_l3_pt2;
  horiz_l3_pt1.x = 0; horiz_l3_pt1.y = meta_pic_.bgh * 2 / 3;
  horiz_l3_pt2.x = meta_pic_.bgw; horiz_l3_pt2.y = meta_pic_.bgh * 2 / 3;
  cv::Point vert_l1_pt1, vert_l1_pt2;
  vert_l1_pt1.x = meta_pic_.bgw / 3; vert_l1_pt1.y = 0;
  vert_l1_pt2.x = meta_pic_.bgw / 3; vert_l1_pt2.y = meta_pic_.bgh;
  cv::Point vert_l2_pt1, vert_l2_pt2;
  vert_l2_pt1.x = meta_pic_.bgw / 2; vert_l2_pt1.y = 0;
  vert_l2_pt2.x = meta_pic_.bgw / 2; vert_l2_pt2.y = meta_pic_.bgh;
  cv::Point vert_l3_pt1, vert_l3_pt2;
  vert_l3_pt1.x = meta_pic_.bgw * 2 / 3; vert_l3_pt1.y = 0;
  vert_l3_pt2.x = meta_pic_.bgw * 2 / 3; vert_l3_pt2.y = meta_pic_.bgh;
  // get pic results
  vector<int> pos;
  vector<Dtype> score;
  for (int i = 0; i < 3; ++i) {
    if (meta_pic_.pic[3*i] >= 0) {
      pos.push_back((int)meta_pic_.pic[3*i]);
      pos.push_back((int)meta_pic_.pic[3*i+1]);
      score.push_back(meta_pic_.pic[3*i+2]);
    }
  }
  // save each pic images.
  for (int i = 0; i < score.size(); ++i) {
    int cx = (Dtype)(pos[2*i] + 0.5) * meta_pic_.bgw / 32;
    int cy = (Dtype)(pos[2*i+1] + 0.5) * meta_pic_.bgh / 18;
    BoundingBox<Dtype> box;
    box.x1_ = cx - meta_pic_.bbox.get_width() / 2;
    box.y1_ = cy - meta_pic_.bbox.get_height() / 2;
    box.x2_ = box.x1_ + meta_pic_.bbox.get_width();
    box.y2_ = box.y1_ + meta_pic_.bbox.get_height();
    box.x1_ = std::min(std::max(box.x1_, (Dtype)0), (Dtype)(meta_pic_.bgw-1));
    box.x2_ = std::min(std::max(box.x2_, (Dtype)0), (Dtype)(meta_pic_.bgw-1));
    box.y1_ = std::min(std::max(box.y1_, (Dtype)0), (Dtype)(meta_pic_.bgh-1));
    box.y2_ = std::min(std::max(box.y2_, (Dtype)0), (Dtype)(meta_pic_.bgh-1));
    // copy roi to bg
    cv::Mat bg_image = cv::Mat(meta_pic_.bgh, meta_pic_.bgw, raw_image.type(), cv::Scalar(255, 255, 255));
    cv::Rect output_rect((int)box.x1_, (int)box.y1_, (int)box.get_width(), (int)box.get_height());
    cv::Mat output_roi_image = bg_image(output_rect);
    roi_image.copyTo(output_roi_image);
    // draw lines
    cv::line(bg_image,horiz_l1_pt1,horiz_l1_pt2,three_div,2);
    cv::line(bg_image,horiz_l2_pt1,horiz_l2_pt2,two_div,2);
    cv::line(bg_image,horiz_l3_pt1,horiz_l3_pt2,three_div,2);
    cv::line(bg_image,vert_l1_pt1,vert_l1_pt2,three_div,2);
    cv::line(bg_image,vert_l2_pt1,vert_l2_pt2,two_div,2);
    cv::line(bg_image,vert_l3_pt1,vert_l3_pt2,three_div,2);
    // draw score
    cv::Point score_point(30, 30);
    char buffer[50];
    snprintf(buffer, sizeof(buffer), "%.3f", score[i]);
    cv::putText(bg_image, buffer, score_point, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);
    // show
    // cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    // cv::imshow("image", bg_image);
    // cv::waitKey(0);
    // save images
    char imagename[256];
    const string image_name = meta_pic_.image_path.substr(0, meta_pic_.image_path.length() - 4);
    sprintf(imagename, "%s/%s_%d.jpg", output_dir_.c_str(), image_name.c_str(), i);
    LOG(INFO) << "Save image: " << imagename;
    cv::imwrite(imagename, bg_image);
  }
}

INSTANTIATE_CLASS(PicVisualizer);
}
