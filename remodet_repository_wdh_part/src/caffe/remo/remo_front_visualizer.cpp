#include "caffe/remo/remo_front_visualizer.hpp"

namespace caffe {

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                              int cam_id, int width, int height,
                              int resized_width, int resized_height,
                              const std::string& network_proto,
                              const std::string& caffe_model,
                              const bool mode, const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size,
                              const DrawnMode drawn_mode) {
  frame_reader_.reset(new FrameReader<Dtype>(cam_id, width, height, resized_width, resized_height));
  net_wrapper_.reset(new NetWrapper<Dtype>(network_proto,caffe_model,mode,gpu_id,proposals,heatmaps,max_dis_size));
  drawn_mode_ = drawn_mode;
  frame_loader_time_ = 0;
  frame_loader_time_sum_ = 0;
  display_time_ = 0;
  display_time_sum_ = 0;
  frames_ = 0;
  //
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                              int cam_id, int width, int height,
                              int resized_width, int resized_height,
                              const std::string& network_proto,
                              const std::string& caffe_model,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size,
                              const DrawnMode drawn_mode) {
  RemoFrontVisualizer(cam_id, width, height,resized_width,resized_height,
                      network_proto,caffe_model,true,gpu_id,proposals,heatmaps,
                      max_dis_size,drawn_mode);
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                              const std::string& video_file,
                              int start_frame,
                              int resized_width, int resized_height,
                              const std::string& network_proto,
                              const std::string& caffe_model,
                              const bool mode, const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size,
                              const DrawnMode drawn_mode) {
  frame_reader_.reset(new FrameReader<Dtype>(video_file, start_frame, resized_width, resized_height));
  net_wrapper_.reset(new NetWrapper<Dtype>(network_proto,caffe_model,mode,gpu_id,proposals,heatmaps,max_dis_size));
  drawn_mode_ = drawn_mode;
  frame_loader_time_ = 0;
  frame_loader_time_sum_ = 0;
  display_time_ = 0;
  display_time_sum_ = 0;
  frames_ = 0;
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                              const std::string& video_file,
                              int start_frame,
                              int resized_width, int resized_height,
                              const std::string& network_proto,
                              const std::string& caffe_model,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size,
                              const DrawnMode drawn_mode) {
  RemoFrontVisualizer(video_file,start_frame,resized_width,resized_height,
                      network_proto,caffe_model,true,gpu_id,
                      proposals,heatmaps,max_dis_size,drawn_mode);
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                              const std::string& ip_addr,
                              int resized_width, int resized_height,
                              const std::string& network_proto,
                              const std::string& caffe_model,
                              const int gpu_id,
                              const std::string& proposals,
                              const std::string& heatmaps,
                              const int max_dis_size,
                              const DrawnMode drawn_mode) {
  frame_reader_.reset(new FrameReader<Dtype>(ip_addr, resized_width, resized_height));
  net_wrapper_.reset(new NetWrapper<Dtype>(network_proto,caffe_model,true,gpu_id,proposals,heatmaps,max_dis_size));
  drawn_mode_ = drawn_mode;
  frame_loader_time_ = 0;
  frame_loader_time_sum_ = 0;
  display_time_ = 0;
  display_time_sum_ = 0;
  frames_ = 0;
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                           const std::string& network_proto,
                           const std::string& caffe_model) {
  frame_reader_.reset(new FrameReader<Dtype>(0, 1280, 720, 512, 288));
  net_wrapper_.reset(new NetWrapper<Dtype>(network_proto,caffe_model,true,0,"proposals","resized_map",1000));
  drawn_mode_ = SKELETON_BOX;
  frame_loader_time_ = 0;
  frame_loader_time_sum_ = 0;
  display_time_ = 0;
  display_time_sum_ = 0;
  frames_ = 0;
}

template <typename Dtype>
RemoFrontVisualizer<Dtype>::RemoFrontVisualizer(
                               const std::string& video_file,
                               const std::string& network_proto,
                               const std::string& caffe_model) {
  frame_reader_.reset(new FrameReader<Dtype>(video_file, 0, 512, 288));
  net_wrapper_.reset(new NetWrapper<Dtype>(network_proto,caffe_model,true,0,"proposals","resized_map",1000));
  drawn_mode_ = SKELETON_BOX;
  frame_loader_time_ = 0;
  frame_loader_time_sum_ = 0;
  display_time_ = 0;
  display_time_sum_ = 0;
  frames_ = 0;
}

template <typename Dtype>
int RemoFrontVisualizer<Dtype>::step(std::vector<std::vector<Dtype> >* meta) {
  // data reader
  caffe::Timer frame_loader;
  frame_loader.Start();
  DataFrame<Dtype> curr_frame;
  if(frame_reader_->pop(&curr_frame)) {
    return 1;
  }
  frame_loader_time_sum_ += frame_loader.MicroSeconds();
  // get netwrapper & drawn image
  caffe::Timer disp_loader;
  switch(drawn_mode_) {
    case NONE: {
      net_wrapper_->get_meta(curr_frame,meta);
      break;
    }
    case BOX: {
      cv::Mat image_box = net_wrapper_->get_bbox(curr_frame,false,meta);
      disp_loader.Start();
      if(!image_box.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_box);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case BOX_ID: {
      cv::Mat image_box_id = net_wrapper_->get_bbox(curr_frame,true,meta);
      disp_loader.Start();
      if(!image_box_id.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_box_id);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case SKELETON: {
      cv::Mat image_skeleton = net_wrapper_->get_skeleton(curr_frame,false,false,meta);
      disp_loader.Start();
      if(!image_skeleton.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_skeleton);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case SKELETON_BOX: {
      cv::Mat image_skeleton_box = net_wrapper_->get_skeleton(curr_frame,true,false,meta);
      disp_loader.Start();
      if(!image_skeleton_box.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_skeleton_box);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case SKELETON_BOX_ID: {
      cv::Mat image_skeleton_box_id = net_wrapper_->get_skeleton(curr_frame,true,true,meta);
      disp_loader.Start();
      if(!image_skeleton_box_id.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_skeleton_box_id);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case HEATMAP: {
      cv::Mat image_heatmap = net_wrapper_->get_heatmap(curr_frame,false,false,meta);
      disp_loader.Start();
      if(!image_heatmap.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_heatmap);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case HEATMAP_BOX: {
      cv::Mat image_heatmap_box = net_wrapper_->get_heatmap(curr_frame,true,false,meta);
      disp_loader.Start();
      if(!image_heatmap_box.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_heatmap_box);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case HEATMAP_BOX_ID: {
      cv::Mat image_heatmap_box_id = net_wrapper_->get_heatmap(curr_frame,true,true,meta);
      disp_loader.Start();
      if(!image_heatmap_box_id.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_heatmap_box_id);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case VECMAP: {
      cv::Mat image_vecmap = net_wrapper_->get_vecmap(curr_frame,false,false,meta);
      disp_loader.Start();
      if(!image_vecmap.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_vecmap);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case VECMAP_BOX: {
      cv::Mat image_vecmap_box = net_wrapper_->get_vecmap(curr_frame,true,false,meta);
      disp_loader.Start();
      if(!image_vecmap_box.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_vecmap_box);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    case VECMAP_BOX_ID: {
      cv::Mat image_vecmap_box_id = net_wrapper_->get_vecmap(curr_frame,true,true,meta);
      disp_loader.Start();
      if(!image_vecmap_box_id.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_vecmap_box_id);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
    default: {
      // use SKELETON_BOX method
      cv::Mat image_default = net_wrapper_->get_skeleton(curr_frame,true,false,meta);
      disp_loader.Start();
      if(!image_default.data ) {
        LOG(FATAL) << "Error - Failed for reading the Mat data.";
      } else {
        cv::namedWindow("Remo", cv::WINDOW_AUTOSIZE);
        cv::imshow( "Remo", image_default);
      }
      display_time_sum_ += disp_loader.MicroSeconds();
      break;
    }
  }
  if (drawn_mode_ != NONE) {
    cv::waitKey(1);
  }
  // stat time
  frames_ = net_wrapper_->get_frames();
  if (frames_ % 30 == 0) {
    frame_loader_time_ = frame_loader_time_sum_ / 30;
    frame_loader_time_sum_ = 0;
    display_time_ = display_time_sum_ / 30;
    display_time_sum_ = 0;
  }
  // return
  return 0;
}

template <typename Dtype>
int RemoFrontVisualizer<Dtype>::step(cv::Mat* image, std::vector<std::vector<Dtype> >* meta) {
  // data reader
  caffe::Timer frame_loader;
  frame_loader.Start();
  DataFrame<Dtype> curr_frame;
  if(frame_reader_->pop(&curr_frame)) {
    return 1;
  }
  frame_loader_time_sum_ += frame_loader.MicroSeconds();
  // get netwrapper & drawn image
  switch(drawn_mode_) {
    case NONE: {
      LOG(INFO) << "No images will be output in mode (NONE).";
      net_wrapper_->get_meta(curr_frame,meta);
      break;
    }
    case BOX: {
      cv::Mat image_box = net_wrapper_->get_bbox(curr_frame,false,meta);
      *image = image_box;
      break;
    }
    case BOX_ID: {
      cv::Mat image_box_id = net_wrapper_->get_bbox(curr_frame,true,meta);
      *image = image_box_id;
      break;
    }
    case SKELETON: {
      cv::Mat image_skeleton = net_wrapper_->get_skeleton(curr_frame,false,false,meta);
      *image = image_skeleton;
      break;
    }
    case SKELETON_BOX: {
      cv::Mat image_skeleton_box = net_wrapper_->get_skeleton(curr_frame,true,false,meta);
      *image = image_skeleton_box;
      break;
    }
    case SKELETON_BOX_ID: {
      cv::Mat image_skeleton_box_id = net_wrapper_->get_skeleton(curr_frame,true,true,meta);
      *image = image_skeleton_box_id;
      break;
    }
    case HEATMAP: {
      cv::Mat image_heatmap = net_wrapper_->get_heatmap(curr_frame,false,false,meta);
      *image = image_heatmap;
      break;
    }
    case HEATMAP_BOX: {
      cv::Mat image_heatmap_box = net_wrapper_->get_heatmap(curr_frame,true,false,meta);
      *image = image_heatmap_box;
      break;
    }
    case HEATMAP_BOX_ID: {
      cv::Mat image_heatmap_box_id = net_wrapper_->get_heatmap(curr_frame,true,true,meta);
      *image = image_heatmap_box_id;
      break;
    }
    case VECMAP: {
      cv::Mat image_vecmap = net_wrapper_->get_vecmap(curr_frame,false,false,meta);
      *image = image_vecmap;
      break;
    }
    case VECMAP_BOX: {
      cv::Mat image_vecmap_box = net_wrapper_->get_vecmap(curr_frame,true,false,meta);
      *image = image_vecmap_box;
      break;
    }
    case VECMAP_BOX_ID: {
      cv::Mat image_vecmap_box_id = net_wrapper_->get_vecmap(curr_frame,true,true,meta);
      *image = image_vecmap_box_id;
      break;
    }
    default: {
      cv::Mat image_default = net_wrapper_->get_skeleton(curr_frame,true,false,meta);
      *image = image_default;
      break;
    }
  }
  // stat time
  frames_ = net_wrapper_->get_frames();
  if (frames_ % 30 == 0) {
    frame_loader_time_ = frame_loader_time_sum_ / 30;
    frame_loader_time_sum_ = 0;
    display_time_ = display_time_sum_ / 30;
    display_time_sum_ = 0;
  }
  return 0;
}

template <typename Dtype>
int RemoFrontVisualizer<Dtype>::save(const int interval,
                                     const bool save_orig,
                                     const std::string& orig_dir,
                                     const std::string& process_dir) {
  // data reader
  caffe::Timer frame_loader;
  frame_loader.Start();
  DataFrame<Dtype> curr_frame;
  // reader
  for (int i = 0; i < interval; ++i) {
    if(frame_reader_->pop(&curr_frame)) {
      return 1;
    }
  }
  frame_loader_time_sum_ += frame_loader.MicroSeconds();
  // save orig
  if (save_orig) {
    char save_orig_name[256];
    sprintf(save_orig_name, "%s/%08d.jpg", orig_dir.c_str(), frames_);
    cv::imwrite(save_orig_name, curr_frame.get_ori_image());
  }
  // return meta
  std::vector<std::vector<Dtype> > meta;
  // save processed image
  char save_pro_name[256];
  sprintf(save_pro_name, "%s/%08d.jpg", process_dir.c_str(), frames_);
  switch(drawn_mode_) {
    case NONE: {
      LOG(INFO) << "No images will be output in mode (NONE).";
      net_wrapper_->get_meta(curr_frame,&meta);
      break;
    }
    case BOX: {
      cv::Mat image_box = net_wrapper_->get_bbox(curr_frame,false,&meta);
      cv::imwrite(save_pro_name, image_box);
      break;
    }
    case BOX_ID: {
      cv::Mat image_box_id = net_wrapper_->get_bbox(curr_frame,true,&meta);
      cv::imwrite(save_pro_name, image_box_id);
      break;
    }
    case SKELETON: {
      cv::Mat image_skeleton = net_wrapper_->get_skeleton(curr_frame,false,false,&meta);
      cv::imwrite(save_pro_name, image_skeleton);
      break;
    }
    case SKELETON_BOX: {
      cv::Mat image_skeleton_box = net_wrapper_->get_skeleton(curr_frame,true,false,&meta);
      cv::imwrite(save_pro_name, image_skeleton_box);
      break;
    }
    case SKELETON_BOX_ID: {
      cv::Mat image_skeleton_box_id = net_wrapper_->get_skeleton(curr_frame,true,true,&meta);
      cv::imwrite(save_pro_name, image_skeleton_box_id);
      break;
    }
    case HEATMAP: {
      cv::Mat image_heatmap = net_wrapper_->get_heatmap(curr_frame,false,false,&meta);
      cv::imwrite(save_pro_name, image_heatmap);
      break;
    }
    case HEATMAP_BOX: {
      cv::Mat image_heatmap_box = net_wrapper_->get_heatmap(curr_frame,true,false,&meta);
      cv::imwrite(save_pro_name, image_heatmap_box);
      break;
    }
    case HEATMAP_BOX_ID: {
      cv::Mat image_heatmap_box_id = net_wrapper_->get_heatmap(curr_frame,true,true,&meta);
      cv::imwrite(save_pro_name, image_heatmap_box_id);
      break;
    }
    case VECMAP: {
      cv::Mat image_vecmap = net_wrapper_->get_vecmap(curr_frame,false,false,&meta);
      cv::imwrite(save_pro_name, image_vecmap);
      break;
    }
    case VECMAP_BOX: {
      cv::Mat image_vecmap_box = net_wrapper_->get_vecmap(curr_frame,true,false,&meta);
      cv::imwrite(save_pro_name, image_vecmap_box);
      break;
    }
    case VECMAP_BOX_ID: {
      cv::Mat image_vecmap_box_id = net_wrapper_->get_vecmap(curr_frame,true,true,&meta);
      cv::imwrite(save_pro_name, image_vecmap_box_id);
      break;
    }
    default: {
      cv::Mat image_default = net_wrapper_->get_skeleton(curr_frame,true,false,&meta);
      cv::imwrite(save_pro_name, image_default);
      break;
    }
  }
  // stat time
  frames_ = net_wrapper_->get_frames();
  if (frames_ % 30 == 0) {
    frame_loader_time_ = frame_loader_time_sum_ / 30;
    frame_loader_time_sum_ = 0;
    display_time_ = display_time_sum_ / 30;
    display_time_sum_ = 0;
  }
  return 0;
}

template <typename Dtype>
void RemoFrontVisualizer<Dtype>::time_us(Dtype* frame_loader, Dtype* preprocess,
                                Dtype* forward, Dtype* drawn, Dtype* display) {
  *frame_loader = frame_loader_time_;
  *preprocess = net_wrapper_->get_preload_us();
  *forward = net_wrapper_->get_forward_us();
  *drawn = net_wrapper_->get_drawn_us();
  *display = display_time_;
}

template <typename Dtype>
void RemoFrontVisualizer<Dtype>::fps(Dtype* net_fps, Dtype* project_fps) {
  Dtype frame_loader, preprocess, forward, drawn, display;
  time_us(&frame_loader, &preprocess, &forward, &drawn, &display);
  *net_fps = (Dtype)1e6 / (forward + 1.0);
  *project_fps = (Dtype)1e6 / (frame_loader+preprocess+forward+drawn+display+1.0);
}

INSTANTIATE_CLASS(RemoFrontVisualizer);

}
