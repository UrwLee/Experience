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

#include "caffe/util/myimg_proc.hpp"
#include "caffe/mask/visual_mtd_layer.hpp"
namespace caffe{
template <typename Dtype>
void VisualMtdLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype> *> &bottom,const vector<Blob<Dtype> *> &top){
	const DetectionOutputParameter &detection_output_param =
      this->layer_param_.detection_output_param();
    visual_param_ = detection_output_param.visual_param();
    num_classes_ = detection_output_param.num_classes();
	}

template <typename Dtype>
void VisualMtdLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,const vector<Blob<Dtype> *> &top) {
	std::vector<int> top_shape(3,1);
	top_shape.push_back(7);
	top[0]->Reshape(top_shape);
	}

template <typename Dtype>
void VisualMtdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,const std::vector<Blob<Dtype> *> &top) {
	const Dtype *roi = bottom[0]->cpu_data();

	std::vector<cv::Mat> images;
	const int channels = bottom[1]->channels();
	int height = bottom[1]->height();
	int width = bottom[1]->width();
	const int nums = bottom[1]->num();
	const Dtype* img_data = bottom[1]->cpu_data();
	for(int i = 0; i < nums; ++i){
		cv::Mat image(height,width,CV_8UC3,cv::Scalar(0,0,0));
		blobTocvImage(img_data,height,width,channels,&image);
		images.push_back(image);
		img_data += bottom[1]->offset(1);
	}
	static clock_t start_clock = clock();
	static clock_t total_time_start = start_clock;
	static long total_run_time = 0;
	static long total_frame = 0;
	const int num_img = images.size();

	float fps =num_img / (static_cast<double>(clock() - start_clock)/CLOCKS_PER_SEC);
	total_run_time = clock()-total_time_start;
	total_frame += num_img;
	float run_ftime_sec = static_cast<double>(total_run_time) / CLOCKS_PER_SEC;
 	int run_ms = (static_cast<long>(run_ftime_sec * 1000)) % 1000;
 	int run_s  = static_cast<int>(run_ftime_sec);
 	int run_hour = run_s / 3600;
	run_s = run_s % 3600;
	int run_min = run_s / 60;
	run_s = run_s % 60;
	std::cout << std::setiosflags(std::ios::fixed);
	std::cout << "=================================================================" << std::endl;
	std::cout << "Current Frame Number  : " << total_frame << "\n";
	std::cout << "Current Process Speed : " << std::setprecision(1) << fps << " FPS\n";
	if(run_hour>0){
	   std::cout << "Current Time          : " << run_hour << " hour " \
             << run_min << " min " << run_s << " s " << run_ms << " ms\n";
 	}
 	else if(run_min>0){
   	std::cout << "Current Time          : " << run_min << " min " \
             << run_s << " s " << run_ms << " ms\n";
 	}
 	else if(run_s>0){
    std::cout << "Current Time          : " << run_s << " s " << run_ms << " ms\n";
 	}
 	else{
   	std::cout << "Current Time          : " << run_ms << " ms\n";
 	}
 	std::cout << std::endl;
	
	CHECK(visual_param_.has_visualize()) << "visualize must be provided.";
 	CHECK(visual_param_.visualize()) << "visualize must be enabled.";
 	CHECK(visual_param_.has_color_param())  << "color_param must be provided";
 	const ColorParameter &color_param =  visual_param_.color_param();
 	CHECK(visual_param_.has_display_maxsize()) << "display_maxsize must be provided.";
 	const int display_max_size = visual_param_.display_maxsize();
 	CHECK(visual_param_.has_line_width()) << "line_width must be provided.";
 	const int box_line_width = visual_param_.line_width();
 	const float conf_threshold = visual_param_.conf_threshold();
 	const float size_threshold = visual_param_.size_threshold();
 	width = images[0].cols;
 	height = images[0].rows;
 	const int maxLen = (width > height) ? width : height;
 	const float ratio = (float)display_max_size / maxLen;
 	const int display_width = static_cast<int>(width * ratio);
	const int display_height = static_cast<int>(height * ratio);
 	CHECK_EQ(color_param.rgb_size(), num_classes_);
 	for (int i = 0; i < color_param.rgb_size(); ++i) {
  		CHECK_EQ(color_param.rgb(i).val_size(), 3);
 	}

 	cv::Mat display_image;
	cv::resize(images[0], display_image, cv::Size(display_width,display_height), cv::INTER_LINEAR);
	std::cout << "The input image size is   : " << width << "x" << height << '\n';
	std::cout << "The display image size is : " << display_width << "x" << display_height<< '\n';
	std::cout << std::endl;
	int num_det = bottom[0]->height();
	for(int i=0;i<num_det;i++){
		int bindex = roi[i*7+0];
		if(bindex<0) continue;
		int c_id = roi[i*7+1];
		CHECK_GE(c_id,0)<<"c_id"<<c_id;		
		const Color& rgb = color_param.rgb(c_id);
		cv::Scalar line_color(rgb.val(2),rgb.val(1),rgb.val(0));
		float w = roi[i*7+5]-roi[i*7+3];
		float h = roi[i*7+6]-roi[i*7+4];
		if(w*h<size_threshold) continue;
		float score = roi[i*7+2];
		if(score<conf_threshold) continue;
		cv::Point top_left_pt(static_cast<int>(roi[i*7+3] * display_width),
                             static_cast<int>(roi[i*7+4] * display_height));
       	cv::Point bottom_right_pt(static_cast<int>(roi[i*7+5] * display_width),
                             static_cast<int>(roi[i*7+6] * display_height));
       	cv::rectangle(display_image, top_left_pt, bottom_right_pt, line_color, box_line_width);       	
	}
	cv::imshow("remo",display_image);
	if (cv::waitKey(1) == 27) {
     raise(SIGINT);
   }
 	start_clock = clock();

 }
  	#ifdef CPU_ONLY
 	STUB_GPU_FORWARD(VisualMtdLayer,Forward)
 	#endif

 	INSTANTIATE_CLASS(VisualMtdLayer);
 	REGISTER_LAYER_CLASS(VisualMtd);
}