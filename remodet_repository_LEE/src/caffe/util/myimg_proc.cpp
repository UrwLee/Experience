#include <algorithm>
#include <csignal>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "caffe/util/myimg_proc.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
//--------------------------------------------------------------------
  void get_grayworld_gains(cv::Mat &src, float *kb, float *kg, float *kr){
    vector<cv::Mat> g_vChannels;
    cv::split(src,g_vChannels);
    cv::Mat imageBlueChannel = g_vChannels.at(0);
    cv::Mat imageGreenChannel = g_vChannels.at(1);
    cv::Mat imageRedChannel = g_vChannels.at(2);

    float imageBlueChannelAvg = cv::mean(imageBlueChannel)[0];
    float imageGreenChannelAvg = cv::mean(imageGreenChannel)[0];
    float imageRedChannelAvg = cv::mean(imageRedChannel)[0];

    float k = (imageBlueChannelAvg+imageGreenChannelAvg+imageRedChannelAvg)/3.0;
    *kb = k / imageBlueChannelAvg;
    *kg = k / imageGreenChannelAvg;
    *kr = k / imageRedChannelAvg;
  }
//--------------------------------------------------------------------
  void grayworld_awb(cv::Mat &src, cv::Mat &dst, float kb, float kg, float kr){
    vector<cv::Mat> g_vChannels;
    cv::split(src,g_vChannels);
    cv::Mat imageBlueChannel = g_vChannels.at(0);
    cv::Mat imageGreenChannel = g_vChannels.at(1);
    cv::Mat imageRedChannel = g_vChannels.at(2);

    cv::addWeighted(imageBlueChannel,kb,0,0,0,imageBlueChannel);
    cv::addWeighted(imageGreenChannel,kg,0,0,0,imageGreenChannel);
    cv::addWeighted(imageRedChannel,kr,0,0,0,imageRedChannel);

    cv::merge(g_vChannels,dst);
  }
//--------------------------------------------------------------------
  void grayworld_awb_single(cv::Mat &src, cv::Mat &dst){
    vector<cv::Mat> g_vChannels;
    cv::split(src,g_vChannels);
    cv::Mat imageBlueChannel = g_vChannels.at(0);
    cv::Mat imageGreenChannel = g_vChannels.at(1);
    cv::Mat imageRedChannel = g_vChannels.at(2);

    float imageBlueChannelAvg = cv::mean(imageBlueChannel)[0];
    float imageGreenChannelAvg = cv::mean(imageGreenChannel)[0];
    float imageRedChannelAvg = cv::mean(imageRedChannel)[0];

    float k = (imageBlueChannelAvg+imageGreenChannelAvg+imageRedChannelAvg)/3.0;
    float kb = k / imageBlueChannelAvg;
    float kg = k / imageGreenChannelAvg;
    float kr = k / imageRedChannelAvg;

    cv::addWeighted(imageBlueChannel,kb,0,0,0,imageBlueChannel);
    cv::addWeighted(imageGreenChannel,kg,0,0,0,imageGreenChannel);
    cv::addWeighted(imageRedChannel,kr,0,0,0,imageRedChannel);

    cv::merge(g_vChannels,dst);
  }
//--------------------------------------------------------------------
  // 求取白点均值，n是数组长度
  // ratio: 前百分之多少，认为是白点.10%
  float whitepoint_ave(cv::Mat frame, int n, float ratio){
    int level_num[256];
    for (int i=0;i<256;i++)
      level_num[i]=0;
    float sum=0;
    // 统计每个亮度的像素点数量
    for (int i=0;i<n;i++)
    {
        int d=frame.at<uchar>(0,i);
        level_num[d]++;
    }
    int n0=255;
    // 统计前ratio的像素点level：n0
    for (int k=255;k>0;k--)
    {
        sum+=level_num[k];
        if (sum>ratio*frame.rows*frame.cols)
          break;
        n0--;
    }
    // 统计算数平均值
    sum=0;
    for (int i=n0;i<256;i++)
      sum+=level_num[i]*i;
    return sum/(ratio*frame.rows*frame.cols);
  }
//--------------------------------------------------------------------
  float whitepoint_ave(cv::Mat frame, float ratio){
    int level_num[256];
    for (int i=0;i<256;i++)
      level_num[i]=0;
    float sum=0;
    // 统计每个像素level的数量
    for (int i=0;i<frame.rows;i++)
    {
        for (int j=0;j<frame.cols;j++)
        {
            int d=(int)frame.at<uchar>(i,j);
            level_num[d]++;
        }
    }
    // 统计前ratio的level
    int n0=255;
    for (int k=255;k>0;k--)
    {
        sum+=level_num[k];
        if (sum>ratio*frame.rows*frame.cols)
        {
            break;
        }
        n0--;
    }
    sum=0;
    for (int i=n0;i<256;i++)
      sum+=level_num[i]*i;
    return sum/(ratio*frame.rows*frame.cols);
  }
//--------------------------------------------------------------------
  void dynamic_awb(cv::Mat &src, cv::Mat &dst, float ratio){
    int width = src.cols;
    int heigth = src.rows;
    int dst_heigth = dst.rows;
    int dst_width = dst.cols;
    int channels = src.channels();
    int dst_channels = dst.channels();

    CHECK_EQ(width, dst_width) << "the src image and dst image should have the same width.";
    CHECK_EQ(heigth, dst_heigth) << "the src image and dst image should have the same heigth.";
    CHECK_EQ(channels, dst_channels) << "the src image and dst image should have the same channels.";

    cv::Mat imageYCrCb = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::cvtColor(src,imageYCrCb,CV_BGR2YCrCb);

    vector<cv::Mat>ycrcb(imageYCrCb.channels());
    cv::split(imageYCrCb,ycrcb);

    float mb,db;
    float mr,dr;

    cv::Mat gavg,gsigma;
    cv::meanStdDev(ycrcb[2],gavg,gsigma);
    mb = gavg.at<float>(0);
    db = gsigma.at<float>(0);
    cv::meanStdDev(ycrcb[1],gavg,gsigma);
    mr = gavg.at<float>(0);
    dr = gsigma.at<float>(0);

    float r,b;
    if(mb < 0)
      b = mb - db;
    else
      b = mb + db;
    if(mr < 0)
      r = 1.5*mr - dr;
    else
      r = 1.5*mr + dr;

    float y_white_ave = whitepoint_ave(ycrcb[0], ratio);
    cv::Mat bwhite = cv::Mat::zeros(1,6000000,CV_8UC1);
    cv::Mat gwhite = cv::Mat::zeros(1,6000000,CV_8UC1);
    cv::Mat rwhite = cv::Mat::zeros(1,6000000,CV_8UC1);

    int count = 0;
    for (int i = 0; i < heigth; i++){
        for (int j = 0; j < width; j++){
            if (((ycrcb[2].at<uchar>(i,j)-b)<(1.5*db))&&((ycrcb[1].at<uchar>(i,j)-r)<(1.5*dr)))
            {
              int d1=src.at<cv::Vec3b>(i,j)[0];
              bwhite.at<uchar>(0,count)=d1;

              int d2=src.at<cv::Vec3b>(i,j)[1];
              gwhite.at<uchar>(0,count)=d2;

              int d3=src.at<cv::Vec3b>(i,j)[2];
              rwhite.at<uchar>(0,count)=d3;
              count++;
            }
        }
     }

     float bave = whitepoint_ave(bwhite, count, ratio);
     float gave = whitepoint_ave(gwhite, count, ratio);
     float rave = whitepoint_ave(rwhite, count, ratio);

     std::cout << "global_white_level is: " << y_white_ave << '\n';
     std::cout << "b_white_level is: " << bave << '\n';
     std::cout << "g_white_level is: " << gave << '\n';
     std::cout << "r_white_level is: " << rave << '\n';

     float bgain=y_white_ave/bave;
     float ggain=y_white_ave/gave;
     float rgain=y_white_ave/rave;

    //  std::cout << "bgain: " << bgain << '\n';
    //  std::cout << "ggain: " << ggain << '\n';
    //  std::cout << "rgain: " << rgain << '\n';

     for (int i = 0; i < heigth; i++){
        for (int j = 0; j < width; j++){
            int tb=bgain*src.at<cv::Vec3b>(i,j)[0];
            int tg=ggain*src.at<cv::Vec3b>(i,j)[1];
            int tr=rgain*src.at<cv::Vec3b>(i,j)[2];
            if (tb>255) tb=255;
            if (tg>255) tg=255;
            if (tr>255) tr=255;
            dst.at<cv::Vec3b>(i,j)[0]=tb;
            dst.at<cv::Vec3b>(i,j)[1]=tg;
            dst.at<cv::Vec3b>(i,j)[2]=tr;
        }
     }
  }
  //--------------------------------------------------------------------
  void sharp_2D(cv::Mat &src, cv::Mat &dst){
    cv::Mat kernela(3, 3, CV_32F, cv::Scalar(0));
    kernela.at<float>(1, 1) = 5.0;
    kernela.at<float>(0, 1) = -1;
    kernela.at<float>(2, 1) = -1;
    kernela.at<float>(1, 0) = -1;
    kernela.at<float>(1, 2) = -1;

    cv::filter2D(src, dst, src.depth(), kernela);
  }

  // -------------------------------------------------------------------
  template <typename Dtype>
  void blobTocvImage(const Dtype *data, const int height, const int width,
                     const int channels, cv::Mat *image) {
     for(int h = 0; h < height; ++h){
       uchar* ptr = (*image).ptr<uchar>(h);
       int img_index = 0;
       for(int w = 0; w < width; ++w){
         for(int c = 0; c < channels; ++c){
           int idx = (c * height + h) * width + w;
           ptr[img_index++] = static_cast<uchar>(data[idx]);
         }
       }
     }
  }
  template void blobTocvImage(const float *data, const int height, const int width,
                     const int channels, cv::Mat *image);
  template void blobTocvImage(const double *data, const int height, const int width,
                    const int channels, cv::Mat *image);
} // namespace caffe
