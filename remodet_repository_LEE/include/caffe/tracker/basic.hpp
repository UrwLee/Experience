#ifndef CAFFE_TRACKER_BASIC_H_
#define CAFFE_TRACKER_BASIC_H_

#include <string>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "glog/logging.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * 一些基础方法，包括目录下的同扩展名遍历索引
 * 以及一些字符串和随机数发生器函数
 */

// *******Number / string conversions*************
// Conversions from number into a string.
std::string num2str(const int num);
std::string num2str(const float num);
std::string num2str(const double num);
std::string num2str(const double num, const int decimal_places);
std::string num2str(const unsigned int num);
std::string num2str(const size_t num);

// Conversions from string into a number.
template<typename Dtype>
Dtype str2num(const std::string& s);

// Template implementation
template<typename Dtype>
Dtype str2num(const std::string& s)
{
   std::istringstream stream (s);
   Dtype t;
   stream >> t;
   return t;
}

// *******File IO *************
/**
 * 查找子目录
 * @param folder      [源目录]
 * @param sub_folders [查找的子目录集合]
 */
void find_subfolders(const boost::filesystem::path& folder, std::vector<std::string>* sub_folders);

/**
 * 查找匹配文件集合
 * @param folder [目录]
 * @param filter [匹配滤波器]
 * @param files  [匹配的文件集合]
 */
void find_matching_files(const boost::filesystem::path& folder, const boost::regex filter,
                         std::vector<std::string>* files);

// *******Probability*************
/**
 * 平均标准化采样
 * @return [返回一个0-1之间的随机数]
 */
float sample_rand_uniform();


/**
 * 随机采样一个指数分布
 * @param  lambda [指数系数]
 * @return        [返回的随机值]
 */
template<typename Dtype>
Dtype sample_exp(const Dtype lambda);


/**
 * 随机采样一个双边指数分布
 * @param  lambda [指数参数]
 * @return        [返回的随机值]
 */
template<typename Dtype>
Dtype sample_exp_two_sided(const Dtype lambda);

}

#endif
