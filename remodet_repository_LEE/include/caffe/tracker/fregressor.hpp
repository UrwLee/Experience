#ifndef CAFFE_TRACKER_FREGRESSOR_H
#define CAFFE_TRACKER_FREGRESSOR_H

#include "caffe/caffe.hpp"
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/fregressor_base.hpp"

namespace caffe {

/**
 * F回归器实现。
 */

template <typename Dtype>
class FRegressor : public FRegressorBase<Dtype> {
 public:

  /**
   * 构造：
   * gpu_id：　计算GPU ID
   * tracker_network_proto： 网络描述文件
   * tracker_caffe_model：　网络权值文件
   * res_features：　回归器结果的Blob名
   * num_inputs：　输入Blobs数量，　默认为１
   */
  FRegressor(const int gpu_id,
             const std::string& tracker_network_proto,
             const std::string& tracker_caffe_model,
             const std::string& res_features,
             const int num_inputs);

  /**
   * 默认构造，num_inputs为1
   */
  FRegressor(const int gpu_id,
             const std::string& tracker_network_proto,
             const std::string& tracker_caffe_model,
             const std::string& res_features);

  /**
   * 在线网络构造回归器
   */
  FRegressor(const int gpu_id,
             const boost::shared_ptr<caffe::Net<Dtype> >& net,
             const std::string& res_features);

  /**
   * 回归方法
   * @param curr [当前特征Blob]
   * @param prev [历史特征Blob]
   * @param bbox [回归结果]
   */
  virtual void Regress(const Blob<Dtype>& curr, const Blob<Dtype>& prev, BoundingBox<Dtype>* bbox);
  virtual void Regress(const std::vector<boost::shared_ptr<Blob<Dtype> > >& curr,
                       const std::vector<boost::shared_ptr<Blob<Dtype> > >& prev,
                       std::vector<BoundingBox<Dtype> >* bboxes);

protected:
  /**
   * 获取回归器结果
   * @param output [输出结果]
   */
  void GetOutput(std::vector<Dtype>* output);
  /**
   * 多样本输入：Reshape
   * @param num_inputs [样本对数]
   */
  void ReshapeNumInputs(const int num_inputs);
  /**
   * 获取指定Blob的输出结果
   * @param feature_name [特征名]
   * @param output       [输出结果]
   */
  void GetFeatures(const std::string& feature_name, std::vector<Dtype>* output);

  /**
   * 估计方法
   * @param curr   [当前特征Blob]
   * @param prev   [历史特征Blob]
   * @param output [输出结果]
   */
  void Estimate(const Blob<Dtype>& curr, const Blob<Dtype>& prev, std::vector<Dtype>* output);
  void Estimate(const std::vector<boost::shared_ptr<Blob<Dtype> > >& curr,
                const std::vector<boost::shared_ptr<Blob<Dtype> > >& prev,
                std::vector<Dtype>* output);

  /**
   * 初始化
   */
  virtual void Init() {}

 private:

  /**
   * 构造网络
   * @param network_proto [网络描述文件]
   * @param caffe_model   [网络权值文件]
   * @param gpu_id        [GPU ID]
   */
  void SetupNetwork(const std::string& network_proto,
                    const std::string& caffe_model,
                    const int gpu_id);
 private:
  //  样本对输入
  int num_inputs_;
  // 输入尺寸
  int input_width_;
  int input_height_;
  // 输入通道数
  int input_channels_;
  // 特征名
  std::string features_;
};

}

#endif
