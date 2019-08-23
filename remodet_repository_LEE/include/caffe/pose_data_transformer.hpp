#ifndef CAFFE_POSE_DATA_TRANSFORMER_HPP
#define CAFFE_POSE_DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracker/bounding_box.hpp"

namespace caffe {

/**
 * 该层为Pose估计的任务提供了图片预处理的所有方法。
 * 包括：
 * １．随机Scale
 * ２．随机裁剪
 * ３．随机旋转
 * ４．随机flip
 * ５．随机颜色失真...
 * 该类提供了对数据进行封装的方法。
 * 该类被集成进pose_data_layer中，负责为该数据输入层提供数据处理能力。
 */

template <typename Dtype>
class PoseDataTransformer {
 public:
   /**
    * 构造方法
    * @param  param [参数,proto定义]
    * @param  phase [TRAIN or TEST]
    * @return       []
    */
  explicit PoseDataTransformer(const PoseDataTransformationParameter& param, Phase phase);
  virtual ~PoseDataTransformer() {}

  // 初始化随机过程
  void InitRand();

  /**
   * 图像转换为Blob [rsvd.]
   * @param cv_img           [图像]
   * @param transformed_blob [输出数据data]
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  /**
   * rsvd.
   * @param mat_vector       [多副图像]
   * @param transformed_blob [输出数据data]
   */
  void Transform(const vector<cv::Mat> & mat_vector, Blob<Dtype>* transformed_blob);

  /**
   * 转换过程。
   * @param xml_file              [XML文件]
   * @param transformed_data      [data数据]
   * @param transformed_vec_mask  [vec-mask数据]
   * @param transformed_heat_mask [heat-mask数据]
   * @param transformed_vecmap    [vec-map数据]
   * @param transformed_heatmap   [heat-map数据]
   */
  void Transform_nv(const string& xml_file,
                    Dtype* transformed_data,
                    Dtype* transformed_vec_mask,
                    Dtype* transformed_heat_mask,
                    Dtype* transformed_vecmap,
                    Dtype* transformed_heatmap);

  /**
   * 转换过程: 多了一个keypoints的真实值
   * 该方法比较少用，仅在测试过程中有所使用。
   * @param transformed_kps       [description]
   */
  void Transform_nv_out(const string& xml_file,
                        Dtype* transformed_data,
                        Dtype* transformed_vec_mask,
                        Dtype* transformed_heat_mask,
                        Dtype* transformed_vecmap,
                        Dtype* transformed_heatmap,
                        Dtype* transformed_kps);

  // rsvd.
  vector<int> InferBlobShape(const cv::Mat& cv_img);

  /**
   * 增广参数
   */
  struct AugmentSelection {
    // flip: TRUE or FALSE
    bool flip;
    // 旋转角度
    float degree;
    // 裁剪的起点位置
    Size crop;
    // SCALE参数
    float scale;
  };

  /**
   * 关节点描述
   */
  struct Joints {
    vector<Point2f> joints;
    vector<int> isVisible;
  };

  // 标注数据结构
  struct MetaData {
    // 图像路径
    string img_path;
    // Mask_ALL 路径
    string mask_all_path;
    // Mask_Miss路径，该Mask图片需要使用
    string mask_miss_path;
    // 数据集类型
    string dataset;
    // 图片尺寸
    Size img_size;
    // rsvd.
    bool isValidation;
    // 除去主要对象外，该图片中剩余的对象数
    int numOtherPeople;
    // 该主要对象的index
    int people_index;
    // 标注文件的index
    int annolist_index;
    // 该对象的中心位置
    Point2f objpos;
    // 该对象的位置
    BoundingBox<Dtype> bbox;
    // 该对象的scale: box_height / 368.
    float scale_self;
    // 该对象的面积,unused.
    float area;
    // 该对象的关节点信息
    Joints joint_self;

    // 其余对象的中心位置
    vector<Point2f> objpos_other;
    // 其余对象的scale
    vector<float> scale_other;
    // 其余对象的area
    vector<float> area_other;
    // 其余对象的关节点信息
    vector<Joints> joint_others;
  };

  /**
   * 生成HeatMap & VecMap的Label
   * @param transformed_vecmap  [vecmap]
   * @param transformed_heatmap [heatmap]
   * @param img_aug             [图片]
   * @param meta                [标注信息]
   */
  void generateLabelMap(Dtype* transformed_vecmap, Dtype* transformed_heatmap, cv::Mat& img_aug, MetaData& meta);

  /**
   * 可视化：保存增广后的图片及标注
   * @param img  [图像]
   * @param meta [标注信息]
   */
  void visualize(Mat& img, MetaData& meta);

  /**
   * 下述方法中不包括对Mask_Miss的处理
   * 该类方法在不适用Mask的情形下使用。
   */
  // flip
  bool augmentation_flip(Mat& img, Mat& img_aug, MetaData& meta);
  // rotate
  float augmentation_rotate(Mat& img, Mat& img_aug, MetaData& meta);
  // scale
  float augmentation_scale(Mat& img, Mat& img_aug, MetaData& meta);
  // crop and pad
  Size augmentation_croppad(Mat& img, Mat& img_aug, MetaData& meta);

  /**
   * 下述方法中包括对Mask_Miss的处理
   * 该类方法包括Mask的情形下使用。
   * 我们目前使用这一类方法。
   */
  // flip
  bool augmentation_flip(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);
  // rotate
  float augmentation_rotate(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);
  // scale
  float augmentation_scale(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);
  // crop and pad
  Size augmentation_croppad(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta, int mode);

  /**
   * 旋转关节点
   * @param p [坐标点]
   * @param R [cv旋转变换矩阵]
   */
  void RotatePoint(Point2f& p, Mat& R);

  /**
   * 判断一个坐标点是否在某个map上
   * @param  p        [坐标点]
   * @param  img_size [map尺寸]
   * @return          [是或不是]
   */
  bool onPlane(Point p, Size img_size);

  /**
   * 对关节点进行左右变换
   * 含义：将一个L# -> R#
   * 含义：将一个R# -> L#
   * @param j [description]
   */
  void swapLeftRight(Joints& j);

  // 17
  int np_;
  bool bg_crop(MetaData& meta);

 protected:
  // 生成一个0~n之间的随机整数
  virtual int Rand(int n);

  // 图形数据转换
  void Transform(const cv::Mat& cv_img, Dtype* transformed_data);
  // 将keypoints坐标输出到一个Blobs数据之中
  void Output_keypoints(MetaData& meta, Dtype* out_kp);
  /**
   * 数据转换过程
   * @param cv_img                [原始图片]
   * @param mask_miss             [cv::Mat mask_miss]
   * @param mask_all              [cv::Mat mask_all]
   * @param meta                  [标注元数据]
   * @param transformed_data      [data]
   * @param transformed_vec_mask  [vec_mask]
   * @param transformed_heat_mask [heat_mask]
   * @param transformed_vecmap    [vec_map]
   * @param transformed_heatmap   [heat_map]
   */
  void Transform_nv(cv::Mat& cv_img,
                    cv::Mat& mask_miss,
                    cv::Mat& mask_all,
                    MetaData& meta,
                    Dtype* transformed_data,
                    Dtype* transformed_vec_mask,
                    Dtype* transformed_heat_mask,
                    Dtype* transformed_vecmap,
                    Dtype* transformed_heatmap);

  /**
   * 从XML中读取标注信息
   * @param  xml_file [XML文件]
   * @param  root_dir [根目录]
   * @param  meta     [标注数据]
   * @return          [读取成功或失败]
   */
  bool ReadMetaDataFromXml(const string& xml_file, const string& root_dir, MetaData& meta);

  /**
   * 将原始标注信息转换为统一的18点；
   * 顺序按照指定的格式进行。
   * @param meta [标注数据]
   */
  void TransformMetaJoints(MetaData& meta);

  /**
   * 对单个目标对象的关节点进行转换
   */
  void TransformJoints(Joints& joints);

  /**
   * 基于某个点，生成一个GaussMap (Heatmap)
   * 最终生成的HeatMap为所有同类型点的叠加。
   * @param map    [map数据]
   * @param center [关节点的位置]
   * @param stride [map相比于原始图像stride]
   * @param grid_x [网格尺寸]
   * @param grid_y [网格尺寸]
   * @param sigma  [gauss分布的sigma系数]
   */
  void putGaussianMaps(Dtype* map, Point2f center, int stride, int grid_x, int grid_y, float sigma);

  /**
   * 基于两个点生成这两个点之间的vecmap
   * 最终生成的VecMap为所有同类型线段（由同类的两个点构成）的叠加
   * 该条线段的两个端点：A/B，方向为：A->B
   * @param vecX    [该条线段X分量Map]
   * @param vecY    [该条线段Y分量Map]
   * @param count   [不同对象共同生成该Map时的累加因子，unused，可以认为一直是0]
   * @param centerA [A点位置]
   * @param centerB [B点位置]
   * @param stride  [map相比于原始图像stride]
   * @param grid_x  [网格尺寸]
   * @param grid_y  [网格尺寸]
   * @param sigma   [线段在方向矢量的法线上的投影范围]
   * @param thre    [阈值]
   */
  void putVecMaps(Dtype* vecX, Dtype* vecY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);

  /**
   * 传统方法：在连段之间放置一些中间点位进行辅助判定。
   * 目前未使用该类方法，可以跳过。
   * rsvd.
   */
  void putVecPeaks(Dtype* vecX, Dtype* vecY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);

  // 转换参数
  PoseDataTransformationParameter param_;

  // 随机数
  shared_ptr<Caffe::RNG> rng_;
  // TRAIN or TEST
  Phase phase_;
  // 平均值Blob, unused.
  Blob<Dtype> data_mean_;

  // 平均值，[104,117,123]
  vector<Dtype> mean_values_;
};

}

#endif
