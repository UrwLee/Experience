#ifndef CAFFE_CONNECT_LIMB_LAYER_HPP_
#define CAFFE_CONNECT_LIMB_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <vector>
#include <map>

namespace caffe {

/**
 * 该层基于估计的Heatmaps/Vecmaps完成Person的拼接。
 * 该层完成的工作：
 * １．穷举所有可能的线段节点对，根据Vecmap计算其置信度，如果有效，则拼接，否则，断开
 * ２．将所有有效的线段对拼接成Person
 * ３．获得所有person的提议(proposals)
 * 如何计算两个点之间是否存在有效的线段呢？
 * １．将两个点之间的线段十等分，这样得到累计10个点的值；
 * ２．检查这10个点的置信度有多少个超过阈值；
 * ３．如果超过阈值的点数达到设定的下限(例如8)，则认为这两个是可以被连起来的。　
 * 如何将不同的线段拼接起来，构成一个人呢？
 * １．我们将人分割为17条线段，编号为0-16，每条线段都有两个点（第一个点为起点，第二个点为终点），且后面的线段的起点都在前面线段中出现过；
 * ２．如果某条线段存在（存在有效线段），且它的起点与某个proposal的前面出现过的点是相同的，那么将这个线段拼接到这个proposal上去。
 * ３．我们遍历的时候，每次将所有的同类线段全部拼接完毕。
 * ４．最终所有线段全部拼接完毕，proposal也全部被提出来了。
 * 如何滤除不好的proposal呢？
 * １．如果某个proposal的可见关节点数太少，低于阈值，则删除之；
 * 关节点数太多，如何处理？
 * １．max_points_可能数量特别大，在耗尽匹配时，计算量太大，耗时太长，如何解决？
 * ２．将每个关节点的数量按照其置信度进行筛选，只保留置信度最高的若干个点，其余的全部删除。
 */

template <typename Dtype>
class ConnectlimbLayer : public Layer<Dtype> {
 public:
  explicit ConnectlimbLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Connectlimb"; }

  /**
   * bottom[0] -> Heatmaps [N,52,H,W]
   * bottom[1] -> Peaks [1,18,num_peaks+1, 3]
   * 52: 0-17 -> Heatmaps and 18-51: Vecmaps
   * num_peaks: -> 获取Heatmaps的极值点时，每个类别关节点的最大极值点数
   * 多出来的+1: -> 这个值是实际检测到的peaks数量
   * 3: -> (x,y,v)，分表表示归一化坐标位置和置信度信息
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0]: proposals [1,N,18+1,3]
   * N: 目标数
   * 18: 关节点数
   * 1: 该目标对象的统计值，包括置信度，平均置信度，可见点数等等
   * 3: (x,y,v)，归一化坐标值和置信度
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

  // 关节点编号
  map<int, string> get_coco_part_name() {
    map<int, string> coco_part_name;
    coco_part_name[0] = "Nose";
    coco_part_name[1] = "Neck";
    coco_part_name[2] = "RShoulder";
    coco_part_name[3] = "RElbow";
    coco_part_name[4] = "RWrist";
    coco_part_name[5] = "LShoulder";
    coco_part_name[6] = "LElbow";
    coco_part_name[7] = "LWrist";
    coco_part_name[8] = "RHip";
    coco_part_name[9] = "RKnee";
    coco_part_name[10] = "RAnkle";
    coco_part_name[11] = "LHip";
    coco_part_name[12] = "LKnee";
    coco_part_name[13] = "LAnkle";
    coco_part_name[14] = "REye";
    coco_part_name[15] = "LEye";
    coco_part_name[16] = "REar";
    coco_part_name[17] = "LEar";
    return coco_part_name;
  }
  int get_coco_num_parts() {
    return 18;
  }
  int get_coco_num_limbs() {
    return 17;
  }

  /**
   * 17条线段的定义：
   * 每条线段定义了2个点：
   * A,B: -> 从A指向B
   */
  //                             2,3, 2,6, 3,4, 4,5, 6,7, 7,8, 2,9, 9,10,10,11,2,12, 12,13, 13,14, 2,1, 1,15, 15,17, 1,16, 16,18
  const int LIMB_SEQ_COCO[34] = {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17};
  vector<int> get_coco_limb_seq() {
    vector<int> coco_limb_seq;
    coco_limb_seq.clear();
    for (int i = 0; i < 34; ++i) {
      coco_limb_seq.push_back(LIMB_SEQ_COCO[i]);
    }
    return coco_limb_seq;
  }

  /**
   * 17条线段在heatmaps中的通道号：
   * 每条线段有x和y两个分量通道。
   * 注意：LIMB_CHANNEL_ID_COCO与LIMB_SEQ_COCO中的线段定义是一一对应的。
   * 例如：从point(1) -> point(2)的线段，其x分量的通道号是30，其y分量的通道号是31
   * 以此类推，得到所有17条线段的通道编号。
   */
  const int LIMB_CHANNEL_ID_COCO[34] = {30,31, 36,37, 32,33, 34,35, 38,39, 40,41, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 42,43, 44,45, 48,49, 46,47, 50,51};
  vector<int> get_coco_limb_channel_id() {
    vector<int> coco_limb_channel_id;
    coco_limb_channel_id.clear();
    for (int i = 0; i < 34; ++i) {
      coco_limb_channel_id.push_back(LIMB_CHANNEL_ID_COCO[i]);
    }
    return coco_limb_channel_id;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  // COCO
  bool type_coco_;
  // 关节点名称编号
  map<int, string> part_name_;
  // limb seqs
  vector<int> limb_seq_;
  // channels id for limb (x & y direction)
  vector<int> limb_channel_id_;
  // num of limbs
  int num_limbs_;
  // num of parts
  int num_parts_;
  // num of max person
  int max_persons_;
  // num of max peaks
  int max_peaks_;

  // PA
  // limb的PA计算等分次数:默认为10次
  int iters_pa_cal_;
  // PA的计算阈值
  Dtype connect_inter_threshold_;
  // 等分计算次数中有效的计算次数:最小值
  int connect_inter_min_nums_;
  // person有效需要的最小关键点数
  int connect_min_subset_cnt_;
  // 关键点平均阈值
  Dtype connect_min_subset_score_;

  // max_peaks
  int max_peaks_use_;
};

}

#endif
