#ifndef CAFFE_EASY_MATCH_HPP
#define CAFFE_EASY_MATCH_HPP

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <vector>
#include <map>

/**
 * 该层主要为帧间对象的匹配提供了基本方法。
 * 该层主要用于解决一个问题：
 * 如何为当前帧中的proposal找到其在上一帧中的对象是哪一个？
 * 即为目标检测结果提供id。
 * 由于该层没有集成目标跟踪，所以称之为easy_match.
 * 实现的主要原理：
 * １．将目标对象池(target_pool，targets)与当前帧的提议池(pro_pool，proposals)进行匹配，匹配系数为两者之间的IOU；
 * ２．将IOU按照大小高低进行排序，然后完成匹配，注意匹配存在阈值；
 * ３．对于未匹配的目标对象(target)，存在两种结果：
 *    （１）该目标对象未检测出来，需要使用Tracker跟踪，但由于没有Tracker，所以直接进行回收（在一定时间段内，如果该对象一直没有找到匹配的proposal，则将其删除）
 *    （２）该目标对象已经消失了（例如，边缘），此时直接将其回收。
 * ４．对于未匹配的提议对象(proposal)，处理方式：
 *    （１）由某个目标对象分裂得到，这里我们没有处理此类情形，直接分配一个新的ID，认为是一个新的对象
 *    （２）是一个新的对象，分配一个新的ID，确认为一个新的对象
 * 当前源码的实现有误，并不是按照这一思路来设计的，因此需要rewritten.
 */

namespace caffe {

template <typename Dtype>
class EasymatchLayer : public Layer<Dtype> {
 public:
  explicit EasymatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Easymatch"; }

  /**
   * bottom[0]: proposals (1,1,N,61)
   * <0-3>: box
   * <4-57>: kps
   * <58>: num_kps
   * <59>: score
   * <60>: id = -1
   * bottom[1]: heatmaps  (1,52,RH,RW)
   */
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * top[0]: proposals (1,1,N,61)
   * the same as bottom[0], but the id is updated.
   */
  virtual inline int ExactNumTopBlobs() const { return 1; }

  /**
   * 关节点
   */
  struct point_t {
    Dtype x;
    Dtype y;
    Dtype v;
  };

  /**
   * person实例的数据结构
   */
  struct person_t {
    // bbox
    Dtype xmin;
    Dtype ymin;
    Dtype xmax;
    Dtype ymax;
    // center location
    Dtype center_x;
    Dtype center_y;
    // size of bbox
    Dtype width;
    Dtype height;
    // kps
    vector<point_t> kps;
    // score
    Dtype score;
    // id and num_kps
    int id;
    int num_points;
    // if is active
    // TRUE: active
    // FALSE: 被回收
    bool active;
    // 回收计时器，超过阈值则彻底在目标池中删除
    int loss_cnt;
  };

  /**
   * 匹配结果
   */
  struct matchsts_t {
    // iou of boxes: 0-1
    Dtype iou;
    // iou(oks) of kps: 0-1
    Dtype oks;
    // similarity of proposals: 0-1
    Dtype similarity;
  };

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  /**
   * 获取两个对象之间的相似度
   * @param person_pro [某个提议对象]
   * @param person_his [某个目标对象]
   * @param sts        [相似度数据结构]
   */
  void get_similarity(person_t& person_pro, person_t& person_his, matchsts_t& sts);

  /**
   * 使用GPU计算两个对象的相似度
   * @param person_pro [某个提议对象]
   * @param person_his [某个目标对象]
   * @param sts        [相似度数据结构]
   */
  void get_similarity_gpu(person_t& person_pro, person_t& person_his, matchsts_t& sts);

  /**
   * 使用提议对象更新目标对象
   * @param person_pro [提议对象]
   * @param person_his [目标对象]
   */
  void update_person(person_t& person_pro, person_t& person_his);

  /**
   * 判断某个目标对象是不是边缘？
   * 该方法不建议使用，不稳定。
   * @param  person [目标对象]
   * @param  gap    [边缘间隙]
   * @return        [是或不是]
   */
  bool at_edge(person_t& person, float gap);

  /**
   * 生成一个新的对象ID
   * 遍历所有目标对象，查找一个最小的不同于所有目标对象的ID
   * @return [返回的ID]
   */
  int get_active_id();

  /**
   * 不建议使用
   * NOTE: should be rewritten
   */
  void get_matched_cols(map<int, map<int, int > >& match_status, int row, vector<int>* indices);

  /**
   * 不建议使用。
   * NOTE: should be rewritten
   */
  void get_matched_rows(map<int, map<int, int > >& match_status, int col, vector<int>* indices);

  /**
   * 当前有效的目标对象池
   */
  vector<person_t> cur_persons_;

  // 当前的待回收对象池
  vector<person_t> temp_persons_;
  // oks计算的variance，不建议使用
  vector<float> vars_;

  // IOU匹配阈值
  Dtype match_iou_thre_;
  // 边缘间隙阈值
  Dtype edge_gap_;
  // 18
  int num_parts_;
  // 17
  int num_limbs_;
};

}  // namespace caffe

#endif  // CAFFE_EASY_MATCH_HPP_
