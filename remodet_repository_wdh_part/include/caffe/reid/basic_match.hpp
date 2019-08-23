#ifndef CAFFE_REID_BASIC_MATCH_HPP_
#define CAFFE_REID_BASIC_MATCH_HPP_

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/tracker/bounding_box.hpp"
#include "caffe/tracker/basic.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * 该文件提供了用于分配ID的所有方法
 * 包括：
 * １．定义对象的数据结构
 * ２．定义处理融合／分裂等过程的方法
 * 注意：该系列方法仅作为参考，不建议直接使用。
 * 请在阅读掌握源码基础上使用上述方法，请做适当修改后使用。
 * 这些代码仅作为辅助参考用，禁止直接使用。
 */

using namespace std;
// 关节点信息
template <typename Dtype>
struct Joint {
  Dtype x;
  Dtype y;
  Dtype v;
};

/**
 * 提议对象：当前帧检测到的所有提议对象
 */
template <typename Dtype>
struct pProp {
  // bbox位置
  BoundingBox<Dtype> bbox;
  // 关节点信息kps
  vector<Joint<Dtype> > kps;
  // 可见点数num_vis
  int num_vis;
  // 置信度评分score
  Dtype score;
  // 特征模板指针：默认不包含模板，当匹配成功或加入到目标池后，其模板会自动生成
  Dtype* fptr = NULL;

  // 是否已匹配，默认没有进行匹配
  bool is_matched = false;
  // 是否是分裂对象，默认不是分裂对象
  // 分裂对象：当多个提议对象与同一个目标对象匹配时，即存在分裂行为，此时提议对象的分裂标记有效
  bool is_split = false;
};

/**
 * 单个对象的数据结构，包含：
 * １．ID
 * ２．特征模板
 * ３．相似度：与前景对象之间的相似度
 */
template <typename Dtype>
struct pPerson {
  // ID
  int id;
  // 特征模板
  shared_ptr<Blob<Dtype> > pT;
  // 相似度
  Dtype simi;
};

/**
 * 目标对象的数据结构：可视化中的目标对象
 * 注意：该目标对象可能保存多个相互遮挡的对象
 */
template <typename Dtype>
struct pTarget {
  // 位置
  BoundingBox<Dtype> bbox;
  // kps（前景）
  vector<Joint<Dtype> > kps;
  // 可见数量（关节点可见数量）
  int num_vis;
  // 得分
  Dtype score;

  // 前景特征模板
  shared_ptr<Blob<Dtype> > pT;

  // 前景目标
  pPerson<Dtype> front;
  // 遮挡目标列表：可能包含多个遮挡对象
  vector<pPerson<Dtype> > back;

  // 回收标志: 新创建目标默认不回收
  // 回收：当目标对象没有匹配，开始准备回收
  bool is_trash = false;
  // 回收时间戳，默认为０
  int miss_frame = 0;

  //　默认值：匹配对象为０
  int matched_id = 0;
  // 匹配相似度，默认为０
  Dtype matched_similarity = 0.;
  // 合并标记，默认是不可合并
  // 合并：当多个目标对象与同一个提议对象匹配，则构成合并
  bool is_merged = false;
  // 可分裂标记，默认不可分裂
  // 分裂：当多个提议对象与同一个目标对象匹配，则构成分裂
  bool can_split = false;
  // 支持分裂数，默认为０
  // 分裂数：内部包含的遮挡对象数
  int num_split = 0;

  // 默认目标没有遮挡
  // 遮挡：当该对象与其他对象之间的最大IOU超过阈值，则认为存在遮挡
  bool is_occluded = false;
};

// APIs
/**
 * 定义了一系列方法来处理遮挡问题
 */

/**
 * 初始化目标对象
 * @param targets [目标对象集合]
 */
template <typename Dtype>
void initTargets(vector<pTarget<Dtype> >& targets);

/**
 * 匹配过程
 * @param targets   [目标对象集合]
 * @param proposals [提议对象集合]
 * @param iou_thre  [匹配IOU阈值]
 */
template <typename Dtype>
void pMatch(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals, const Dtype iou_thre);

/**
 * 融合操作
 * @param targets [目标对象集合]
 */
template <typename Dtype>
void pMerge(vector<pTarget<Dtype> >& targets);

/**
 * 目标对象更新过程
 * @param targets   [目标对象集合]
 * @param proposals [提议对象集合]
 */
template <typename Dtype>
void updateTargets(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals);

/**
 * 处理未匹配的目标对象
 * @param targets [目标对象集合]
 */
template <typename Dtype>
void processUnmatchedTargets(vector<pTarget<Dtype> >& targets);

/**
 * 根据特征模板计算两个对象之间的相似度
 * @param f_src [对象１]
 * @param f_dst [对象２]
 * @param thre  [有效阈值]
 * @param s     [相似度]
 */
template <typename Dtype>
void fsimilarity(const Blob<Dtype>& f_src, const Blob<Dtype>& f_dst, const Dtype thre, Dtype* s);

/**
 * 更新目标对象中的前景对象
 * @param targets [目标对象集合]
 * @param thre    [阈值]
 */
template <typename Dtype>
void foreTargetsUpdate(vector<pTarget<Dtype> >& targets, const Dtype thre);

/**
 * 更新目标对象的遮挡状态
 * @param targets   [目标对象集合]
 * @param proposals [提议对象集合]
 * @param thre      [阈值]
 */
template <typename Dtype>
void updateOccludedStatus(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals, const Dtype thre);

/**
 * 模板更新
 * @param targets    [目标对象集合]
 * @param scale_str  [结构化颜色模板更新系数]
 * @param scale_area [区域颜色模板更新系数]
 */
template <typename Dtype>
void updateTemplates(vector<pTarget<Dtype> >& targets, const Dtype scale_str, const Dtype scale_area);

/**
 * 处理为匹配的提议对象
 * @param targets           [目标对象集合]
 * @param proposals         [提议对象集合]
 * @param iou_thre          [IOU阈值-1]
 * @param simi_thre         [相似度阈值-2]
 * @param thre_for_simi_cal [相似度计算阈值-3]
 */
template <typename Dtype>
void processUnmatchedProposals(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals,
                               const Dtype iou_thre, const Dtype simi_thre, const Dtype thre_for_simi_cal);
/**
 * 生成新的目标对象
 * @param prop       [提议对象]
 * @param new_target [新的目标对象]
 */
template <typename Dtype>
void getNewTarget(pProp<Dtype>& prop, pTarget<Dtype>* new_target);

/**
 * 获取新的对象ID
 * @param  targets [目标对象集合]
 * @return         [新的ID]
 */
template <typename Dtype>
int getNewId(vector<pTarget<Dtype> >& targets);

/**
 * 打印目标对象信息
 * @param targets [目标对象集合]
 */
template <typename Dtype>
void printInfo(vector<pTarget<Dtype> >& targets);

}  // namespace caffe

#endif
