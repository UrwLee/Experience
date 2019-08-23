#include <vector>

#include "caffe/reid/allocate_id_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AllocateIdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  match_iou_thre_ = this->layer_param_.allocate_id_param().match_iou_thre();
  thre_for_cal_similarity_ = this->layer_param_.allocate_id_param().thre_for_cal_similarity();
  occu_coverage_thre_ = this->layer_param_.allocate_id_param().occu_coverage_thre();
  scale_for_update_str_ = this->layer_param_.allocate_id_param().scale_for_update_str();
  scale_for_update_area_ = this->layer_param_.allocate_id_param().scale_for_update_area();
  split_iou_thre_ = this->layer_param_.allocate_id_param().split_iou_thre();
  split_simi_thre_ = this->layer_param_.allocate_id_param().split_simi_thre();
  props_.clear();
  targets_.clear();
}

template <typename Dtype>
void AllocateIdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(3), 61);
  CHECK_EQ(bottom[1]->shape(1), bottom[0]->shape(2));
  CHECK_EQ(bottom[1]->shape(2), 11);
  CHECK_EQ(bottom[1]->shape(3), 512);
  vector<int> shape(4,1);
  shape[2] = bottom[0]->shape(2);
  shape[3] = 63;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void AllocateIdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 提议数
  const int N = bottom[0]->shape(2);
  // 提议数据指针
  const Dtype* proposal = bottom[0]->cpu_data();
  // 特征表示
  Dtype* fmap = bottom[1]->mutable_cpu_data();

  // 11*512
  const int offs_item = bottom[1]->shape(2) * bottom[1]->shape(3);
  // 512
  // const int offs_map = bottom[1]->shape(3);

  // 获取提议池
  props_.clear();
  for (int i = 0; i < N; ++i) {
    // 根据score判断该提议是否有效
    if (proposal[i*61+59] > 0) {
      pProp<Dtype> prop;
      // 获取bbox
      prop.bbox.x1_ = proposal[i*61];
      prop.bbox.y1_ = proposal[i*61+1];
      prop.bbox.x2_ = proposal[i*61+2];
      prop.bbox.y2_ = proposal[i*61+3];
      // 获取kps
      prop.kps.resize(18);
      for (int j = 0; j < 18; ++j) {
        prop.kps[j].x = proposal[i*61+4+3*j];
        prop.kps[j].y = proposal[i*61+4+3*j+1];
        prop.kps[j].v = proposal[i*61+4+3*j+2];
      }
      // num_vis
      prop.num_vis = proposal[i*61+58];
      // score
      prop.score = proposal[i*61+59];
      // fptr
      prop.fptr = fmap + i * offs_item;

      // 送入提议池
      props_.push_back(prop);
    }
  }

  //? 是否需要将６提前到３之后？
  /**
   * 获取当前的提议池后，开始进行如下步骤：
   * １．初始化目标池中的对象，为匹配做准备
   * ２．与提议池进行匹配；
   * ３．合并目标池中的对象；
   * ４．更新目标中的位置参数和特征参数
   * ５．将未匹配目标对象进行回收；
   * ６．处理非匹配的提议对象：　新增或分裂　　【生成新的对象】
   * ７．更新目标对象的前景id
   * ８．更新目标对象的遮挡状态
   * ９．更新目标对象的特征模板
   * １０．将非回收的目标对象输出
   */
  /**
   * STEP1: 初始化目标对象
   */
  initTargets(targets_);
  /**
   * STEP2: 匹配
   */
  pMatch(targets_, props_, match_iou_thre_);
  /**
   * STEP3: 合并
   */
  pMerge(targets_);
  printInfo(targets_);
  /**
   * STEP4: 更新位置和特征
   */
  updateTargets(targets_, props_);
  /**
   * STEP5: 回收未匹配对象
   */
  processUnmatchedTargets(targets_);
  /**
   * STEP6: 处理非匹配的提议对象: 新增或分裂，得到新的对象
   */
  processUnmatchedProposals(targets_, props_, split_iou_thre_, split_simi_thre_, thre_for_cal_similarity_);
  /**
   * STEP7: 更新前景ID
   */
  foreTargetsUpdate(targets_, thre_for_cal_similarity_);
  /**
   * STEP8: 更新目标遮挡状态
   */
  updateOccludedStatus(targets_, props_, occu_coverage_thre_);
  /**
   * STEP9: 更新目标特征模板
   */
  updateTemplates(targets_, scale_for_update_str_, scale_for_update_area_);
  /**
   * STEP10: 输出非回收的目标对象
   */
  // 　统计输出数量
  int num_output = 0;
  for (int i = 0; i < targets_.size(); ++i) {
    if (targets_[i].is_trash || targets_[i].matched_id < 0) continue;
    ++num_output;
  }
  if (num_output == 0) {
    top[0]->Reshape(1,1,1,63);
    caffe_set(top[0]->count(), (Dtype)-1, top[0]->mutable_cpu_data());
  } else {
    top[0]->Reshape(1,1,num_output,63);
    // 输出
    int pidx = 0;
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < targets_.size(); ++i) {
      if (targets_[i].is_trash || targets_[i].matched_id < 0) continue;
      // box
      top_data[pidx*63] = targets_[i].bbox.x1_;
      top_data[pidx*63+1] = targets_[i].bbox.y1_;
      top_data[pidx*63+2] = targets_[i].bbox.x2_;
      top_data[pidx*63+3] = targets_[i].bbox.y2_;
      // kps
      for (int j = 0; j < 18; ++j) {
        top_data[pidx*63+4+3*j] = targets_[i].kps[j].x;
        top_data[pidx*63+5+3*j] = targets_[i].kps[j].y;
        top_data[pidx*63+6+3*j] = targets_[i].kps[j].v;
      }
      // num_vis
      top_data[pidx*63+58] = targets_[i].num_vis;
      // score
      top_data[pidx*63+59] = targets_[i].score;
      // id
      top_data[pidx*63+60] = targets_[i].front.id;
      // similarity
      // 61 -> 前景的相似度
      // 62 -> 最大的背景相似度
      top_data[pidx*63+61] = targets_[i].front.simi;
      // 计算最大背景相似度
      Dtype max_back_simi = 0;
      for (int k = 0; k < targets_[i].back.size(); ++k) {
        if (targets_[i].back[k].simi > max_back_simi) {
          max_back_simi = targets_[i].back[k].simi;
        }
      }
      top_data[pidx*63+62] = max_back_simi;
      // 指向下一个对象
      ++pidx;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AllocateIdLayer);
#endif

INSTANTIATE_CLASS(AllocateIdLayer);
REGISTER_LAYER_CLASS(AllocateId);

}  // namespace caffe
