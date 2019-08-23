#include "caffe/reid/basic_match.hpp"

#include <string>
#include <cstdio>

namespace caffe {

/**
 * 初始化目标对象
 * １．matched_id／max_similarity -> 在pMatch匹配过程重新设置
 * ２．is_merged/can_split/num_split -> 在pMerge合并过程中重新设置　
 */
template <typename Dtype>
void initTargets(vector<pTarget<Dtype> >& targets) {
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 在匹配前，默认所有目标都没有匹配
    target.matched_id = -1;
    target.matched_similarity = 0.;
    // 在匹配前，默认所有目标都无需合并／不可分裂
    target.is_merged = false;
    target.can_split = false;
    target.num_split = 0;
    // 在匹配前，默认所有目标都没有遮挡
    target.is_occluded = false;
  }
}
template void initTargets(vector<pTarget<float> >& targets);
template void initTargets(vector<pTarget<double> >& targets);

/**
 * 完成初步匹配：
 * １．每个目标对象匹配一个最大IOU提议对象；
 * 在该函数中，对提议池和目标池中的所有对象进行了匹配标记处理
 */
template <typename Dtype>
void pMatch(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals, const Dtype iou_thre) {
  // KPS结构化定义
  const int limb[34] = {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17};
  // 遍历所有目标池对象
  for (int i = 0; i < targets.size(); ++i) {
    // 遍历每个提议对象，查找具有最大相似度的提议对象作为匹配对象
    pTarget<Dtype>& target = targets[i];
    // 最大相似度
    Dtype max_similarity = 0.;
    // 最佳匹配提议id－默认没有匹配
    int best_idx = -1;
    // 遍历所有的提议对象
    for (int j = 0; j < proposals.size(); ++j) {
      pProp<Dtype>& proposal = proposals[j];
      // ##############################
      // STEP1: 计算IOU
      // ##############################
      Dtype iou = 0;
      if (target.bbox.x1_ > proposal.bbox.x2_ || target.bbox.x2_ < proposal.bbox.x1_ ||
          target.bbox.y1_ > proposal.bbox.y2_ || target.bbox.y2_ < proposal.bbox.y1_) {
        continue;
      }
      Dtype intersec_xmin = std::max(target.bbox.x1_, proposal.bbox.x1_);
      Dtype intersec_ymin = std::max(target.bbox.y1_, proposal.bbox.y1_);
      Dtype intersec_xmax = std::min(target.bbox.x2_, proposal.bbox.x2_);
      Dtype intersec_ymax = std::min(target.bbox.y2_, proposal.bbox.y2_);
      Dtype intersec_width = intersec_xmax - intersec_xmin;
      Dtype intersec_height = intersec_ymax - intersec_ymin;
      if (intersec_width <= 0 || intersec_height <= 0) continue;
      Dtype boxsize_obj = target.bbox.get_width() * target.bbox.get_height();
      Dtype boxsize_pro = proposal.bbox.get_width() * proposal.bbox.get_height();
      Dtype intersec_size = intersec_width * intersec_height;
      iou = intersec_size / (boxsize_pro + boxsize_obj - intersec_size);
      // ##############################
      // STEP2: 计算相似度
      // ##############################
      if (iou > iou_thre) {
        Dtype similarity;
        int num_compare_limbs = 0;
        Dtype sum_compare_limbs = 0;
        for (int l = 0; l < 17; ++l) {
          int part_a = limb[2*l];
          int part_b = limb[2*l+1];
          if ((proposal.kps[part_a].v > 0.01) && (proposal.kps[part_b].v > 0.01) &&
              (target.kps[part_a].v > 0.01) && (target.kps[part_b].v > 0.01)) {
            num_compare_limbs++;
            Dtype dx_obj = target.kps[part_a].x - target.kps[part_b].x;
            Dtype dy_obj = target.kps[part_a].y - target.kps[part_b].y;
            Dtype dx_pro = proposal.kps[part_a].x - proposal.kps[part_b].x;
            Dtype dy_pro = proposal.kps[part_a].y - proposal.kps[part_b].y;
            Dtype dx = dx_pro - dx_obj;
            Dtype dy = dy_pro - dy_obj;
            sum_compare_limbs += (dx*dx + dy*dy);
          }
        }
        if (num_compare_limbs > 2) {
          similarity = iou * (1+exp(-100*(sum_compare_limbs/num_compare_limbs)));
        } else {
          similarity = iou;
        }
        // ##############################
        // STEP2: 获取最大相似度
        // ##############################
        if (similarity > max_similarity) {
          max_similarity = similarity;
          best_idx = j;
        }
      }
    }
    // ###########################################
    // 设置匹配结果: Targets & Props
    // ###########################################
    target.matched_id = best_idx;
    target.matched_similarity = max_similarity;
    if (best_idx >= 0) {
      pProp<Dtype>& matched_prop = proposals[best_idx];
      matched_prop.is_matched = true;
    }
  }
}
template void pMatch(vector<pTarget<float> >& targets, vector<pProp<float> >& proposals, const float iou_thre);
template void pMatch(vector<pTarget<double> >& targets, vector<pProp<double> >& proposals, const double iou_thre);

/**
 * 对象合并
 * 1. 具有相同匹配id的pTargets需要进行合并
 * 2. 删除被合并的对象
 * 3. 删除对象中具有同名id的对象
 * 4. 设置目标对象的分裂能力
 */
template <typename Dtype>
void pMerge(vector<pTarget<Dtype> >& targets) {
  // ##########################################
  // STEP1: 设置合并标记
  // ##########################################
  for (int i = 0; i < targets.size(); ++i) {
    // 已合并,跳过
    if (targets[i].is_merged) continue;
    pTarget<Dtype>& target = targets[i];
    // 获取匹配的prop-id
    int matched_id = target.matched_id;
    // 未匹配，跳过
    if (matched_id < 0) continue;
    // 依次处理后面的对象:　查找是否有同匹配id的目标对象，这些对象需要被合并
    for (int j = i+1; j < targets.size(); ++j) {
      pTarget<Dtype>& target_o = targets[j];
      if (target_o.matched_id == matched_id) {
        // 记录合并标记
        target_o.is_merged = true;
        // 合并Front
        pPerson<Dtype> back_p;
        back_p.id = target_o.front.id;
        back_p.pT = target_o.front.pT;
        target.back.push_back(back_p);
        // 合并back
        if (target_o.back.size() == 0) continue;
        for (int k = 0; k < target_o.back.size(); ++k) {
          pPerson<Dtype> back_pp;
          back_pp.id = target_o.back[k].id;
          back_pp.pT = target_o.back[k].pT;
          target.back.push_back(back_pp);
        }
      }
    }
  }
  // ##########################################
  // STEP2: 将合并的对象全部删除
  // ##########################################
  for (typename std::vector<pTarget<Dtype> >::iterator it = targets.begin(); it != targets.end();) {
    if (it->is_merged) {
      it = targets.erase(it);
    } else {
      ++it;
    }
  }
  // ##########################################
  // STEP3: 将当前对象中同名的对象删除
  // ##########################################
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 删除target内部具有同id的对象
    vector<int> id_pool;
    id_pool.push_back(target.front.id);
    for (typename vector<pPerson<Dtype> >::iterator it = target.back.begin(); it != target.back.end();) {
      int oid = it->id;
      bool found = false;
      for (int j = 0; j < id_pool.size(); ++j) {
        if (id_pool[j] == oid) {
          found = true;
          break;
        }
      }
      if (found) {
        it = target.back.erase(it);
      } else {
        id_pool.push_back(oid);
        ++it;
      }
    }
  }
  // ##########################################
  // STEP4: 统计目标对象的分裂能力
  // ##########################################
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 未匹配，跳过
    if (target.matched_id < 0) continue;
    // 统计分裂性
    target.num_split = target.back.size();
    target.can_split = (target.num_split > 0);
  }
  // 结束
}
template void pMerge(vector<pTarget<float> >& targets);
template void pMerge(vector<pTarget<double> >& targets);

/**
 * 1. 更新目标对象的位置信息
 * 2. 更新目标对象的特征模板
 */
template <typename Dtype>
void updateTargets(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals) {
  if (targets.size() == 0) return;
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    if (target.matched_id < 0) continue;
    // update from proposals[matched_id]
    pProp<Dtype>& proposal = proposals[target.matched_id];
    // bbox
    target.bbox = proposal.bbox;
    // kps
    target.kps.resize(18);
    for (int k = 0; k < 18; ++k) {
      target.kps[k] = proposal.kps[k];
    }
    // num_vis
    target.num_vis = proposal.num_vis;
    // score
    target.score = proposal.score;
    // flags
    target.is_trash = false;
    target.miss_frame = 0;
    // 特征模板
    target.pT->Reshape(1,1,11,512);
    target.pT->set_cpu_data(proposal.fptr);
  }
}
template void updateTargets(vector<pTarget<float> >& targets, vector<pProp<float> >& proposals);
template void updateTargets(vector<pTarget<double> >& targets, vector<pProp<double> >& proposals);

/**
 * 处理未匹配目标对象
 * １．使用Tracker进行跟踪，暂时不使用
 * ２．目标已经消失，直接删除
 * NOTE: 此处将其标记为回收对象，将其ID保留10帧　
 * 回收对象停留过长时间后，直接删除
 */
template <typename Dtype>
void processUnmatchedTargets(vector<pTarget<Dtype> >& targets){
  for (typename vector<pTarget<Dtype> >::iterator it = targets.begin(); it != targets.end();) {
    if (it->matched_id < 0) {
      // 未匹配
      it->is_trash = true;
      it->miss_frame++;
      if (it->miss_frame > 10) {
        it = targets.erase(it);
      }
    } else {
      // 匹配，指向下一个对象
      ++it;
    }
  }
}
template void processUnmatchedTargets(vector<pTarget<float> >& targets);
template void processUnmatchedTargets(vector<pTarget<double> >& targets);

/**
 * 计算两个颜色模板之间的综合相似度
 * thre -> Limbs相似度超过阈值则进行统计
 */
template <typename Dtype>
void fsimilarity(const Blob<Dtype>& f_src, const Blob<Dtype>& f_dst, const Dtype thre, Dtype* s) {
  CHECK_EQ(f_src.count(), f_dst.count()) << "Feature Templates are not euqal.";
  CHECK_EQ(f_src.height(), 11);
  CHECK_EQ(f_src.width(), 512);
  CHECK_EQ(f_dst.height(), 11);
  CHECK_EQ(f_dst.width(), 512);
  // 分段计算
  vector<Dtype> simi_limbs(9,0);
  // 躯干相比于Limbs的权值
  Dtype torso_weight = 2.;
  // 结构化权重(相比于bbox的权重)
  // Dtype w_str = 8.;
  // Limbs/Torso/bbox的相似度
  Dtype s_limbs = 0.;
  Dtype s_torso = 0.;
  // 结构化相似度及统计权值
  Dtype s_str = 0.;
  Dtype str_len = 0.;
  // 特征模板数据指针
  const Dtype* src_data = f_src.cpu_data();
  const Dtype* dst_data = f_dst.cpu_data();
  // ###############################################################
  // STEP1: 计算匹配的线段, 相似度超过阈值，认为是匹配的线段，用于进行计算
  // ###############################################################
  // 超过阈值的线段数
  int num_limbs = 0;
  // 累加相似度和
  Dtype sum_limbs = 0.;
  for (int i = 0; i < 9; ++i) {
    // 不可见-不统计
    if (src_data[i*512] < -0.1 || dst_data[i*512] < -0.1) {
      continue;
    } else {
      //　可见，计算相似度
      simi_limbs[i] = caffe_cpu_dot<Dtype>(512, src_data+i*512, dst_data+i*512);
      // 相似度超过阈值，认为是匹配的线段
      if (simi_limbs[i] > thre) {
        num_limbs++;
        sum_limbs += simi_limbs[i];
      }
    }
  }
  // 至少需要４条线段进行平均相似度计算
  if (num_limbs >= 3) {
    s_limbs = sum_limbs / num_limbs;
    s_str += s_limbs;
    str_len++;
  }
  // ###############################################################
  // 计算TORSO相似度
  // ###############################################################
  // 不可见，跳过
  if (src_data[9*512] < -0.1 || dst_data[9*512] < -0.1) { s_torso = 0; }
  else {
    s_torso = caffe_cpu_dot<Dtype>(512, src_data+9*512, dst_data+9*512);
    s_str += torso_weight * s_torso;
    str_len += torso_weight;
  }
  // ###############################################################
  // 计算结构化颜色相似度
  // ###############################################################
  // 结构化数据有效
  if (str_len > 0) {
    s_str /= str_len;
  }
  // ###############################################################
  // 计算BBOX颜色相似度
  // ###############################################################
  // Dtype s_bbox = caffe_cpu_dot<Dtype>(512, src_data+10*512, dst_data+10*512);
  // ###############################################################
  // 综合相似度评价
  // ###############################################################
  // 当目标存在遮挡时，目标的特征模板将不再更新，此时s_bbox会比较小，主要依靠s_str来计算相似度
  // 当目标不存在遮挡时，目标的特征模板将不断更新，此时s_bbox接近于１，相似度主要依靠s_bbox
  // 也可以采用两者总和的方法来判定
  *s = std::max(s_str, (Dtype)0);
  // *s = s_str > 0.01 ? ((s_bbox + w_str * s_str)/(1 + w_str)) : s_bbox;

}
template void fsimilarity(const Blob<float>& f_src, const Blob<float>& f_dst, const float thre, float* s);
template void fsimilarity(const Blob<double>& f_src, const Blob<double>& f_dst, const double thre, double* s);

/**
 * 判断当前目标对象的特征属于其内部哪一个person
 * 1. 确定当前目标对象的前景id
 * thre -> 相似度计算的Limbs阈值
 * 方法：
 * 1. 选取特征模板与内部所有成员进行相似度计算，选取相似度最高的进行
 * 2. 交换前景与某个back[]之间的id和模板地址指针
 */
template <typename Dtype>
void foreTargetsUpdate(vector<pTarget<Dtype> >& targets, const Dtype thre) {
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 回收对象
    if (target.is_trash || target.matched_id < 0) continue;
    // 默认为前景对象
    int fore_idx = 0;
    Dtype best_simi = 0;
    fsimilarity(*(target.pT.get()), *(target.front.pT.get()), thre, &best_simi);
    target.front.simi = best_simi;
    // 无遮挡对象
    if (target.back.size() == 0) continue;
    // 统计back
    for (int j = 0; j < target.back.size(); ++j) {
      Dtype simi;
      fsimilarity(*(target.pT.get()), *(target.back[j].pT.get()), thre, &simi);
      target.back[j].simi = simi;
      if (simi > best_simi) {
        best_simi = simi;
        fore_idx = j + 1;
      }
    }
    // 更新前景
    if (fore_idx > 0) {
      // 交换id
      int temp_id = target.front.id;
      target.front.id = target.back[fore_idx-1].id;
      target.back[fore_idx-1].id = temp_id;
      // 交换特征模板地址
      Dtype* back_ptr = target.front.pT->mutable_cpu_data();
      Dtype* front_ptr = target.back[fore_idx-1].pT->mutable_cpu_data();
      target.front.pT->set_cpu_data(front_ptr);
      target.back[fore_idx-1].pT->set_cpu_data(back_ptr);
      // 交换simi
      Dtype temp_simi = target.front.simi;
      target.front.simi = target.back[fore_idx-1].simi;
      target.back[fore_idx-1].simi = temp_simi;
    }
  }
}
template void foreTargetsUpdate(vector<pTarget<float> >& targets, const float thre);
template void foreTargetsUpdate(vector<pTarget<double> >& targets, const double thre);

/**
 * 计算每个目标对象的遮挡情况，如果遮挡部位超过阈值，则停止目标的更新
 * max of Coverage(proposals, other targets)
 * 超过阈值，则认为被遮挡，否则，没有被遮挡
 * thre -> coverage的阈值
 */
template <typename Dtype>
void updateOccludedStatus(vector<pTarget<Dtype> >& targets, vector<pProp<Dtype> >& proposals, const Dtype thre) {
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 回收对象／未匹配对象直接跳过
    if (target.matched_id < 0) continue;
    if (target.is_trash) continue;
    Dtype max_coverage = 0;
    // 首先计算与其他targets的最大Coverage
    Dtype boxsize_obj = target.bbox.get_width() * target.bbox.get_height();
    // ###############################
    // 计算与其他目标对象的Coverage
    // ###############################
    for (int j = 0; j < targets.size(); ++j) {
      if (j == i) continue;
      Dtype coverage;
      pTarget<Dtype>& target_j = targets[j];
      if (target_j.is_trash || target_j.matched_id < 0) continue;
      if (target.bbox.x1_ > target_j.bbox.x2_ || target.bbox.x2_ < target_j.bbox.x1_ ||
          target.bbox.y1_ > target_j.bbox.y2_ || target.bbox.y2_ < target_j.bbox.y1_) {
        continue;
      }
      Dtype intersec_xmin = std::max(target.bbox.x1_, target_j.bbox.x1_);
      Dtype intersec_ymin = std::max(target.bbox.y1_, target_j.bbox.y1_);
      Dtype intersec_xmax = std::min(target.bbox.x2_, target_j.bbox.x2_);
      Dtype intersec_ymax = std::min(target.bbox.y2_, target_j.bbox.y2_);
      Dtype intersec_width = intersec_xmax - intersec_xmin;
      Dtype intersec_height = intersec_ymax - intersec_ymin;
      if (intersec_width <= 0 || intersec_height <= 0) continue;
      Dtype intersec_size = intersec_width * intersec_height;
      coverage = intersec_size / boxsize_obj;
      if (coverage > max_coverage) {
        max_coverage = coverage;
      }
    }
    // ###############################
    // 计算与提议对象的Coverage
    // ###############################
    for (int p = 0; p < proposals.size(); ++p) {
      pProp<Dtype>& prop = proposals[p];
      Dtype coverage;
      if (p == target.matched_id) continue;
      if (target.bbox.x1_ > prop.bbox.x2_ || target.bbox.x2_ < prop.bbox.x1_ ||
          target.bbox.y1_ > prop.bbox.y2_ || target.bbox.y2_ < prop.bbox.y1_) {
        continue;
      }
      Dtype intersec_xmin = std::max(target.bbox.x1_, prop.bbox.x1_);
      Dtype intersec_ymin = std::max(target.bbox.y1_, prop.bbox.y1_);
      Dtype intersec_xmax = std::min(target.bbox.x2_, prop.bbox.x2_);
      Dtype intersec_ymax = std::min(target.bbox.y2_, prop.bbox.y2_);
      Dtype intersec_width = intersec_xmax - intersec_xmin;
      Dtype intersec_height = intersec_ymax - intersec_ymin;
      if (intersec_width <= 0 || intersec_height <= 0) continue;
      Dtype intersec_size = intersec_width * intersec_height;
      coverage = intersec_size / boxsize_obj;
      if (coverage > max_coverage) {
        max_coverage = coverage;
      }
    }
    // ###############################
    // 设置遮挡标记
    // ###############################
    target.is_occluded = max_coverage > thre;
  }
}
template void updateOccludedStatus(vector<pTarget<float> >& targets, vector<pProp<float> >& proposals, const float thre);
template void updateOccludedStatus(vector<pTarget<double> >& targets, vector<pProp<double> >& proposals, const double thre);

/**
 * 更新模板，默认只更新没有back(没有遮挡对象)的目标对象
 * scale_str -> 结构模板的更新率，较小
 * scale_area -> 区域模板的更新率，较大
 */
template <typename Dtype>
void updateTemplates(vector<pTarget<Dtype> >& targets, const Dtype scale_str, const Dtype scale_area) {
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    // 未匹配／回收对象，跳过
    if (target.matched_id < 0) continue;
    if (target.is_trash) continue;
    // 有遮挡对象，跳过
    if (target.back.size() > 0) continue;
    // 目标被遮挡，跳过
    if (target.is_occluded) continue;
    // LOG(INFO) << "Target " << i << " is updated.";
    // #############################################
    // 更新front的模板：包括0-8／9／10
    // #############################################
    // 首先str -> 0-8+9 [10]
    caffe_cpu_axpby<Dtype>(10*512, scale_str, target.pT->cpu_data(),
                    1.0-scale_str, target.front.pT->mutable_cpu_data());
    // 然后area -> 10 [1]
    caffe_cpu_axpby<Dtype>(512, scale_area, target.pT->cpu_data()+10*512,
                    1.0-scale_area, target.front.pT->mutable_cpu_data()+10*512);
    // Normalize
    for (int j = 0; j < 11; ++j) {
      Blob<Dtype> temp_data(1,1,1,512);
      caffe_sqr<Dtype>(512, target.front.pT->cpu_data()+j*512, temp_data.mutable_cpu_data());
      Dtype norm_data = pow(caffe_cpu_asum<Dtype>(512, temp_data.cpu_data()) + (Dtype)0.001, Dtype(0.5));
      caffe_cpu_scale<Dtype>(512, Dtype(1.0 / norm_data), target.front.pT->cpu_data()+j*512,
                             target.front.pT->mutable_cpu_data()+j*512);
    }
  }
}
template void updateTemplates(vector<pTarget<float> >& targets, const float scale_str, const float scale_area);
template void updateTemplates(vector<pTarget<double> >& targets, const double scale_str, const double scale_area);

/**
 * 处理未匹配的提议对象
 * 1. 是否属于分裂对象
 * 2. 属于新对象
 * 3. 噪声          【认为没有噪声】
 * 分裂对象需要满足的条件：
 * (1) 搜索所有可分裂目标，定义该提议对象的最大IOU值
 * (2) 与该目标对象的所有back对象进行相似度对比，确定该分裂对象的ID　【务必满足相似度对比条件】
 * (3) 分裂过程：１．分裂母对象将该对应的back对象的模板进行复制，然后删除之
 *            　２．将分裂母对象的所有back对象全部复制一个副本到该分裂对象中去
 *            　３．将其他目标池中所有同名对象删除 [该对象转化为前景对象]
 *     原因：　遮挡对象有可能分裂到了分裂对象中去
 * (4) 一旦分裂成功后，将其他所有back中的同id对象全部删除。
 */
template <typename Dtype>
void processUnmatchedProposals(vector<pTarget<Dtype> >& targets,
                               vector<pProp<Dtype> >& proposals,
                               const Dtype iou_thre,
                               const Dtype simi_thre,
                               const Dtype thre_for_simi_cal) {
  for (int j = 0; j < proposals.size(); ++j) {
    pProp<Dtype>& prop = proposals[j];
    // 已经匹配，属于历史对象，跳过
    if (prop.is_matched) continue;
    // 最大IOU
    Dtype max_iou = 0.;
    // 匹配的idx
    int best_idx = -1;
    // 匹配的back id
    int best_pidx = -1;
    // 匹配的相似度
    Dtype best_simi = 0.;
    // ###################################################
    // 计算与每个可分裂对象的最大IOU
    // ###################################################
    for (int i = 0; i < targets.size(); ++i) {
      pTarget<Dtype>& target = targets[i];
      // 不可分裂对象，跳过
      if (target.back.size() == 0) continue;
      if (target.matched_id < 0 || target.is_trash) continue;
      // 计算IOU
      Dtype iou;
      if (target.bbox.x1_ > prop.bbox.x2_ || target.bbox.x2_ < prop.bbox.x1_ ||
          target.bbox.y1_ > prop.bbox.y2_ || target.bbox.y2_ < prop.bbox.y1_) {
        continue;
      }
      Dtype intersec_xmin = std::max(target.bbox.x1_, prop.bbox.x1_);
      Dtype intersec_ymin = std::max(target.bbox.y1_, prop.bbox.y1_);
      Dtype intersec_xmax = std::min(target.bbox.x2_, prop.bbox.x2_);
      Dtype intersec_ymax = std::min(target.bbox.y2_, prop.bbox.y2_);
      Dtype intersec_width = intersec_xmax - intersec_xmin;
      Dtype intersec_height = intersec_ymax - intersec_ymin;
      if (intersec_width <= 0 || intersec_height <= 0) continue;
      Dtype intersec_size = intersec_width * intersec_height;
      Dtype boxsize_pro = prop.bbox.get_width() * prop.bbox.get_height();
      Dtype boxsize_obj = target.bbox.get_width() * target.bbox.get_height();
      iou = intersec_size / (boxsize_pro+boxsize_obj-intersec_size);
      if (iou > max_iou) {
        max_iou = iou;
        best_idx = i;
      }
    }
    // ###################################################
    // 超过IOU阈值，进一步判定与遮挡对象的最大相似度
    // 如果最大相似度超过阈值，则认为是可分裂对象
    // ###################################################
    if (max_iou > iou_thre) {
      // 获得proposal的特征模板
      Blob<Dtype> pro_Template(1,1,11,512);
      pro_Template.set_cpu_data(prop.fptr);
      vector<pPerson<Dtype> >& persons = targets[best_idx].back;
      CHECK_GT(persons.size(), 0) << "Error when split. The matched targets should include occluded targets.";
      for (int p = 0; p < persons.size(); ++p) {
        pPerson<Dtype>& person = persons[p];
        Dtype simi;
        fsimilarity(pro_Template, *(person.pT.get()), thre_for_simi_cal, &simi);
        if (simi > best_simi) {
          best_simi = simi;
          best_pidx = p;
        }
      }
      if (best_simi > simi_thre) {
        prop.is_split = true;
      }
    }
    // ###################################################
    // 处理该对象：分裂或新增
    // ###################################################
    if (prop.is_split) {
      // 匹配对象
      pPerson<Dtype>& person = targets[best_idx].back[best_pidx];
      // #########################################
      // 使用当前prop创建一个新目标对象
      // #########################################
      pTarget<Dtype> new_obj;
      getNewTarget(prop, &new_obj);
      new_obj.front.id = person.id;
      new_obj.front.pT = person.pT;
      // #########################################
      // 删除分裂母对象中的对应对象
      // #########################################
      targets[best_idx].back.erase(targets[best_idx].back.begin()+best_pidx);
      // #########################################
      // 复制分裂母对象中的其他遮挡对象
      // #########################################
      for (int j = 0; j < targets[best_idx].back.size(); ++j) {
        pPerson<Dtype>& person_j = targets[best_idx].back[j];
        if (person_j.id == new_obj.front.id) continue;
        pPerson<Dtype> add_p;
        add_p.id = person_j.id;
        add_p.pT = person_j.pT;
        new_obj.back.push_back(add_p);
      }
      // #########################################
      // 删除与该分裂对象同id的所有其他遮挡对象
      // #########################################
      int del_id = new_obj.front.id;
      for (int i = 0; i < targets.size(); ++i) {
        pTarget<Dtype>& target = targets[i];
        if (target.back.size() == 0) continue;
        for (typename vector<pPerson<Dtype> >::iterator it = target.back.begin(); it != target.back.end();) {
          if (it->id == del_id) {
            it = target.back.erase(it);
          } else {
            ++it;
          }
        }
      }
      // #########################################
      // 分裂对象处理结束，添加到目标池中
      // #########################################
      targets.push_back(new_obj);
    } else {
    // －> 新增对象
      // #########################################
      // 使用当前prop创建一个新目标对象
      // #########################################
      pTarget<Dtype> new_obj;
      getNewTarget(prop, &new_obj);
      // #########################################
      // 分配新的ID以及创建前景模板
      // #########################################
      new_obj.front.id = getNewId(targets);
      new_obj.front.pT.reset(new Blob<Dtype>(1,1,11,512));
      const Dtype* f_ptr = prop.fptr;
      caffe_copy(new_obj.front.pT->count(), f_ptr, new_obj.front.pT->mutable_cpu_data());
      // #########################################
      // 添加到目标池
      // #########################################
      targets.push_back(new_obj);
    }
  }
}
template void processUnmatchedProposals(vector<pTarget<float> >& targets,
                                        vector<pProp<float> >& proposals,
                                        const float iou_thre,const float simi_thre,
                                        const float thre_for_simi_cal);
template void processUnmatchedProposals(vector<pTarget<double> >& targets,
                                        vector<pProp<double> >& proposals,
                                        const double iou_thre,const double simi_thre,
                                        const double thre_for_simi_cal);
/**
 * 创建一个新目标对象
 */
template <typename Dtype>
void getNewTarget(pProp<Dtype>& prop, pTarget<Dtype>* new_target) {
  new_target->bbox = prop.bbox;
  new_target->kps.resize(18);
  for (int i = 0; i < 18; ++i) {
    new_target->kps[i] = prop.kps[i];
  }
  new_target->num_vis = prop.num_vis;
  new_target->score = prop.score;
  // 复制特征模板
  new_target->pT.reset(new Blob<Dtype>(1,1,11,512));
  new_target->pT->set_cpu_data(prop.fptr);
}
template void getNewTarget(pProp<float>& prop, pTarget<float>* new_target);
template void getNewTarget(pProp<double>& prop, pTarget<double>* new_target);

/**
 * 获取一个新的id
 */
template <typename Dtype>
int getNewId(vector<pTarget<Dtype> >& targets) {
  // ID 从１开始选取
  int id = 1;
  while(1) {
    bool found = false;
    for (int i = 0; i < targets.size(); ++i) {
      pTarget<Dtype>& target = targets[i];
      if (target.front.id == id) { found = true; break; }
      for (int j = 0; j < target.back.size(); ++j) {
        if (target.back[j].id == id) { found = true; break; }
      }
      if (found) break;
    }
    if (found) { ++id; }
    else { return id; }
  }
}
template int getNewId(vector<pTarget<float> >& targets);
template int getNewId(vector<pTarget<double> >& targets);

template <typename Dtype>
void printInfo(vector<pTarget<Dtype> >& targets) {
  for (int i = 0; i < targets.size(); ++i) {
    pTarget<Dtype>& target = targets[i];
    LOG(INFO) << "Target " << i << ": " << target.front.id
              << ", back nums: " << target.back.size()
              << ". The first back id is: " << (target.back.size() > 0 ? target.back[0].id : -1);
  }
}
template void printInfo(vector<pTarget<float> >& targets);
template void printInfo(vector<pTarget<double> >& targets);

}
