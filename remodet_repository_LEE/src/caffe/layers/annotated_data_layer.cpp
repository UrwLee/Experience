#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //获取batchsize
  const int batch_size = this->layer_param_.data_param().batch_size();
  //获取标注层参数
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  //定义batch采样器
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  // 定义标签文件
  label_map_file_ = anno_data_param.label_map_file();

  // Read a data point, and use it to initialize the top blob.
  // 尝试读入一个datum,用于初始化输出blob
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  // 初始化输出shape
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  // top[0]是输出的图像数据
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  // 将预取的data全部初始化
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    // 标注类型
    has_anno_type_ = anno_datum.has_type();
    // 标签shape
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  // 载入所有的batch
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // 获取一个导入样本
    // 下面要进行增强处理
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // 声明一个采样的数据Datum
    // 该Datum为增强后的样本
    AnnotatedDatum sampled_datum;
    // 使用采样器进行裁剪增强!
    if (batch_samplers_.size() > 0) {
      // 每个采样器生成一个box
      // 获得一系列的boxes
      vector<NormalizedBBox> sampled_bboxes;
      // 每个采样器随机获取一个满足条件的box
      // 由于有多个采样器,可能得到多个裁剪的crop-boxes
      GenerateBatchSamples(anno_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the anno_datum.
        // 如果采样到了box,则随机使用其中某个box对原图片进行裁剪!!
        /**
         * 使用采样器的原因:
         * 随机裁剪的时候可能把目标对象裁剪掉,所以通过设置裁剪IOU的方法来获取,确保
         * 裁剪后目标对象不会完全丢失,仍然会保留部分
         * 那么,对应的box也会根据采样的box进行计算
         */
        // 随机使用其中裁剪box进行增强
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        // 完成裁剪,以及目标box的重新修正!
        // 原图片使用采样box进行裁剪
        // 所有的标注也按照采样box进行修订
        // 标注保留条件:与裁剪box存在交集
        this->data_transformer_->CropImage(anno_datum, sampled_bboxes[rand_idx],
                                           &sampled_datum);
      } else {
        // 如果没有采样到任何box,直接使用源图像
        sampled_datum.CopyFrom(anno_datum);
      }
    } else {
      // 如果未定义采样器,直接使用原图像
      sampled_datum.CopyFrom(anno_datum);
    }
    // Apply data transformations (mirror, scale, crop...)
    // sampled_datum已经是随机裁剪增强后的样本,其box也已经进行了修订
    // 获得batch的样本指针偏移
    int offset = batch->data_.offset(item_id);
    // 设定变换器的对应数据指针
    this->transformed_data_.set_cpu_data(top_data + offset);
    // 多个标记组
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum.has_type()) << "Some datum misses AnnotationType.";
        CHECK_EQ(anno_type_, sampled_datum.type()) <<
            "Different AnnotationType.";
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        /**
         * 采样器采样的图片经过TRANSFORM,输出到top[0]
         * 标注经过修正后,输出到transformed_anno_vec中
         */
        this->data_transformer_->Transform(sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // 统计所有剩下的boxes
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        // 将标注列表加入到<样本号,标注队列>中
        all_anno[item_id] = transformed_anno_vec;
      } else {
        // 普通分类的label标签
        this->data_transformer_->Transform(sampled_datum.datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum.datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum.datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum.datum(),
                                         &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

  // 将获取的标注列表输出到Labels中
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              // 样本号:属于哪一个样本
              top_label[idx++] = item_id;
              // 类号:属于哪一类
              top_label[idx++] = anno_group.group_label();
              // 实例号,该box属于这一类下面的第几个实例
              top_label[idx++] = anno.instance_id();
              // box的坐标
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              // 这个box是否属于difficult
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
