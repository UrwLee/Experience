#include "caffe/remo/basic.hpp"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
void transfer_meta(const Dtype* proposals, const int num, std::vector<PMeta<Dtype> >* meta) {
  meta->clear();
  if (num == 1 && proposals[0] < 0) {
    return;
  }
  int idx = 0;
  for (int i = 0; i < num; ++i) {
    PMeta<Dtype> pmeta;
    // box
    pmeta.bbox.x1_ = proposals[idx++];
    pmeta.bbox.y1_ = proposals[idx++];
    pmeta.bbox.x2_ = proposals[idx++];
    pmeta.bbox.y2_ = proposals[idx++];
    // kps
    pmeta.kps.resize(18);
    for (int k = 0; k < 18; ++k) {
      pmeta.kps[k].x = proposals[idx++];
      pmeta.kps[k].y = proposals[idx++];
      pmeta.kps[k].v = proposals[idx++];
    }
    // np
    pmeta.num_points = proposals[idx++];
    // score
    pmeta.score = proposals[idx++];
    // id
    pmeta.id = proposals[idx++];
    // similarity
    // pmeta.similarity = proposals[idx++];
    // max_back_similarity
    // pmeta.max_back_similarity = proposals[idx++];
    meta->push_back(pmeta);
  }
}
template void transfer_meta(const float* proposals, const int num, std::vector<PMeta<float> >* meta);
template void transfer_meta(const double* proposals, const int num, std::vector<PMeta<double> >* meta);

}
