#include "caffe/tracker/regressor_base.hpp"

namespace caffe {

template <typename Dtype>
RegressorBase<Dtype>::RegressorBase() {
}

INSTANTIATE_CLASS(RegressorBase);

}
