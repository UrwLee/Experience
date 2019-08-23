#include "caffe/tracker/fregressor_base.hpp"

namespace caffe {

template <typename Dtype>
FRegressorBase<Dtype>::FRegressorBase() {
}

INSTANTIATE_CLASS(FRegressorBase);

}
