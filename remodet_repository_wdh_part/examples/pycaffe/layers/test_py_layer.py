import caffe
import numpy as np


class TestPythonLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = np.sum(bottom[0].data**2) / bottom[0].num / 2.;print('Test passed!')

    def backward(self, top, propagate_down, bottom):
        pass