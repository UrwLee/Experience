# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

sys.path.append('../')

IP = 115
test_iter=1000000
caffemodel_index = -1