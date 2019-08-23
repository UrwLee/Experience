# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
import math
import os
import shutil
import stat
import subprocess
import sys
import time

sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *
sys.dont_write_bytecode = True

from  Net import HPNet
############################# NOTE TO CHANGE THE BASE FUNCTION!!!!!!!!!!!!!
dim = [1, 3, 96, 96]
net = caffe.NetSpec()
net["data"] = L.DummyData(shape={'dim': dim})
net = HPNet(net)
net_param = net.to_proto()


del net_param.layer[0]
net_param.input.extend(["data"])
net_param.input_shape.extend([caffe_pb2.BlobShape(dim=dim)])
# del net_param.layer[0]
# # print(net_param.layer)
# # del net_param.layer[0]
# # net_param.input.extend(['data'])
# # net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[1, 3, 224, 224])])
# #

fh = open('test_tmp.prototxt','w')
print(net_param, file=fh)
fh.close()
# print(os.getcwd().split('/')[-1])
# input: "data"
# input_shape{
#     dim:1
#     dim:384
#     dim:46
#     dim:46
# }

