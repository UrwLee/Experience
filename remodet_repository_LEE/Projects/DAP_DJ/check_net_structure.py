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
sys.dont_write_bytecode = True


from solverParam import *
from PyLib.NetLib.VggNet import VGG16_BaseNet_ChangeChannel
from PyLib.NetLib.YoloNet import YoloNetPart
from AddC6 import *
from DAPNet import DAPNetVGGReduce
from DAPNet import DAPNetDarkPoseEltWise

############################# NOTE TO CHANGE THE BASE FUNCTION!!!!!!!!!!!!!
dims = [1, 3, 288, 512]
net = caffe.NetSpec()
net["data"] = L.DummyData(shape={'dim': dims})
strid_convs = [1, 1, 1, 0, 0]
kernel_size_first = 3
stride_first = 2
channel_divides = (1, 1, 1, 1, 1)
num_channel_conv5_5 = 512
lr_basenet = 0.1
# BaseNet
channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
strides = (True, True, True, False, False)
kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
pool_last = (False,False,False,True,True)
net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
                                  kernels=kernels,freeze_layers=[], pool_last=pool_last,lr_mult=0.1,decay_mult=1,
                                  use_global_stats = None,flag_withparamname=True,use_bn=False,pose_string='_pose')



net_param = net.to_proto()


del net_param.layer[0]
net_param.input.extend(["data"])
net_param.input_shape.extend([caffe_pb2.BlobShape(dim=dims)])


fh = open('trunck_det.prototxt','w')
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

