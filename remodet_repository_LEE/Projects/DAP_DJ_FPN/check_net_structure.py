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
from caffe.proto import caffe_pb2
from FaceBoxAlikeNet import FaceBoxAlikeNet
from FaceBoxFPNNet import FaceBoxFPNAdapNet
from FaceBoxFPNNet import FaceBoxFPNNet
############################# NOTE TO CHANGE THE BASE FUNCTION!!!!!!!!!!!!!
dims = [1, 3, 288, 512]
net = caffe.NetSpec()
net["data"] = L.DummyData(shape={'dim': dims})
# net["label"] = L.DummyData(shape={'dim': [1,1,1,7]})
# c = ((32,),(32,),(32,32,128),(64,64,128),(128,128,256))
# # c = ((64,64),(128,128),(256,256,256),(512,512,512),(512,512,512))
# net = VGG16_BaseNet_ChangeChannel(net,"data",channels=c)
# net = DAPNetVGGReduce(net)
# net = YoloNetPart(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5, final_pool=True, lr=1, decay=1)
channels = ((32,),(32,),(64,32,128),(128,64,128,64,256),(256,128,256,128,256))
net = FaceBoxFPNAdapNet(net)
net_param = net.to_proto()
print(len(net.keys()))


del net_param.layer[0]
net_param.input.extend(["data"])
net_param.input_shape.extend([caffe_pb2.BlobShape(dim=dims)])


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

