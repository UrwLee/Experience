from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
from BaseNet import ResidualVariant_Base_A
net = caffe.NetSpec()
net["data"] = L.DummyData(shape={'dim': [1, 3, 256, 128]})

net = ResidualVariant_Base_A(net,use_sub_layers = (3,4,5),num_channels = (128, 256, 512),output_channels = (0,0,2048))
net_param = net.to_proto()
with open('sfds.prototxt', 'w') as f:
    print(net_param, file=f)

