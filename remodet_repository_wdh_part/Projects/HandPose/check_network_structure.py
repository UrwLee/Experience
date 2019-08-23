from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
from ReidNet import MultiScale_Dilation
net = caffe.NetSpec()
net["data"] = L.DummyData(shape={'dim': [1, 3, 112, 112]})
net = MultiScale_Dilation(net)
net_param = net.to_proto()
fh = open('sfds.prototxt','w')
print(net_param, file=fh)
fh.close()
print(np.sqrt(0.17))