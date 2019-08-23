# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

from ConvBNLayer import *
from PvaNet import *
from FireLayer import *
from YoloNet import *

def FSNet(net, from_layer="data", lr=1, decay=1):
    net = YoloNetPart(net, from_layer="data", use_layers=2, use_sub_layers=1, final_pool=False)
    # pool2
    net.pool2 = L.Pooling(net["conv2"], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv3
    net = FireLayer_NBN(net, "pool2", "conv3_1", s_channels=16, e_channels_1=64, e_channels_2=64, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv3_1", "conv3_2", s_channels=16, e_channels_1=64, e_channels_2=64, lr=lr, decay=decay)
    # pool3
    net.pool3 = L.Pooling(net["conv3_2"], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv4
    net = FireLayer_NBN(net, "pool3", "conv4_1", s_channels=32, e_channels_1=128, e_channels_2=128, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv4_1", "conv4_2", s_channels=32, e_channels_1=128, e_channels_2=128, lr=lr, decay=decay)
    # net = FireLayer_NBN(net, "conv4_2", "conv4_3", s_channels=32, e_channels_1=128, e_channels_2=128, lr=lr, decay=decay)
    # pool4
    net.pool4 = L.Pooling(net["conv4_2"], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv5
    net = FireLayer_NBN(net, "pool4", "conv5_1", s_channels=64, e_channels_1=256, e_channels_2=256, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv5_1", "conv5_2", s_channels=64, e_channels_1=256, e_channels_2=256, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv5_2", "conv5_3", s_channels=64, e_channels_1=256, e_channels_2=256, lr=lr, decay=decay)
    # pool5
    net.pool5 = L.Pooling(net["conv5_3"], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv6
    net = FireLayer_NBN(net, "pool5", "conv6_1", s_channels=128, e_channels_1=512, e_channels_2=512, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv6_1", "conv6_2", s_channels=128, e_channels_1=512, e_channels_2=512, lr=lr, decay=decay)
    net = FireLayer_NBN(net, "conv6_2", "conv6_3", s_channels=128, e_channels_1=512, e_channels_2=512, lr=lr, decay=decay)
    return net

def ReDNet(net, from_layer="data", use_bn=False):
    # 3/1/1 -32-64
    net = YoloNetPart(net, from_layer="data", use_bn=use_bn, use_layers=2, use_sub_layers=1, final_pool=False)
    # pool2
    net.pool2 = L.Pooling(net["conv2"], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv3
    from_layer = "pool2"
    out_layer = "conv3_1"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv3_2"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv3_3"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv3_4"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv3_5"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv3_6"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer

    out_layer = "pool3"
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    from_layer = out_layer

    out_layer = "conv4_1"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv4_2"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv4_3"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv4_4"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv4_5"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv4_6"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer

    out_layer = "pool4"
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    from_layer = out_layer

    out_layer = "conv5_1"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv5_2"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv5_3"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv5_4"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv5_5"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv5_6"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer

    out_layer = "pool5"
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    from_layer = out_layer

    out_layer = "conv6_1"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv6_3"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv6_4"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv6_5"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer
    out_layer = "conv6_6"
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    from_layer = out_layer

    return net
