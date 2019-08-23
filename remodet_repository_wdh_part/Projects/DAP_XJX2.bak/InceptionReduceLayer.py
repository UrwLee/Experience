# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *
sys.dont_write_bytecode = True

# from_layer/out_layer
# cross_stage: 跨阶段连接
# channels_1: 1x1 conv
# channels_3: reduced(1x1) -> 3x3 conv
# channels_5: reduced(1x1) -> 3x3 conv -> 3x3 conv
# channels_pool: pool (maxpool-3/2) -> 1x1 conv
# use_out_conv: 拼接后是否使用一个1x1卷基层， channels_output -> 该输出卷积层的channels
# out_bn: 输出特征是否使用bn/scale/relu处理
# use_shortcut: 是否使用残差连接
# F -->[BN/SCALE/RELU]--|-> 1x1----------------->| C |---->[CONV]-->[BN/SCALE/RELU]---->[ E ]--> [bn]Output
# |    (use_prior_bn)   |-> reduced-3x3--------->| O | (use_out_conv)   (use_bn)        [ l ]
# |                     |-> reduced-3x3-3x3----->| N |                                  [ t ]
# |                     |-> maxpool-1x1--------->|Cat|                                  [ w ]
# |------------------------------------------------------------------------------------>[ise]
#                                    (use_shortcut)
#
def InceptionReduceLayer(net, from_layer, out_layer, channels_1=1,channels_3=[],channels_5=[],channels_ave=1,inter_bn = True,leaky=False):
    fea_layer = from_layer

    concatlayers = []
    mid_layer = "{}/incep/1x1".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_1, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])
    start_layer = mid_layer
    mid_layer = "{}/incep/1_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True,num_output=channels_3[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_3[1], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/2_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3_1".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[1], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3_2".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[2], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/pool".format(out_layer)
    net[mid_layer] = L.Pooling(net[fea_layer], pool=P.Pooling.AVE, kernel_size=3, stride=1, pad=1)
    start_layer = mid_layer
    mid_layer = "{}/incep/pool_1x1".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_ave, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])
    # incep
    layer_name = "{}/incep".format(out_layer)
    name = "{}/incep".format(out_layer)
    net[name] = L.Concat(*concatlayers, name=layer_name, axis=1)

    return net

def InceptionReduceLayerStride(net, from_layer, out_layer, channels_3=[],channels_5=[], channels_3_reduce = True,inter_bn = True,leaky=False):

    fea_layer = from_layer
    concatlayers = []
    if channels_3_reduce:
        mid_layer = "{}/incep/1_reduce".format(out_layer)
        ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True,num_output=channels_3[0], kernel_size=1, pad=0,
                        stride=1, use_scale=True, leaky=leaky)
        start_layer = mid_layer
    else:
        start_layer = fea_layer
    mid_layer = "{}/incep/3x3".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_3[-1], kernel_size=3, pad=1,
                    stride=2, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    if channels_5[0]>0:
        mid_layer = "{}/incep/2_reduce".format(out_layer)
        ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[0], kernel_size=1, pad=0,
                        stride=1, use_scale=True, leaky=leaky)
        start_layer = mid_layer
    else:
        start_layer = fea_layer

    mid_layer = "{}/incep/3x3_1".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[1], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3_2".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[2], kernel_size=3, pad=1,
                    stride=2, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/pool".format(out_layer)
    net[mid_layer] = L.Pooling(net[fea_layer], pool=P.Pooling.AVE, kernel_size=3, stride=2, pad=0)
    concatlayers.append(net[mid_layer])


    # incep
    layer_name = "{}/incep".format(out_layer)
    name = "{}/incep".format(out_layer)
    net[name] = L.Concat(*concatlayers, name=layer_name, axis=1)

    return net
