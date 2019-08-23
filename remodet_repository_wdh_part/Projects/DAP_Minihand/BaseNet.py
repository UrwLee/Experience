# -*- coding: utf-8 -*-
import os
import sys
import math
sys.dont_write_bytecode = True
#sys.path.insert(0, "/home/zhangming/work/minihand/remodet_repository/python")
sys.path.insert(0, '/home/xjx/work/remodet_repository_DJ/python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.InceptionLayer import *
from PyLib.NetLib.MultiScaleLayer import *
# ##############################################################################
# ------------------------------------------------------------------------------
# BaseNet
# ResNet-Unit
def ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel,bridge = False,num_channel_change = 0,
                     flag_hasresid = True,channel_scale = 4,check_macc = False,lr=1,decay=1):
    add_layer = name_prefix + '_1x1Conv'
    ConvBNUnitLayer(net, base_layer, add_layer, use_bn=True, use_relu=True,
					num_output=num_channel/channel_scale, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=False,lr_mult=lr,
                    decay_mult=decay)
    from_layer = add_layer
    add_layer = name_prefix + '_3x3Conv'
    if num_channel_change != 0:
        num_channel = num_channel_change
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=num_channel, kernel_size=3, pad=1, stride=stride, use_scale=True,
                    n_group=1,lr_mult=lr,
                    decay_mult=decay)
    if flag_hasresid:
        from_layer = add_layer
        if stride == 2:
            feature_layers = []
            feature_layers.append(net[from_layer])
            add_layer = name_prefix + '_AVEPool'
            net[add_layer] = L.Pooling(net[base_layer], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)
            feature_layers.append(net[add_layer])
            add_layer = name_prefix + '_Concat'
            net[add_layer] = L.Concat(*feature_layers, axis=1)
        else:
            add_layer1 = from_layer
            if bridge:
                from_layer = base_layer
                add_layer = name_prefix + '_bridge'
                ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                                num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,lr_mult=lr,
                    decay_mult=decay)
                add_layer2 = add_layer
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)

def ResidualVariant_Base_A(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,num_channel_deconv = 128,lr=1,decay=1,flag_pool1 = True,flag_with_deconv = False,add_strs=""):
    out_layer = 'conv1' + add_strs
    # NOTE: @ZhangM -> BN = false
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=32, kernel_size=7, pad=3, stride=2, use_scale=True, leaky=False, lr_mult=lr,
                    decay_mult=decay)
    if flag_pool1:
        from_layer = out_layer
        out_layer = 'pool1' + add_strs
        net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    for layer in xrange(0, len(use_sub_layers)):
        num_channel_layer = num_channels[layer]
        output_channel_layer = output_channels[layer]
        for sublayer in xrange(use_sub_layers[layer]):
            base_layer = out_layer
            name_prefix = 'conv{}_{}'.format(layer + 2, sublayer + 1) + add_strs
            if sublayer == 0:
                stride = 2
            else:
                stride = 1
            if sublayer == 1:
                bridge = True
            else:
                bridge = False
            if not output_channel_layer == 0 and sublayer == use_sub_layers[layer] - 1:
                num_channel_change = output_channel_layer
                bridge = True
            else:
                num_channel_change = 0
            ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel_layer, bridge=bridge, num_channel_change=num_channel_change,
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False,lr=lr, decay=decay)
            out_layer = name_prefix + '_relu'
    if flag_with_deconv:
        deconv_param = {
            'num_output': num_channel_deconv,
            'kernel_size': 2,
            'pad': 0,
            'stride': 2,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),
            'group': 1,
        }
        kwargs_deconv = {
            'param': [dict(lr_mult=lr, decay_mult=decay)],
            'convolution_param': deconv_param
        }
        from_layer = "conv3_{}{}_Add".format(use_sub_layers[-1], add_strs)
        add_layer = from_layer + "_deconv"
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    return net

def HandBase(net, data_layer="data", use_bn=False):
    out_layer = 'conv1_recon'
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=use_bn, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=1,
                    decay_mult=1)
    from_layer = out_layer
    out_layer = 'conv2_hand'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=1,
                    decay_mult=1)
    return net
