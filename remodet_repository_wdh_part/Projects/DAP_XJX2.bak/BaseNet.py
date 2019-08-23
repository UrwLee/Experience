# -*- coding: utf-8 -*-
import os
import sys
import math
sys.dont_write_bytecode = True
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
# from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.InceptionLayer import *
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.VggNet import VGG16_BaseNet_ChangeChannel
# ##############################################################################
# ##############################BaseNet for Detection###########################
# ResNet-Unit
def ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel,bridge = False,num_channel_change = 0,
                     flag_hasresid = True,channel_scale = 4,check_macc = False,lr_mult=0.1,decay_mult=1.0,flag_withparamname=False,pose_string=''):
    add_layer = name_prefix + '_1x1Conv'
    ConvBNUnitLayer(net, base_layer, add_layer, use_bn=True, use_relu=True,lr_mult=lr_mult, decay_mult=decay_mult,
                    num_output=num_channel/channel_scale, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=False,
                    check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    from_layer = add_layer+pose_string

    add_layer = name_prefix + '_3x3Conv'
    if num_channel_change != 0:
        num_channel = num_channel_change
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,lr_mult=lr_mult, decay_mult=decay_mult,
                    num_output=num_channel, kernel_size=3, pad=1, stride=stride, use_scale=True,
                    n_group=1,check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    # for old_name in net.keys():
    #     print old_name,'$$$$$'
    if flag_hasresid:
        from_layer = add_layer+pose_string
        if stride == 2:
            feature_layers = []
            feature_layers.append(net[from_layer])
            add_layer = name_prefix + '_AVEPool'+pose_string
            net[add_layer] = L.Pooling(net[base_layer], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)

            feature_layers.append(net[add_layer])
            add_layer = name_prefix + '_Concat'+pose_string
            net[add_layer] = L.Concat(*feature_layers, axis=1)
        # for old_name in net.keys():
        #     print old_name,'^^^'
        else:
            add_layer1 = from_layer
            if bridge:
                from_layer = base_layer
                add_layer = name_prefix + '_bridge'
                # for old_name in net.keys():
                #     print old_name,'!!!'
                ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,lr_mult=lr_mult, decay_mult=decay_mult,
                                num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
                # for old_name in net.keys():
                #     print old_name,'~~~'
                add_layer2 = add_layer+pose_string
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'+pose_string
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'+pose_string
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
# ##############################################################################
# ------------------------------------------------------------------------------
# ResNet
def ResidualVariant_Base_A_base(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,num_channel_deconv = 128,lr=1,decay=1,add_strs="",flag_withparamname=False,pose_string=''):
    out_layer = 'conv1' + add_strs
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=0.1,
                    decay_mult=decay,flag_withparamname=flag_withparamname,pose_string=pose_string)
    out_layer=out_layer+pose_string
    from_layer = out_layer#conv1_recon
   
    out_layer = 'pool1' + add_strs+pose_string

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
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False,lr_mult=lr,decay_mult=decay,flag_withparamname=flag_withparamname,pose_string=pose_string)
            out_layer = name_prefix + '_relu'+pose_string
    return net

def VGGDarkNet(net, data_layer="data",pool_last = (False,False,False,True,True),flag_withparamname=False,pose_string=''):
    channels = ((32,), (32,), (64, 32, 128), (128, 64, 128, 64, 256), (256, 128, 256, 128, 256))
    strides = (True, True, True, False, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
    net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides, kernels=kernels,
                                      freeze_layers=[],pool_last=pool_last,flag_withparamname=flag_withparamname,pose_string=pose_string)
    return net
