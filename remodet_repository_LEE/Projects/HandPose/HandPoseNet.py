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
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.InceptionLayer import *
from PyLib.NetLib.MultiScaleLayer import *
# ##############################################################################
# using FocusLoss to minimize performance-degrade for class imbalance
using_focus = False
flag_useglobalpool = True
# ##############################################################################
# ##############################BaseNet for Detection###########################
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
        if stride == 2 and False:  # Disable this branch
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
                                num_output=num_channel, kernel_size=1, pad=0, stride=stride, use_scale=True,lr_mult=lr,
                    decay_mult=decay)
                add_layer2 = add_layer
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
# ##############################################################################
# ------------------------------------------------------------------------------
# ResNet
def ResidualVariant(net, data_layer="data",use_sub_layers = (3,3),num_channels = (128, 256),output_channels = (0, 0),
    channel_scale = 4,num_channel_deconv = 128,lr=1,decay=1,add_strs=""):
    out_layer = 'conv1' + add_strs
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=0,
                    decay_mult=0)
    from_layer = out_layer
    out_layer = 'conv2' + add_strs
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=lr,
                    decay_mult=decay)
    for layer in xrange(0, len(use_sub_layers)):
        num_channel_layer = num_channels[layer]
        output_channel_layer = output_channels[layer]
        for sublayer in xrange(use_sub_layers[layer]):
            base_layer = out_layer
            name_prefix = 'conv{}_{}'.format(layer + 3, sublayer + 1) + add_strs
            bridge = False
            stride = 1
            if sublayer == 0:
                stride = 2
                bridge = True
            # if sublayer == 0:
            #     bridge = True
            if not output_channel_layer == 0 and sublayer == use_sub_layers[layer] - 1:
                num_channel_change = output_channel_layer
                bridge = True
            else:
                num_channel_change = 0
            ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel_layer, bridge=bridge, num_channel_change=num_channel_change,
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False,lr=lr, decay=decay)
            out_layer = name_prefix + '_relu'
    return net, out_layer



def HandPoseResNet(net, train=True, data_layer="data", gt_label="label", lr=1, decay=1):
    from_layer = data_layer
    # ConvNet
    net, from_layer = ResidualVariant(net, data_layer=from_layer, use_sub_layers = (3,3), num_channels = (128, 256), output_channels = (0, 0),
                    channel_scale = 4, lr=1, decay=1, add_strs="")
    if flag_useglobalpool:
        add_layer = "pool_global"
        net[add_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, global_pooling=True)
        from_layer = add_layer

    # 全连接层
    fc_kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
    }
    # 全连接层
    fc_channels = [128, 128]
    # relu
    fc_relu = [True, True]
    # dropout
    fc_dropout = [True, True]
    # dropout参数
    fc_dropout_ratio = [0.5, 0.5]
    assert len(fc_channels) == len(fc_dropout)
    assert len(fc_channels) == len(fc_dropout_ratio)
    for i in range(len(fc_channels)):
        out_layer = "fc{}".format(i+1)
        net[out_layer] = L.InnerProduct(net[from_layer],num_output=fc_channels[i],**fc_kwargs)
        from_layer = out_layer
        if fc_relu[i]:
            out_layer = "fc{}_relu".format(i+1)
            net[out_layer] = L.ReLU(net[from_layer], in_place=True)
            from_layer = out_layer
        if fc_dropout[i] and train:
            out_layer = "fc{}_drop".format(i+1)
            net[out_layer] = L.Dropout(net[from_layer], in_place=True, dropout_param=dict(dropout_ratio=fc_dropout_ratio[i]))
            from_layer = out_layer
    # 结果层
    out_layer = "pred"

    net[out_layer] = L.InnerProduct(net[from_layer],num_output=10,**fc_kwargs)
    from_layer = out_layer
    # 损失层或评估层
    if train:
        out_layer = "loss"
        net[out_layer] = L.SoftmaxWithLoss(net[from_layer], net[gt_label])
        out_layer = "softmax"
        net[out_layer] = L.Softmax(net[from_layer])
        from_layer = out_layer
        out_layer = "acc"
        net[out_layer] = L.Accuracy(net[from_layer], net[gt_label])
    else:
        out_layer = "loss"
        net[out_layer] = L.SoftmaxWithLoss(net[from_layer], net[gt_label])
        
        out_layer = "softmax"
        net[out_layer] = L.Softmax(net[from_layer])
        from_layer = out_layer
        out_layer = "accu"
        net[out_layer] = L.HandPoseEval(net[from_layer],net[gt_label])
        out_layer = "acc"
        net[out_layer] = L.Accuracy(net[from_layer],net[gt_label])

    return net
