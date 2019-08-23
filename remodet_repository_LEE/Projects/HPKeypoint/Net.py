# -*- coding: utf-8 -*-
import os
import sys
import math
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *
from Data import *
sys.dont_write_bytecode = True
# ##############################################################################
# ##############################################################################
# ------------------------------------------------------------------------------
# 卷积通道
# size           1/2 1/4 1/4 1/8  1/8  1/8  1/16
conv_channels = [32, 64, 64, 64, 64, 64]
# 卷积核
conv_kernels  = [3, 3, 3, 3, 3, 3]
# 卷积后是否Pooling
conv_pooling  = [False,False,False,False,False,False]
# Stride
conv_stride = [2, 1,2, 1,1,1]
num_stages = 3
# 网络结构生成

def Deconv(net,from_layer,num_output,group,kernel_size,stride,lr_mult,decay_mult,use_bn,use_scale,use_relu):
    deconv_param = {
        'num_output': num_output,
        'kernel_size': kernel_size,
        'pad': 0,
        'stride': stride,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        'group': group,
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
        'convolution_param': deconv_param
    }
    out_layer = from_layer + "_deconv"
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    base_conv_name = out_layer
    from_layer = out_layer
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
    }
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=lr_mult, decay_mult=0), dict(lr_mult=lr_mult, decay_mult=0)],
        'filler': dict(type='constant', value=1.0),
        'bias_filler': dict(type='constant', value=0.2),
    }
    if use_bn:
        bn_name = '{}_bn'.format(base_conv_name)
        net[bn_name] = L.BatchNorm(net[from_layer], in_place=True, **bn_kwargs)
        from_layer = bn_name
    if use_scale:
        sb_name = '{}_scale'.format(base_conv_name)
        net[sb_name] = L.Scale(net[from_layer], in_place=True, **sb_kwargs)
        from_layer = sb_name
    if use_relu:
        relu_name = '{}_relu'.format(base_conv_name)
        net[relu_name] = L.ReLU(net[from_layer], in_place=True)

def stagenet(net,from_layer,base_layer,numlayers,num_channels,kernel_size,stage,out_layer,short_cut=True,label_heat="label_heat",label_vec="label_vec",lr=1,decay=1):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2 * lr, decay_mult=0)],
              'weight_filler': dict(type='gaussian', std=0.01),
              'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from_layer1 = from_layer
    from_layer2 = from_layer
    for layer in range(numlayers):
        # heat
        conv = "stage{}_{}".format(stage+1,layer + 1)
        net[conv] = L.Convolution(net[from_layer1], num_output=num_channels, pad=(kernel_size - 1) / 2,
                                      kernel_size=kernel_size, **kwargs)
        relu_vec = "stage{}_relu{}".format(stage+1,layer + 1)
        net[relu_vec] = L.ReLU(net[conv], in_place=True)
        from_layer1 = relu_vec

        conv = "stage{}_{}_vec".format(stage + 1, layer + 1)
        net[conv] = L.Convolution(net[from_layer2], num_output=num_channels, pad=(kernel_size - 1) / 2,
                                  kernel_size=kernel_size, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage + 1, layer + 1)
        net[relu_vec] = L.ReLU(net[conv], in_place=True)
        from_layer2 = relu_vec

    conv_name_heat = "stage{}_out".format(stage + 1)
    net[conv_name_heat] = L.Convolution(net[from_layer1], num_output=num_keypoints, pad=(kernel_size - 1) / 2, kernel_size=kernel_size, **kwargs)
    loss = "stage{}_loss".format(stage + 1)
    net[loss] = L.EuclideanLoss(net[conv_name_heat], net[label_heat], loss_weight=1)

    conv_name_vec = "stage{}_out_vec".format(stage + 1)
    net[conv_name_vec] = L.Convolution(net[from_layer2], num_output=num_limbs*2, pad=(kernel_size - 1) / 2,
                                   kernel_size=kernel_size, **kwargs)
    loss = "stage{}_loss_vec".format(stage + 1)
    net[loss] = L.EuclideanLoss(net[conv_name_vec], net[label_vec], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_name_heat])
        fea_layers.append(net[conv_name_vec])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
def HPKeypointNet(net, train=True, data_layer="data", gt_label="label", lr=1, decay=1):
    if train:
        slice_point = (num_keypoints + num_limbs*2)* resized_height / target_stride * resized_width / target_stride*train_batchsize

    else:
        slice_point = (num_keypoints + num_limbs*2) * resized_height / target_stride * resized_width / target_stride * test_batchsize

    net.label_tmp,net.label_box, = \
        L.Slice(net[gt_label], ntop=2, slice_param=dict(slice_point=[slice_point,], axis=3))
    print net.keys()
    net["silence"] = L.Silence(net.label_box,ntop=0)
    net["label_heatvec"] = L.Reshape(net.label_tmp, reshape_param={'shape':{'dim': [-1, (num_keypoints + num_limbs*2), resized_height/target_stride, resized_width/target_stride]}})
    net.label_heat, net.label_vec, = \
        L.Slice(net["label_heatvec"], ntop=2, slice_param=dict(slice_point=[num_keypoints, ], axis=1))


    from_layer = data_layer
    assert len(conv_channels) == len(conv_kernels)
    assert len(conv_channels) == len(conv_pooling)
    assert len(conv_channels) == len(conv_stride)
    # 卷积层
    for i in range(len(conv_channels)):
        out_layer = "conv{}".format(i+1)
        kernel = conv_kernels[i]
        pad = (kernel - 1) / 2
        stride = conv_stride[i]
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
                         num_output=conv_channels[i], kernel_size=kernel, pad=pad, stride=stride, \
                         use_scale=True, leaky=False, lr_mult=lr,decay_mult=decay)
        from_layer = out_layer
        if conv_pooling[i]:
            out_layer = "pool{}".format(i+1)
            net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
            from_layer = out_layer
    flag_deconv = False
    num_output = 128
    group = 1
    stride = 2
    kernel_size = 2
    if flag_deconv:
        Deconv(net, from_layer, num_output, group, kernel_size, stride, lr, decay, use_bn=True, use_scale=True, use_relu=True)
        from_layer += "_deconv"
    flag_deconv = False
    flag_conv = False
    if flag_deconv:
        Deconv(net, from_layer, num_output, group, kernel_size, stride, lr, decay, use_bn=True, use_scale=True,
               use_relu=True)
        from_layer += "_deconv"
    if flag_conv:
        out_layer = "deconv_conv1"
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
                        num_output=128, kernel_size=3, pad=1, stride=1, \
                        use_scale=True, leaky=False, lr_mult=lr, decay_mult=decay)
        from_layer = out_layer
    base_layer = from_layer
    numlayers = 2
    num_channels = 64
    kernel_size = 3
    for istage in xrange(num_stages):
        out_layer = "concat_stage{}".format(istage)
        if istage <num_stages - 1:
            short_cut = True
        else:
            short_cut = False
        stagenet(net, from_layer, base_layer, numlayers, num_channels, kernel_size, istage, out_layer, short_cut=short_cut,
                 lr=1, decay=1)
        from_layer = out_layer

    return net
