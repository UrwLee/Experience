# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.path.append('../')

from PyLib.Utils import par as Upar
from PyLib.Utils import path as Upath

sys.dont_write_bytecode = True

# Create decomp-convLayer
# from_layer -> prevLayer
# out_layer -> targetLayer
# num_output -> output channels
# kernel_size/pad/stride -> kernel
# R1 -> 第一个卷积层的输出通道
# R2 -> 第二个卷积层的输出通道
# use_bias -> 是否包含bias
# init_xavier　－> 参数初始化方法
def Decomp_ConvLayer(net,from_layer="conv1",out_layer="conv2",num_output=32,kernel_size=3,pad=1,stride=1, \
    R1=12,R2=12,lr=1,decay=1,use_bias=True,init_xavier=False):
    if init_xavier:
      kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
      kwargs_bias = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    else:
      kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
      kwargs_bias = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0)
        }
    # L1
    conv_name = '{}_a'.format(out_layer)
    net[conv_name] = L.Convolution(net[from_layer], num_output=R1,
        kernel_size=1, pad=0, stride=1, **kwargs)
    start_layer = conv_name
    # L2
    conv_name = '{}_b'.format(out_layer)
    net[conv_name] = L.Convolution(net[start_layer], num_output=R2,
        kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)
    start_layer = conv_name
    # L3
    conv_name = '{}_c'.format(out_layer)
    if use_bias:
        net[conv_name] = L.Convolution(net[start_layer], num_output=num_output,
            kernel_size=1, pad=0, stride=1, **kwargs_bias)
    else:
        net[conv_name] = L.Convolution(net[start_layer], num_output=num_output,
            kernel_size=1, pad=0, stride=1, **kwargs)
    return conv_name
