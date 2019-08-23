# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

def FireLayer(net, from_layer, out_layer, s_channels=16, e_channels_1=64, e_channels_2=64, lr=1, decay=1):
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    start_layer = from_layer
    # squeeze Layer
    name = "{}/squee/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer], num_output=s_channels, \
        kernel_size=1, pad=0, stride=1, **conv_kwargs)
    start_layer = name
    name = "{}/squee/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], in_place=True, **bn_kwargs)
    start_layer = name
    name = "{}/squee/scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
    start_layer = name
    name = "{}/squee/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], in_place=True)
    start_layer = name
    # expand-1
    start_layer_1 = start_layer
    name = "{}/expand1/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer_1], num_output=e_channels_1, \
        kernel_size=1, pad=0, stride=1, **conv_kwargs)
    start_layer_1 = name
    name = "{}/expand1/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer_1], in_place=True, **bn_kwargs)
    start_layer_1 = name
    name = "{}/expand1/scale".format(out_layer)
    net[name] = L.Scale(net[start_layer_1], in_place=True, **scale_kwargs)
    start_layer_1 = name
    name = "{}/expand1/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer_1], in_place=True)
    start_layer_1 = name
    # expand2
    start_layer_2 = start_layer
    name = "{}/expand2/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer_2], num_output=e_channels_2, \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer_2 = name
    name = "{}/expand2/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer_2], in_place=True, **bn_kwargs)
    start_layer_2 = name
    name = "{}/expand2/scale".format(out_layer)
    net[name] = L.Scale(net[start_layer_2], in_place=True, **scale_kwargs)
    start_layer_2 = name
    name = "{}/expand2/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer_2], in_place=True)
    start_layer_2 = name
    # concat
    net[out_layer] = L.Concat(net[start_layer_1], net[start_layer_2], axis=1)
    return net

def FireLayer_NBN(net, from_layer, out_layer, s_channels=16, e_channels_1=64, e_channels_2=64, lr=1, decay=1):
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    start_layer = from_layer
    # squeeze Layer
    name = "{}/squee/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer], num_output=s_channels, \
        kernel_size=1, pad=0, stride=1, **conv_kwargs)
    start_layer = name
    name = "{}/squee/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], in_place=True)
    start_layer = name
    # expand-1
    start_layer_1 = start_layer
    name = "{}/expand1/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer_1], num_output=e_channels_1, \
        kernel_size=1, pad=0, stride=1, **conv_kwargs)
    start_layer_1 = name
    name = "{}/expand1/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer_1], in_place=True)
    start_layer_1 = name
    # expand2
    start_layer_2 = start_layer
    name = "{}/expand2/conv".format(out_layer)
    net[name] = L.Convolution(net[start_layer_2], num_output=e_channels_2, \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer_2 = name
    name = "{}/expand2/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer_2], in_place=True)
    start_layer_2 = name
    # concat
    net[out_layer] = L.Concat(net[start_layer_1], net[start_layer_2], axis=1)
    return net
