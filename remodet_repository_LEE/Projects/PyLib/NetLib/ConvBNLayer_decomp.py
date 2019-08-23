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
from Conv_decomp import *

sys.dont_write_bytecode = True

# Create Conv-BN UnitLayer with decomp
def ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn, use_relu, num_output, \
    kernel_size, pad, stride, lr=1, decay=1, R1_channels=12, R2_channels=12, use_scale=True, eps=0.001, \
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn', scale_prefix='',
    scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',leaky=False,leaky_ratio=0.1, \
    init_xavier=False):
  if use_bn:
    use_bias = False
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    use_bias = True
  # Conv Layer
  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  conv_name = Decomp_ConvLayer(net,from_layer=from_layer,out_layer=conv_name,num_output=num_output, \
                   kernel_size=kernel_size,pad=pad,stride=stride,R1=R1_channels,R2=R2_channels, \
                   lr=lr,decay=decay,use_bias=use_bias,init_xavier=init_xavier)
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(out_layer)
    if leaky:
      leaky_kwargs = {"negative_slope":leaky_ratio}
      net[relu_name] = L.ReLU(net[conv_name], in_place=True,**leaky_kwargs)
    else:
      net[relu_name] = L.ReLU(net[conv_name], in_place=True)

  return conv_name
