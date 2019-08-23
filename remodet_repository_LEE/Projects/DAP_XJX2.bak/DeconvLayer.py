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

# Create Deconv-BN UnitLayer
def DeconvBNUnitLayer(net, from_layer, out_layer, use_bn, use_relu, num_output, \
    kernel_size, pad, stride, lr_mult=1, decay_mult=1, \
    dilation=1, use_conv_bias=False, use_scale=True, eps=0.001, \
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn', \
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',leaky=False,leaky_ratio=0.1, \
    init_xavier=True):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    if use_conv_bias:
        if init_xavier:
            kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
                'convolution_param': {
                    'num_output': num_output,
                    'kernel_size': kernel_size,
                    'pad': pad,
                    'stride': stride,
                    'weight_filler': dict(type='xavier'),
                    'bias_filler': dict(type='constant', value=0)
                }
            }
        else:
            kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
                'convolution_param': {
                    'num_output': num_output,
                    'kernel_size': kernel_size,
                    'pad': pad,
                    'stride': stride,
                    'weight_filler': dict(type='gaussian', std=0.01),
                    'bias_filler': dict(type='constant', value=0)
                }
            }
    else:
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
            'convolution_param': {
                'num_output': num_output,
                'kernel_size': kernel_size,
                'pad': pad,
                'stride': stride,
                'weight_filler': dict(type='xavier'),
                'bias_term': False
            }
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr_mult, decay_mult=0), dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    if init_xavier:
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }
    else:
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)
            }
  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = Upar.UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = Upar.UnpackVariable(pad, 2)
  [stride_h, stride_w] = Upar.UnpackVariable(stride, 2)
  net[conv_name] = L.Deconvolution(net[from_layer], **kwargs)

  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
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
    relu_name = '{}_relu'.format(conv_name)
    if leaky:
      leaky_kwargs = {"negative_slope":leaky_ratio}
      net[relu_name] = L.ReLU(net[conv_name], in_place=True,**leaky_kwargs)
    else:
      net[relu_name] = L.ReLU(net[conv_name], in_place=True)