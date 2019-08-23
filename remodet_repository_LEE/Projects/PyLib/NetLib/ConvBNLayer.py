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

# Create Conv-BN UnitLayer
def ConvBNUnitLayer(net, from_layer, out_layer, use_bn, use_relu, num_output, \
    kernel_size, pad, stride, lr_mult=1, decay_mult=1, \
    dilation=1, use_conv_bias=False, use_scale=True, eps=0.001, \
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn', \
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',leaky=False,leaky_ratio=0.1, \
    init_xavier=True,n_group=1,flag_bninplace=True,engine="CUDNN",flag_withparamname = False,pose_string='',
                    constant_value=0.1,truncvalue = -1,use_global_stats=None):
  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  if truncvalue>0:
      use_bn = False
  # pose_string='_pose'
  if use_bn:
    # parameters for convolution layer with batchnorm.
    if use_conv_bias:
        if init_xavier:
            if flag_withparamname:
                kwargs = {
                    'param': [dict(name=conv_name + "_paramconv0",lr_mult=lr_mult, decay_mult=decay_mult), dict(name=conv_name + "_paramconv1",lr_mult=2 * lr_mult, decay_mult=0)],
                    'weight_filler': dict(type='xavier'),
                    'bias_filler': dict(type='constant', value=0)
                }
            else:
                kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
                    }
        else:
            if flag_withparamname:
                kwargs = {
                    'param': [dict(name=conv_name + "_paramconv0",lr_mult=lr_mult, decay_mult=decay_mult), dict(name=conv_name + "_paramconv1",lr_mult=2 * lr_mult, decay_mult=0)],
                    'weight_filler': dict(type='gaussian', std=0.01),
                    'bias_filler': dict(type='constant', value=0)
                }
            else:
                kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
                'weight_filler': dict(type='gaussian', std=0.01),
                'bias_filler': dict(type='constant', value=0)
                    }
    else:
        if flag_withparamname:
            kwargs = {
                'param': [dict(name=conv_name + "_paramconv0",lr_mult=lr_mult, decay_mult=decay_mult)],
                'weight_filler': dict(type='gaussian', std=0.01),
                'bias_term': False,
            }
        else:
            kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
                'weight_filler': dict(type='gaussian', std=0.01),
                'bias_term': False,
                }
    # parameters for batchnorm layer.
    if flag_withparamname:
        if use_global_stats is None:
            bn_kwargs = {
                'param': [dict(name=conv_name + "_parambn0",lr_mult=0, decay_mult=0), dict(name=conv_name + "_parambn1",lr_mult=0, decay_mult=0), dict(name=conv_name + "_parambn2",lr_mult=0, decay_mult=0)],
                'eps': eps,
            }
        else:
            bn_kwargs = {
                'param': [dict(name=conv_name + "_parambn0", lr_mult=0, decay_mult=0),
                          dict(name=conv_name + "_parambn1", lr_mult=0, decay_mult=0),
                          dict(name=conv_name + "_parambn2", lr_mult=0, decay_mult=0)],
                'eps': eps,
                'use_global_stats':use_global_stats
            }
    else:
        if use_global_stats is None:
            bn_kwargs = {
                'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                'eps': eps,
                }
        else:
            bn_kwargs = {
                'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                'eps': eps,
                'use_global_stats': use_global_stats
            }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
        if flag_withparamname:
            sb_kwargs = {
                'bias_term': True,
                'param': [dict(name=conv_name + "_paramsb0",lr_mult=lr_mult, decay_mult=0), dict(name=conv_name + "_paramsb1",lr_mult=lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=constant_value),
            }
        else:
          sb_kwargs = {
              'bias_term': True,
              'param': [dict(lr_mult=lr_mult, decay_mult=0), dict(lr_mult=lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=1.0),
              'bias_filler': dict(type='constant', value=constant_value),
              }
    else:
        if flag_withparamname:
            bias_kwargs = {
                'param': [dict(name=conv_name + "_parambias0",lr_mult=lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }

        else:
          bias_kwargs = {
              'param': [dict(lr_mult=lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=0.0),
              }
  else:
    if init_xavier:
        if flag_withparamname:
            kwargs = {
                'param': [dict(name=conv_name + "_paramconv0",lr_mult=lr_mult, decay_mult=decay_mult), dict(name=conv_name + "_paramconv1",lr_mult=2 * lr_mult, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
            }
        else:
            kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
                }
    else:
        if flag_withparamname:
            kwargs = {
                'param': [dict(name=conv_name + "_paramconv0",lr_mult=lr_mult, decay_mult=decay_mult), dict(name=conv_name + "_paramconv1",lr_mult=2 * lr_mult, decay_mult=0)],
                'weight_filler': dict(type='gaussian', std=0.01),
                'bias_filler': dict(type='constant', value=0)
            }
        else:
            kwargs = {
                'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult), dict(lr_mult=2*lr_mult, decay_mult=0)],
                'weight_filler': dict(type='gaussian', std=0.01),
                'bias_filler': dict(type='constant', value=0)
                }
  if engine == "CAFFE":
      engine_param = P.Convolution.CAFFE
  else:
      engine_param = P.Convolution.CUDNN

  [kernel_h, kernel_w] = Upar.UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = Upar.UnpackVariable(pad, 2)
  [stride_h, stride_w] = Upar.UnpackVariable(stride, 2)
  conv_name_pose=conv_name+pose_string
  if kernel_h == kernel_w:
      if truncvalue>0:
        net[conv_name_pose] = L.Convolution(net[from_layer], num_output=num_output,
                                     kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=n_group,weight_satvalue=truncvalue, **kwargs)
      else:
        net[conv_name_pose] = L.Convolution(net[from_layer], num_output=num_output,
                                              kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=n_group, **kwargs)

  else:
      # if n_group<num_output/4 or check_macc:
      #     net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
      #       kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=n_group,engine=engine_param,**kwargs)
      # else:
      #     conv_param = {"num_output":num_output,"kernel_size":kernel_h, "pad":pad_h, "stride":stride_h, "group":n_group}
      #     for key in kwargs.keys():
      #         if key != "param":
      #             conv_param[key] = kwargs[key]
      #     net[conv_name] = L.ConvolutionDepthwise(net[from_layer], convolution_param=conv_param,param = kwargs["param"])

      net[conv_name_pose] = L.Convolution(net[from_layer], num_output=num_output,
                                     kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                     stride_h=stride_h, stride_w=stride_w, group=n_group,weight_satvalue=truncvalue, **kwargs)
      # if n_group < num_output/4:
      #     net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
      #     kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
      #     stride_h=stride_h, stride_w=stride_w, group=n_group,engine=engine_param,**kwargs)
      # else:
      #     conv_param = {"num_output": num_output,
      #      "kernel_h": kernel_h, "kernel_w": kernel_w, "pad_h": pad_h, "pad_w": pad_w,
      #      "stride_h": stride_h, "stride_w": stride_w, "group": n_group}
      #     for key in kwargs.keys():
      #         if key != "param":
      #           conv_param[key] = kwargs[key]
      #     net[conv_name] = L.ConvolutionDepthwise(net[from_layer], convolution_param = conv_param,param = kwargs["param"])

  if dilation > 1:
      net.update(conv_name_pose, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)+pose_string
    net[bn_name] = L.BatchNorm(net[conv_name_pose], in_place=flag_bninplace, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)+pose_string
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)+pose_string
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)+pose_string
    if use_bn:
        if use_scale:
            conv_name = sb_name
        else:
            conv_name = bias_name
    else:
        conv_name = conv_name_pose
    if leaky:
      leaky_kwargs = {"negative_slope":leaky_ratio}
      net[relu_name] = L.ReLU(net[conv_name], in_place=True,**leaky_kwargs)
    else:
      net[relu_name] = L.ReLU(net[conv_name], in_place=True)

