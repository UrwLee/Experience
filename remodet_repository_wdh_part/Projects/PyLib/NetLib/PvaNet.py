# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

def smCReLULayer(net, from_layer, out_layer, channels=32, use_reduced_layer=False, reduced_layers=[], \
                 lr=1, decay=1):
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    power_kwargs = {'power': 1, 'scale': -1.0, 'shift': 0}
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    start_layer = from_layer
    # 1x1 convLayer
    if use_reduced_layer:
        name = "{}/reduced/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[0], \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/reduced/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=True, **bn_kwargs)
        start_layer = name
        name = "{}/reduced/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/reduced/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    # 3x3 convLayer
    if use_reduced_layer:
        name = "{}/inter/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[1], \
            kernel_size=3, pad=1, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/inter/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=False, **bn_kwargs)
        start_layer = name
        neg_name = "{}/inter/neg".format(out_layer)
        net[neg_name] = L.Power(net[start_layer], **power_kwargs)
        name = "{}/inter/concat".format(out_layer)
        net[name] = L.Concat(net[start_layer], net[neg_name], axis=1)
        start_layer = name
        name = "{}/inter/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/inter/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    else:
        name = "{}/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=channels, \
            kernel_size=3, pad=1, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=False, **bn_kwargs)
        start_layer = name
        neg_name = "{}/neg".format(out_layer)
        net[neg_name] = L.Power(net[start_layer], **power_kwargs)
        name = "{}/concat".format(out_layer)
        net[name] = L.Concat(net[start_layer], net[neg_name], axis=1)
        start_layer = name
        name = "{}/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    # 1x1
    if use_reduced_layer:
        name = "{}/out/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[2], \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/out/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=True, **bn_kwargs)
        start_layer = name
        name = "{}/out/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/out/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    return net

def smCReLULayer_NBN(net, from_layer, out_layer, channels=32, use_reduced_layer=False, reduced_layers=[], \
                 lr=1, decay=1):
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    power_kwargs = {'power': 1, 'scale': -1.0, 'shift': 0}
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    conv_nb_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    start_layer = from_layer
    # 1x1 convLayer
    if use_reduced_layer:
        name = "{}/reduced/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[0], \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/reduced/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    # 3x3 convLayer
    if use_reduced_layer:
        name = "{}/inter/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[1], \
            kernel_size=3, pad=1, stride=1, **conv_nb_kwargs)
        start_layer = name
        name = "{}/inter/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=False, **bn_kwargs)
        start_layer = name
        neg_name = "{}/inter/neg".format(out_layer)
        net[neg_name] = L.Power(net[start_layer], **power_kwargs)
        name = "{}/inter/concat".format(out_layer)
        net[name] = L.Concat(net[start_layer], net[neg_name], axis=1)
        start_layer = name
        name = "{}/inter/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/inter/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    else:
        name = "{}/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=channels, \
            kernel_size=3, pad=1, stride=1, **conv_nb_kwargs)
        start_layer = name
        name = "{}/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], in_place=False, **bn_kwargs)
        start_layer = name
        neg_name = "{}/neg".format(out_layer)
        net[neg_name] = L.Power(net[start_layer], **power_kwargs)
        name = "{}/concat".format(out_layer)
        net[name] = L.Concat(net[start_layer], net[neg_name], axis=1)
        start_layer = name
        name = "{}/scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], in_place=True, **scale_kwargs)
        start_layer = name
        name = "{}/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    # 1x1
    if use_reduced_layer:
        name = "{}/out/conv".format(out_layer)
        net[name] = L.Convolution(net[start_layer], num_output=reduced_layers[2], \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        name = "{}/out/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], in_place=True)
        start_layer = name
    return net

def mCReLULayer(net, from_layer, out_layer, reduced_channels=24, \
                inter_channels=24, output_channels=48, lr=1, decay=1, \
                use_prior_bn=True, cross_stage=False, has_pool=False):
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    power_kwargs = {'power': 1, 'scale': -1.0, 'shift': 0}
    input_kwargs = {'power': 1, 'scale': 1, 'shift': 0}
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    eltwise_kwargs = {'operation': 1, 'coeff': [1, 1]}
    # conv/1: bn/scale/relu/conv
    start_layer = from_layer
    if use_prior_bn:
        layer_name = "{}/1/bn".format(out_layer)
        name = "{}/1/pre".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=False, **bn_kwargs)
        start_layer = name
        layer_name = "{}/1/bn_scale".format(out_layer)
        name = "{}/1/bn_scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
        start_layer = name
        layer_name = "{}/1/relu".format(out_layer)
        name = "{}/1/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
        start_layer = name
    layer_name = "{}/1/conv".format(out_layer)
    name = "{}/1".format(out_layer)
    if has_pool:
        stride = 2
    else:
        stride = 1
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=reduced_channels, \
        kernel_size=1, pad=0, stride=stride, **conv_kwargs)
    start_layer = name

    # conv/2: bn/scale/relu/conv
    layer_name = "{}/2/bn".format(out_layer)
    name = "{}/2/pre".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=False, **bn_kwargs)
    start_layer = name
    layer_name = "{}/2/bn_scale".format(out_layer)
    name = "{}/2/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/2/relu".format(out_layer)
    name = "{}/2/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    layer_name = "{}/2/conv".format(out_layer)
    name = "{}/2".format(out_layer)
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=inter_channels, \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer = name

    # conv/3: bn/neg/concat/scale/relu/conv
    feaLayers = []
    bn_layer = "{}/3/bn".format(out_layer)
    bn_name = "{}/3/pre".format(out_layer)
    net[bn_name] = L.BatchNorm(net[start_layer], name=bn_layer, in_place=False, **bn_kwargs)
    feaLayers.append(net[bn_name])
    start_layer = bn_name
    neg_layer = "{}/3/neg".format(out_layer)
    neg_name = "{}/3/neg".format(out_layer)
    net[neg_name] = L.Power(net[start_layer], name=neg_layer, **power_kwargs)
    feaLayers.append(net[neg_name])
    concat_layer = "{}/3/concat".format(out_layer)
    concat_name = "{}/3/preAct".format(out_layer)
    net[concat_name] = L.Concat(*feaLayers, name=concat_layer, axis=1)
    layer_name = "{}/3/scale".format(out_layer)
    name = "{}/3/scale".format(out_layer)
    net[name] = L.Scale(net[concat_name], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/3/relu".format(out_layer)
    name = "{}/3/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    layer_name = "{}/3/conv".format(out_layer)
    name = "{}/3".format(out_layer)
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=output_channels, \
        kernel_size=1, pad=0, stride=1, **conv_kwargs)
    start_layer = name
    mlayers = []
    mlayers.append(net[name])
    # proj or input
    if cross_stage:
        layer_name = "{}/proj".format(out_layer)
        name = "{}/proj".format(out_layer)
        if has_pool:
            start_layer = "{}/1/pre".format(out_layer)
            stride = 2
        else:
            start_layer = from_layer
            stride = 1
        net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=output_channels, \
            kernel_size=1, pad=0, stride=stride, **conv_kwargs)
        mlayers.append(net[name])
    else:
        layer_name = "{}/input".format(out_layer)
        name = "{}/input".format(out_layer)
        start_layer = from_layer
        net[name] = L.Power(net[start_layer], name=layer_name, **input_kwargs)
        mlayers.append(net[name])

    # eltwise
    layer_name = out_layer
    name = out_layer
    net[name] = L.Eltwise(*mlayers, name=layer_name, **eltwise_kwargs)

    return net


def ResInceptionLayer(net, from_layer, out_layer, cross_stage=False, channels_1=64, \
                      channels_3=[48,128], channels_5=[24,48,128],channels_pool=128, \
                      channels_output=256, lr=1, decay=1, out_bn=False):
    assert len(channels_3) == 2
    assert len(channels_5) == 3
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    input_kwargs = {'power': 1, 'scale': 1, 'shift': 0}
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    convbias_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    eltwise_kwargs = {'operation': 1, 'coeff': [1, 1]}
    start_layer = from_layer
    if cross_stage:
        stride = 2
    else:
        stride = 1
    # pre-stage: bn/scale/relu
    layer_name = "{}/incep/bn".format(out_layer)
    name = "{}/incep/pre".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=False, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/bn_scale".format(out_layer)
    name = "{}/incep/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/relu".format(out_layer)
    name = "{}/incep/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    fea_layer = name

    mlayers = []
    # conv-1x1
    layer_name = "{}/incep/0/conv".format(out_layer)
    name = "{}/incep/0".format(out_layer)
    net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_1, \
        kernel_size=1, pad=0, stride=stride, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/0/bn".format(out_layer)
    name = "{}/incep/0/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/0/bn_scale".format(out_layer)
    name = "{}/incep/0/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/0/relu".format(out_layer)
    name = "{}/incep/0/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    mlayers.append(net[name])

    # conv-3x3
    layer_name = "{}/incep/1_reduce/conv".format(out_layer)
    name = "{}/incep/1_reduce".format(out_layer)
    net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_3[0], \
        kernel_size=1, pad=0, stride=stride, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_reduce/bn".format(out_layer)
    name = "{}/incep/1_reduce/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_reduce/bn_scale".format(out_layer)
    name = "{}/incep/1_reduce/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_reduce/relu".format(out_layer)
    name = "{}/incep/1_reduce/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    layer_name = "{}/incep/1_0/conv".format(out_layer)
    name = "{}/incep/1_0".format(out_layer)
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_3[1], \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_0/bn".format(out_layer)
    name = "{}/incep/1_0/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_0/bn_scale".format(out_layer)
    name = "{}/incep/1_0/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/1_0/relu".format(out_layer)
    name = "{}/incep/1_0/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    mlayers.append(net[name])

    # conv-5x5
    layer_name = "{}/incep/2_reduce/conv".format(out_layer)
    name = "{}/incep/2_reduce".format(out_layer)
    net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_5[0], \
        kernel_size=1, pad=0, stride=stride, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_reduce/bn".format(out_layer)
    name = "{}/incep/2_reduce/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_reduce/bn_scale".format(out_layer)
    name = "{}/incep/2_reduce/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_reduce/relu".format(out_layer)
    name = "{}/incep/2_reduce/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    layer_name = "{}/incep/2_0/conv".format(out_layer)
    name = "{}/incep/2_0".format(out_layer)
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_5[1], \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_0/bn".format(out_layer)
    name = "{}/incep/2_0/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_0/bn_scale".format(out_layer)
    name = "{}/incep/2_0/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_0/relu".format(out_layer)
    name = "{}/incep/2_0/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    layer_name = "{}/incep/2_1/conv".format(out_layer)
    name = "{}/incep/2_1".format(out_layer)
    net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_5[2], \
        kernel_size=3, pad=1, stride=1, **conv_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_1/bn".format(out_layer)
    name = "{}/incep/2_1/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_1/bn_scale".format(out_layer)
    name = "{}/incep/2_1/bn_scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/incep/2_1/relu".format(out_layer)
    name = "{}/incep/2_1/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    mlayers.append(net[name])

    # pool
    if cross_stage:
        layer_name = "{}/incep/pool".format(out_layer)
        name = "{}/incep/pool".format(out_layer)
        net[name] = L.Pooling(net[fea_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2)
        start_layer = name
        layer_name = "{}/incep/poolproj/conv".format(out_layer)
        name = "{}/incep/poolproj".format(out_layer)
        net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_pool, \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        layer_name = "{}/incep/poolproj/bn".format(out_layer)
        name = "{}/incep/poolproj/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
        start_layer = name
        layer_name = "{}/incep/poolproj/bn_scale".format(out_layer)
        name = "{}/incep/poolproj/bn_scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
        start_layer = name
        layer_name = "{}/incep/poolproj/relu".format(out_layer)
        name = "{}/incep/poolproj/relu".format(out_layer)
        net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
        mlayers.append(net[name])

    # incep
    layer_name = "{}/incep".format(out_layer)
    name = "{}/incep".format(out_layer)
    net[name] = L.Concat(*mlayers, name=layer_name, axis=1)
    start_layer = name
    # out-conv
    scLayers = []
    if not out_bn:
        layer_name = "{}/out/conv".format(out_layer)
        name = "{}/out".format(out_layer)
        net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_output, \
            kernel_size=1, pad=0, stride=1, **convbias_kwargs)
        scLayers.append(net[name])
    else:
        layer_name = "{}/out/conv".format(out_layer)
        name = "{}/out".format(out_layer)
        net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_output, \
            kernel_size=1, pad=0, stride=1, **conv_kwargs)
        start_layer = name
        layer_name = "{}/out/bn".format(out_layer)
        name = "{}/out/bn".format(out_layer)
        net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
        start_layer = name
        layer_name = "{}/out/bn_scale".format(out_layer)
        name = "{}/out/bn_scale".format(out_layer)
        net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
        scLayers.append(net[name])

    # proj or input
    if cross_stage:
        layer_name = "{}/proj".format(out_layer)
        name = "{}/proj".format(out_layer)
        net[name] = L.Convolution(net[from_layer], name=layer_name, num_output=channels_output, \
            kernel_size=1, pad=0, stride=2, **convbias_kwargs)
        scLayers.append(net[name])
    else:
        layer_name = "{}/input".format(out_layer)
        name = "{}/input".format(out_layer)
        net[name] = L.Power(net[from_layer], name=layer_name, **input_kwargs)
        scLayers.append(net[name])

    # Eltwise
    layer_name = out_layer
    name = out_layer
    net[name] = L.Eltwise(*scLayers, name=layer_name, **eltwise_kwargs)

    return net

def pva_convHeader(net, from_layer, out_layer, use_pool=True, lr=1, decay=1):
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    power_kwargs = {'power': 1, 'scale': -1.0, 'shift': 0}
    conv_kwargs = {
        'param': [dict(lr_mult=lr, decay_mult=decay)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    layer_name = "{}/conv".format(out_layer)
    name = "{}/conv".format(out_layer)
    net[name] = L.Convolution(net[from_layer], name=layer_name, num_output=16, \
        kernel_size=7, pad=3, stride=2, **conv_kwargs)
    start_layer = name
    layer_name = "{}/bn".format(out_layer)
    name = "{}/bn".format(out_layer)
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    feaLayers = []
    feaLayers.append(net[name])
    start_layer = name
    neg_layer = "{}/neg".format(out_layer)
    neg_name = "{}/neg".format(out_layer)
    net[neg_name] = L.Power(net[start_layer], name=neg_layer, **power_kwargs)
    feaLayers.append(net[neg_name])
    concat_layer = "{}/concat".format(out_layer)
    concat_name = out_layer
    net[concat_name] = L.Concat(*feaLayers, name=concat_layer, axis=1)
    start_layer = concat_name
    layer_name = "{}/scale".format(out_layer)
    name = "{}/scale".format(out_layer)
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "{}/relu".format(out_layer)
    name = "{}/relu".format(out_layer)
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
    start_layer = name
    # pool
    if use_pool:
        layer_name = "pool1"
        name = "pool1"
        net[name] = L.Pooling(net[start_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2)

    return net

def PvaNet(net, from_layer="data", lr=1, decay=1):
    # input Layer
    pva_convHeader(net, from_layer, "conv1_1", use_pool=True, lr=lr, decay=decay)
    # conv2_1
    mCReLULayer(net, "pool1", "conv2_1", reduced_channels=24, \
                    inter_channels=24, output_channels=64, lr=lr, decay=decay, \
                    use_prior_bn=False, cross_stage=True, has_pool=False)
    # conv2_2
    mCReLULayer(net, "conv2_1", "conv2_2", reduced_channels=24, \
                    inter_channels=24, output_channels=64, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=False, has_pool=False)
    # conv2_3
    mCReLULayer(net, "conv2_2", "conv2_3", reduced_channels=24, \
                    inter_channels=24, output_channels=64, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=False, has_pool=False)
    # conv3_1
    mCReLULayer(net, "conv2_3", "conv3_1", reduced_channels=48, \
                    inter_channels=48, output_channels=128, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=True, has_pool=True)
    # conv3_2
    mCReLULayer(net, "conv3_1", "conv3_2", reduced_channels=48, \
                    inter_channels=48, output_channels=128, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=False, has_pool=False)
    # conv3_3
    mCReLULayer(net, "conv3_2", "conv3_3", reduced_channels=48, \
                    inter_channels=48, output_channels=128, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=False, has_pool=False)
    # conv3_4
    mCReLULayer(net, "conv3_3", "conv3_4", reduced_channels=48, \
                    inter_channels=48, output_channels=128, lr=lr, decay=decay, \
                    use_prior_bn=True, cross_stage=False, has_pool=False)
    # conv4_1
    ResInceptionLayer(net, "conv3_4", "conv4_1", cross_stage=True, channels_1=64, \
                          channels_3=[48,128], channels_5=[24,48,48],channels_pool=128, \
                          channels_output=256, lr=lr, decay=decay)
    # conv4_2
    ResInceptionLayer(net, "conv4_1", "conv4_2", cross_stage=False, channels_1=64, \
                          channels_3=[64,128], channels_5=[24,48,48], \
                          channels_output=256, lr=lr, decay=decay)
    # conv4_3
    ResInceptionLayer(net, "conv4_2", "conv4_3", cross_stage=False, channels_1=64, \
                          channels_3=[64,128], channels_5=[24,48,48], \
                          channels_output=256, lr=lr, decay=decay)
    # conv4_4
    ResInceptionLayer(net, "conv4_3", "conv4_4", cross_stage=False, channels_1=64, \
                          channels_3=[64,128], channels_5=[24,48,48], \
                          channels_output=256, lr=lr, decay=decay)
    # conv5_1
    ResInceptionLayer(net, "conv4_4", "conv5_1", cross_stage=True, channels_1=64, \
                          channels_3=[96,192], channels_5=[32,64,64],channels_pool=128, \
                          channels_output=384, lr=lr, decay=decay)
    # conv5_2
    ResInceptionLayer(net, "conv5_1", "conv5_2", cross_stage=False, channels_1=64, \
                          channels_3=[96,192], channels_5=[32,64,64], \
                          channels_output=384, lr=lr, decay=decay)
    # conv5_3
    ResInceptionLayer(net, "conv5_2", "conv5_3", cross_stage=False, channels_1=64, \
                          channels_3=[96,192], channels_5=[32,64,64], \
                          channels_output=384, lr=lr, decay=decay)
    # conv5_4
    ResInceptionLayer(net, "conv5_3", "conv5_4", cross_stage=False, channels_1=64, \
                          channels_3=[96,192], channels_5=[32,64,64], \
                          channels_output=384, lr=lr, decay=decay, out_bn=True)
    # build last bn/scale/relu
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=True),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          }
    start_layer = net.keys()[-1]
    layer_name = "conv5_4/last_bn"
    name = layer_name
    net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
    start_layer = name
    layer_name = "conv5_4/last_bn_scale"
    name = layer_name
    net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
    start_layer = name
    layer_name = "conv5_4/last_relu"
    name = layer_name
    net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)

    return net
