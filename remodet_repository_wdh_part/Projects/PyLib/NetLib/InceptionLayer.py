# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

# from_layer/out_layer
# cross_stage: 跨阶段连接
# channels_1: 1x1 conv
# channels_3: reduced(1x1) -> 3x3 conv
# channels_5: reduced(1x1) -> 3x3 conv -> 3x3 conv
# channels_pool: pool (maxpool-3/2) -> 1x1 conv
# use_out_conv: 拼接后是否使用一个1x1卷基层， channels_output -> 该输出卷积层的channels
# out_bn: 输出特征是否使用bn/scale/relu处理
# use_shortcut: 是否使用残差连接
# F -->[BN/SCALE/RELU]--|-> 1x1----------------->| C |---->[CONV]-->[BN/SCALE/RELU]---->[ E ]--> [bn]Output
# |    (use_prior_bn)   |-> reduced-3x3--------->| O | (use_out_conv)   (use_bn)        [ l ]
# |                     |-> reduced-3x3-3x3----->| N |                                  [ t ]
# |                     |-> maxpool-1x1--------->|Cat|                                  [ w ]
# |------------------------------------------------------------------------------------>[ise]
#                                    (use_shortcut)
#
# def InceptionLayer(net, from_layer, out_layer, use_prior_bn=True, cross_stage=False, channels_1=0, \
#                       channels_3=[], channels_5=[],channels_pool=0, use_out_conv=True, \
#                       channels_output=0, lr=1, decay=1, out_bn=True, use_shortcut=False):
#     assert len(channels_3) == 2
#     assert channels_3[0] > 0
#     assert channels_3[1] > 0
#     if channels_5:
#         assert len(channels_5) == 3
#         assert channels_5[0] > 0
#         assert channels_5[1] > 0
#         assert channels_5[2] > 0
#     if use_out_conv:
#         assert channels_output > 0
#     if out_bn:
#         assert use_out_conv
#     if use_shortcut:
#         assert use_out_conv
#     if cross_stage:
#         assert channels_pool > 0
#     assert channels_1 > 0

#     bn_kwargs = {
#         'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
#         'batch_norm_param': dict(use_global_stats=True),
#         }
#     scale_kwargs = {
#           'bias_term': True,
#           'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
#           }
#     input_kwargs = {'power': 1, 'scale': 1, 'shift': 0}
#     conv_kwargs = {
#         'param': [dict(lr_mult=lr, decay_mult=decay)],
#         'weight_filler': dict(type='xavier'),
#         'bias_term': False,
#         }
#     convbias_kwargs = {
#         'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
#         'weight_filler': dict(type='xavier'),
#         'bias_filler': dict(type='constant', value=0)
#         }
#     eltwise_kwargs = {'operation': 1, 'coeff': [1, 1]}
#     start_layer = from_layer
#     fea_layer = from_layer
#     if cross_stage:
#         stride = 2
#     else:
#         stride = 1
#     # pre-stage: bn/scale/relu
#     if use_prior_bn:
#         layer_name = "{}/incep/bn".format(out_layer)
#         name = "{}/incep/pre".format(out_layer)
#         net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=False, **bn_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/bn_scale".format(out_layer)
#         name = "{}/incep/bn_scale".format(out_layer)
#         net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/relu".format(out_layer)
#         name = "{}/incep/relu".format(out_layer)
#         net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#         fea_layer = name

#     mlayers = []
#     # conv-1x1
#     layer_name = "{}/incep/0/conv".format(out_layer)
#     name = "{}/incep/0".format(out_layer)
#     net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_1, \
#         kernel_size=1, pad=0, stride=stride, **conv_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/0/bn".format(out_layer)
#     name = "{}/incep/0/bn".format(out_layer)
#     net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/0/bn_scale".format(out_layer)
#     name = "{}/incep/0/bn_scale".format(out_layer)
#     net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/0/relu".format(out_layer)
#     name = "{}/incep/0/relu".format(out_layer)
#     net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#     mlayers.append(net[name])

#     # conv-3x3
#     layer_name = "{}/incep/1_reduce/conv".format(out_layer)
#     name = "{}/incep/1_reduce".format(out_layer)
#     net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_3[0], \
#         kernel_size=1, pad=0, stride=stride, **conv_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_reduce/bn".format(out_layer)
#     name = "{}/incep/1_reduce/bn".format(out_layer)
#     net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_reduce/bn_scale".format(out_layer)
#     name = "{}/incep/1_reduce/bn_scale".format(out_layer)
#     net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_reduce/relu".format(out_layer)
#     name = "{}/incep/1_reduce/relu".format(out_layer)
#     net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#     start_layer = name
#     layer_name = "{}/incep/1_0/conv".format(out_layer)
#     name = "{}/incep/1_0".format(out_layer)
#     net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_3[1], \
#         kernel_size=3, pad=1, stride=1, **conv_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_0/bn".format(out_layer)
#     name = "{}/incep/1_0/bn".format(out_layer)
#     net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_0/bn_scale".format(out_layer)
#     name = "{}/incep/1_0/bn_scale".format(out_layer)
#     net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#     start_layer = name
#     layer_name = "{}/incep/1_0/relu".format(out_layer)
#     name = "{}/incep/1_0/relu".format(out_layer)
#     net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#     mlayers.append(net[name])

#     # conv-5x5
#     if channels_5:
#         layer_name = "{}/incep/2_reduce/conv".format(out_layer)
#         name = "{}/incep/2_reduce".format(out_layer)
#         net[name] = L.Convolution(net[fea_layer], name=layer_name, num_output=channels_5[0], \
#             kernel_size=1, pad=0, stride=stride, **conv_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_reduce/bn".format(out_layer)
#         name = "{}/incep/2_reduce/bn".format(out_layer)
#         net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_reduce/bn_scale".format(out_layer)
#         name = "{}/incep/2_reduce/bn_scale".format(out_layer)
#         net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_reduce/relu".format(out_layer)
#         name = "{}/incep/2_reduce/relu".format(out_layer)
#         net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#         start_layer = name
#         layer_name = "{}/incep/2_0/conv".format(out_layer)
#         name = "{}/incep/2_0".format(out_layer)
#         net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_5[1], \
#             kernel_size=3, pad=1, stride=1, **conv_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_0/bn".format(out_layer)
#         name = "{}/incep/2_0/bn".format(out_layer)
#         net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_0/bn_scale".format(out_layer)
#         name = "{}/incep/2_0/bn_scale".format(out_layer)
#         net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_0/relu".format(out_layer)
#         name = "{}/incep/2_0/relu".format(out_layer)
#         net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#         start_layer = name
#         layer_name = "{}/incep/2_1/conv".format(out_layer)
#         name = "{}/incep/2_1".format(out_layer)
#         net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_5[2], \
#             kernel_size=3, pad=1, stride=1, **conv_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_1/bn".format(out_layer)
#         name = "{}/incep/2_1/bn".format(out_layer)
#         net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_1/bn_scale".format(out_layer)
#         name = "{}/incep/2_1/bn_scale".format(out_layer)
#         net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/2_1/relu".format(out_layer)
#         name = "{}/incep/2_1/relu".format(out_layer)
#         net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#         mlayers.append(net[name])

#     # pool
#     if cross_stage:
#         layer_name = "{}/incep/pool".format(out_layer)
#         name = "{}/incep/pool".format(out_layer)
#         net[name] = L.Pooling(net[fea_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2)
#         start_layer = name
#         layer_name = "{}/incep/poolproj/conv".format(out_layer)
#         name = "{}/incep/poolproj".format(out_layer)
#         net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_pool, \
#             kernel_size=1, pad=0, stride=1, **conv_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/poolproj/bn".format(out_layer)
#         name = "{}/incep/poolproj/bn".format(out_layer)
#         net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/poolproj/bn_scale".format(out_layer)
#         name = "{}/incep/poolproj/bn_scale".format(out_layer)
#         net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#         start_layer = name
#         layer_name = "{}/incep/poolproj/relu".format(out_layer)
#         name = "{}/incep/poolproj/relu".format(out_layer)
#         net[name] = L.ReLU(net[start_layer], name=layer_name, in_place=True)
#         mlayers.append(net[name])

#     # incep
#     layer_name = "{}/incep".format(out_layer)
#     name = "{}/incep".format(out_layer)
#     net[name] = L.Concat(*mlayers, name=layer_name, axis=1)
#     start_layer = name
#     incep_layer = name

#     # out-conv
#     scLayers = []
#     if use_out_conv:
#         if not out_bn:
#             layer_name = "{}/out/conv".format(out_layer)
#             name = "{}/out".format(out_layer)
#             net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_output, \
#                 kernel_size=1, pad=0, stride=1, **convbias_kwargs)
#             scLayers.append(net[name])
#         else:
#             layer_name = "{}/out/conv".format(out_layer)
#             name = "{}/out".format(out_layer)
#             net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_output, \
#                 kernel_size=1, pad=0, stride=1, **conv_kwargs)
#             start_layer = name
#             layer_name = "{}/out/bn".format(out_layer)
#             name = "{}/out/bn".format(out_layer)
#             net[name] = L.BatchNorm(net[start_layer], name=layer_name, in_place=True, **bn_kwargs)
#             start_layer = name
#             layer_name = "{}/out/bn_scale".format(out_layer)
#             name = "{}/out/bn_scale".format(out_layer)
#             net[name] = L.Scale(net[start_layer], name=layer_name, in_place=True, **scale_kwargs)
#             scLayers.append(net[name])

#     # proj or input
#     if use_shortcut:
#         if cross_stage:
#             layer_name = "{}/proj".format(out_layer)
#             name = "{}/proj".format(out_layer)
#             net[name] = L.Convolution(net[from_layer], name=layer_name, num_output=channels_output, \
#                 kernel_size=1, pad=0, stride=2, **convbias_kwargs)
#             scLayers.append(net[name])
#         else:
#             layer_name = "{}/input".format(out_layer)
#             name = "{}/input".format(out_layer)
#             net[name] = L.Power(net[from_layer], name=layer_name, **input_kwargs)
#             scLayers.append(net[name])

#     # Eltwise
#     if len(scLayers) > 1:
#         layer_name = out_layer
#         name = out_layer
#         net[name] = L.Eltwise(*scLayers, name=layer_name, **eltwise_kwargs)
#     elif len(scLayers) == 1:
#         net[out_layer] = scLayers[0]
#     else:
#         net[out_layer] = net[incep_layer]

#     return net
def InceptionLayer(net, from_layer, out_layer, use_prior_bn=True, cross_stage=False, channels_1=0, \
                      channels_3=[], channels_5=[],channels_pool=0, use_out_conv=True, \
                      channels_output=0, lr=1, decay=1, out_bn=True, use_shortcut=False,stride=1):
    assert len(channels_3) == 2
    assert channels_3[0] > 0
    assert channels_3[1] > 0
    if channels_5:
        assert len(channels_5) == 3
        assert channels_5[0] > 0
        assert channels_5[1] > 0
        assert channels_5[2] > 0
    if use_out_conv:
        assert channels_output > 0
    if out_bn:
        assert use_out_conv
    if use_shortcut:
        assert use_out_conv
    if cross_stage:
        assert channels_pool > 0
    assert channels_1 > 0

    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'batch_norm_param': dict(use_global_stats=False),
        }
    scale_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=lr, decay_mult=0), dict(lr_mult=lr, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
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
    fea_layer = from_layer
    # pre-stage: bn/scale/relu
    if use_prior_bn:
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
    if channels_5:
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
        net[name] = L.Pooling(net[fea_layer], pool=P.Pooling.MAX, kernel_size=3, stride=stride,pad=1)
        start_layer = name
        layer_name = "{}/incep/poolproj/conv".format(out_layer)
        name = "{}/incep/poolproj".format(out_layer)
        net[name] = L.Convolution(net[start_layer], name=layer_name, num_output=channels_pool, \
            kernel_size=1, pad=0, stride=stride, **conv_kwargs)
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
    layer_name = "{}".format(out_layer)
    name = "{}".format(out_layer)
    net[name] = L.Concat(*mlayers, name=layer_name, axis=1)
    start_layer = name
    incep_layer = name

    # out-conv
    scLayers = []
    if use_out_conv:
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
    if use_shortcut:
        if cross_stage:
            layer_name = "{}/proj".format(out_layer)
            name = "{}/proj".format(out_layer)
            net[name] = L.Convolution(net[from_layer], name=layer_name, num_output=channels_output, \
                kernel_size=1, pad=0, stride=stride, **convbias_kwargs)
            scLayers.append(net[name])
        else:
            layer_name = "{}/input".format(out_layer)
            name = "{}/input".format(out_layer)
            net[name] = L.Power(net[from_layer], name=layer_name, **input_kwargs)
            scLayers.append(net[name])

    # Eltwise
    if len(scLayers) > 1:
        layer_name = out_layer
        name = out_layer
        net[name] = L.Eltwise(*scLayers, name=layer_name, **eltwise_kwargs)
    elif len(scLayers) == 1:
        net[out_layer] = scLayers[0]
    else:
        net[out_layer] = net[incep_layer]

    return net
