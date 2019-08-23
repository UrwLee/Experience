 # -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.ConvBNLayer import *

from DeconvLayer import *
from InceptionReduceLayer import *
from mPoseBaseNet import *

def mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1,
                      use_3_layers=5, use_1_layers=2, short_cut=True, base_layer="convf", lr=0, decay=0, num_channels = 128,
                      flag_output = True, addstrs = '',kernel_size=3):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer) + addstrs
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer) + addstrs
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        if flag_output:
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        if flag_output:
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_StageXDepthwise_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
               use_3_layers=5, use_1_layers=2, short_cut=True, base_layer="convf", lr=1, decay=1, num_channels = 128, kernel_size=3, group_divide=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        if layer == 1:
            conv_vec = "stage{}_conv{}_vec".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=1, kernel_size=3,
                                          engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=1, kernel_size=3,
                                          engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec
        else:
            n_group = num_channels / group_divide
            conv_vec = "stage{}_conv{}_vec_dw".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size - 1) / 2,
                                          kernel_size=kernel_size, group=n_group, engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec

            conv_vec = "stage{}_conv{}_vec".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,
                                          engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage, layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size - 1) / 2,
                                           kernel_size=kernel_size, group=n_group, engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage, layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat

            conv_vec = "stage{}_conv{}_heat".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,
                                          engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
def mPose_StageX_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True,base_layer="convf", lr=1.0, decay=1.0,num_channels = 128,flag_sigmoid = False,
                       kernel_size=3,addstrs = '',flag_change_layer=False,flag_hasoutput=True,flag_hasloss=True,id_layer_until=0, relu_layer_until = False):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        if layer == numlayers - 1 and flag_change_layer:
            num_channels = 64
        conv_vec = "stage{}_conv{}_vec".format(stage,layer) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)

        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)

        if layer == id_layer_until:
            if relu_layer_until:
                relu_vec = "stage{}_relu{}_vec".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                relu_heat = "stage{}_relu{}_heat".format(stage,layer)
                net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
                return net
            else:
                return net
        else:
            relu_vec = "stage{}_relu{}_vec".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            relu_heat = "stage{}_relu{}_heat".format(stage, layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
    if flag_hasoutput:
        if use_1_layers > 0:
            for layer in range(1, use_1_layers):
                # vec
                conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer) + addstrs
                net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
                relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer) + addstrs
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from1_layer = relu_vec
                # heat
                conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer) + addstrs
                net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
                relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer) + addstrs
                net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
                from2_layer = relu_heat
            # output
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
        else:
            # output by 3x3
            if flag_change_layer:
                kernel_size = 3
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
            if flag_sigmoid:
                conv_vec_sig = conv_vec + "_sig"
                net[conv_vec_sig] = L.Sigmoid(net[conv_vec])
                conv_vec = conv_vec_sig
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
            if flag_sigmoid:
                conv_heat_sig = conv_heat + "_sig"
                net[conv_heat_sig] = L.Sigmoid(net[conv_heat])
                conv_heat = conv_heat_sig
        if flag_hasloss:
            weight_vec = "weight_stage{}_vec".format(stage)
            weight_heat = "weight_stage{}_heat".format(stage)
            loss_vec = "loss_stage{}_vec".format(stage)
            loss_heat = "loss_stage{}_heat".format(stage)
            net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
            net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=0.2)
            net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
            net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=0.2)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_StageXShuffle_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers = 0,short_cut=True,base_layer="convf", lr=1.0, decay=1.0, num_channels = [128,],
                              n_group = 8, addstrs = '',flag_hasoutput=True,flag_has_loss=True,deploy=False,flag_lossscale=False,
                              layer_insert_1layers=0):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    layer_base_number = 0
    cnt_channles = 0

    for layer in range(1, use_3_layers):
        num_ch_out = num_channels[cnt_channles]
        # vec 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,layer + layer_base_number) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_ch_out, pad=1, kernel_size=3,group=n_group, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer + layer_base_number) + addstrs
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        if layer != use_3_layers - 1 and not deploy:
            relu_vec_shuffle = "stage{}_relu{}_vec".format(stage,layer) + addstrs + '_shf'
            net[relu_vec_shuffle] = L.ShuffleChannel(net[relu_vec], shuffle_channel_param=dict(group=n_group))
            from1_layer = relu_vec_shuffle
        else:
            from1_layer = relu_vec
        # heat 3x3
        conv_heat = "stage{}_conv{}_heat".format(stage,layer + layer_base_number) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_ch_out, pad=1, kernel_size=3,group=n_group, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer + layer_base_number) + addstrs
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        if layer != use_3_layers - 1 and not deploy:
            relu_heat_shuffle = "stage{}_relu{}_heat".format(stage,layer) + addstrs + '_shf'
            net[relu_heat_shuffle] = L.ShuffleChannel(net[relu_heat], shuffle_channel_param=dict(group=n_group))
            from2_layer = relu_heat_shuffle
        else:
            from2_layer = relu_heat
        cnt_channles += 1
        if layer == layer_insert_1layers:
            num_ch_out = num_channels[cnt_channles]
            for layer_1 in xrange(use_1_layers):
                # vec 1x1
                conv_vec = "stage{}_conv{}_vec".format(stage, layer + layer_1 + 1) + addstrs
                net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_ch_out, pad=0, kernel_size=1, group=1,
                                              **kwargs)
                relu_vec = "stage{}_relu{}_vec".format(stage, layer + layer_1 + 1) + addstrs
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from1_layer = relu_vec
            for layer_1 in xrange(use_1_layers):
                # heat 1x1
                conv_heat = "stage{}_conv{}_heat".format(stage, layer + layer_1 + 1) + addstrs
                net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_ch_out, pad=0, kernel_size=1, group=1,
                                               **kwargs)
                relu_heat = "stage{}_relu{}_heat".format(stage, layer + layer_1 + 1) + addstrs
                net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
                from2_layer = relu_heat
            layer_base_number = use_1_layers
            cnt_channles += 1
    if layer_insert_1layers == 0:
        num_ch_out = num_channels[cnt_channles]
        for layer in xrange(use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage, use_3_layers + layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_ch_out, pad=0, kernel_size=1, group=1,
                                          **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, use_3_layers + layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage, use_3_layers + layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_ch_out, pad=0, kernel_size=1, group=1,
                                           **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage, use_3_layers + layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat

    if flag_hasoutput:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers + use_1_layers) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers + use_1_layers) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)
        if flag_has_loss:
            weight_vec = "weight_stage{}_vec".format(stage)
            weight_heat = "weight_stage{}_heat".format(stage)
            loss_vec = "loss_stage{}_vec".format(stage)
            loss_heat = "loss_stage{}_heat".format(stage)
            if flag_lossscale:
                loss_weight = 1.0/34.0
            else:
                loss_weight = 1.0
            net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
            net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=loss_weight)
            net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
            if flag_lossscale:
                loss_weight = 1.0/18.0
            else:
                loss_weight = 1.0
            net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=loss_weight)
            # 特征拼接
            if short_cut:
                fea_layers = []
                fea_layers.append(net[conv_vec])
                fea_layers.append(net[conv_heat])
                assert base_layer in net.keys()
                fea_layers.append(net[base_layer])
                net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
def mPose_StageXDepthwise_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1,mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1.0, decay=1.0, kernel_size=3,num_channels = 64,group_divide=1, flag_layer1DW = False, nchan_input = 0):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):

        if layer == 1:
            if not flag_layer1DW:
                # vec
                conv_vec = "stage{}_conv{}_vec".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
                relu_vec = "stage{}_relu{}_vec".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from1_layer = relu_vec
                # heat
                conv_vec = "stage{}_conv{}_heat".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
                relu_vec = "stage{}_relu{}_heat".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from2_layer = relu_vec
            else:
                # vec
                conv_vec = "stage{}_conv{}_vec_dw".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from1_layer], num_output=nchan_input, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=nchan_input,
                                              engine=P.Convolution.CAFFE, **kwargs)
                relu_vec = "stage{}_relu{}_vec_dw".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from1_layer = relu_vec
                conv_vec = "stage{}_conv{}_vec".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
                relu_vec = "stage{}_relu{}_vec".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from1_layer = relu_vec

                # heat
                conv_vec = "stage{}_conv{}_heat_dw".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from2_layer], num_output=nchan_input, pad=(kernel_size-1)/2, kernel_size=kernel_size, group=nchan_input,
                                              engine=P.Convolution.CAFFE, **kwargs)
                relu_vec = "stage{}_relu{}_heat_dw".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from2_layer = relu_vec
                conv_vec = "stage{}_conv{}_heat".format(stage, layer)
                net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1, engine=P.Convolution.CAFFE, **kwargs)
                relu_vec = "stage{}_relu{}_heat".format(stage, layer)
                net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
                from2_layer = relu_vec

        else:
            n_group = num_channels/group_divide
            # vec
            conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec

            conv_vec = "stage{}_conv{}_vec".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat

            conv_vec = "stage{}_conv{}_heat".format(stage, layer)
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,engine=P.Convolution.CAFFE, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
########### No loss
def mPose_StageXDepthwiseNoLoss_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, use_3_layers=5, use_1_layers=2, lr=1.0, decay=1.0, kernel_size=3,
                                      num_channels = 64,group_divide=1,addstrs='',flag_hasoutput = False,mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label",
                                      label_heat="heat_label",base_layer="convf",flag_hasloss=True,short_cut= False):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):

        if layer == 1:

            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

        else:
            n_group = num_channels/group_divide
            # vec
            conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
            # vec
            if layer == numlayers-1:
                num_channels = 64
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

    if flag_hasoutput:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage, numlayers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,
                                      engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage, numlayers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,
                                           engine=P.Convolution.CAFFE, **kwargs)
        if flag_hasloss:
            weight_vec = "weight_stage{}_vec".format(stage)
            weight_heat = "weight_stage{}_heat".format(stage)
            loss_vec = "loss_stage{}_vec".format(stage)
            loss_heat = "loss_stage{}_heat".format(stage)
            net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
            net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
            net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
            net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
########### No loss
def mPose_StageXDepthwiseNoLoss_Test(net, from_layer="concat_stage1", out_layer="concat_stage2",stage=1, use_3_layers=5, use_1_layers=2, lr=1.0, decay=1.0, kernel_size=3,
                                      num_channels = 64,group_divide=1,addstrs='',flag_hasoutput = False,base_layer="convf",short_cut= False):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):

        if layer == 1:
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

        else:
            n_group = num_channels/group_divide
            # vec
            conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
            # vec
            if layer == numlayers-1:
                num_channels = 64
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

    if flag_hasoutput:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage, numlayers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,
                                      engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage, numlayers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,
                                           engine=P.Convolution.CAFFE, **kwargs)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

########### Replace the first layer after concatenate
def mPose_StageXDepthwiseNoLossA_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, use_3_layers=5, use_1_layers=2, lr=1.0, decay=1.0, kernel_size=3,
                                      num_channels = 64,group_divide=1,addstrs='',flag_hasoutput = False,mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label",
                                      label_heat="heat_label",base_layer="convf",short_cut= False):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):

        if layer == 1:

            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=64, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=64, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

        else:
            n_group = num_channels/group_divide
            # vec
            conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
            # vec
            if layer == numlayers-1:
                num_channels = 64
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

    if flag_hasoutput:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage, numlayers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,
                                      engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage, numlayers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,
                                           engine=P.Convolution.CAFFE, **kwargs)
        weight_vec = "weight_stage{}_vec".format(stage)
        weight_heat = "weight_stage{}_heat".format(stage)
        loss_vec = "loss_stage{}_vec".format(stage)
        loss_heat = "loss_stage{}_heat".format(stage)
        net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
        net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

########### Replace the first layer after concatenate
def mPose_StageXDepthwiseNoLossA_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, use_3_layers=5, use_1_layers=2, lr=1.0, decay=1.0, kernel_size=3,
                                      num_channels = 64,group_divide=1,addstrs='',flag_hasoutput = False,base_layer="convf",short_cut= False):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):

        if layer == 1:

            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=64, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=64, pad=1, kernel_size=3,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

        else:
            n_group = num_channels/group_divide
            # vec
            conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
            relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
            # vec
            if layer == numlayers-1:
                num_channels = 64
            conv_vec = "stage{}_conv{}_vec".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_vec = "stage{}_conv{}_heat".format(stage, layer) + addstrs
            net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
            relu_vec = "stage{}_relu{}_heat".format(stage, layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from2_layer = relu_vec

    if flag_hasoutput:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage, numlayers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,
                                      engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage, numlayers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,
                                           engine=P.Convolution.CAFFE, **kwargs)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
#### THIS LAYER HAS PROBLEMS!!!!!!!!!!!!!!!!!!!!
def mPose_StageXDepthwise_TrainA(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1.0, decay=1.0, kernel_size=3,num_channels = 64,group_divide=1):

    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    for layer in range(1, numlayers):
        # vec
        if layer == 1:
            if stage==1:
                n_group = 384 / group_divide
            else:
                n_group = 438 / group_divide
        else:
            n_group = num_channels/group_divide
        conv_vec = "stage{}_conv{}_vec_dw".format(stage,layer)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
        relu_vec = "stage{}_relu{}_vec_dw".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec

        conv_vec = "stage{}_conv{}_vec".format(stage, layer)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, group=1,engine=P.Convolution.CAFFE, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage, layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat_dw".format(stage,layer)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size,group=n_group,engine=P.Convolution.CAFFE, **kwargs)
        relu_heat = "stage{}_relu{}_heat_dw".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat

        conv_vec = "stage{}_conv{}_heat".format(stage, layer)
        net[conv_vec] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, group=1,engine=P.Convolution.CAFFE,**kwargs)
        relu_vec = "stage{}_relu{}_heat".format(stage, layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from2_layer = relu_vec
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3,engine=P.Convolution.CAFFE, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3,engine=P.Convolution.CAFFE, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
############################################################
############################################################
def mPose_StageXResid_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1, decay=1, num_channels = 128):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=1, kernel_size=3, **kwargs)
        resid_vec = "stage{}_conv{}_vec_resid".format(stage,layer)
        net[resid_vec] = L.Eltwise(net[from_layer], net[conv_vec],
                                   eltwise_param=dict(operation=P.Eltwise.SUM))
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=1, kernel_size=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
#############################################################
## using inception
############################################################
def mPose_StageXInceptionA_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers

    layer = 1
    # vec
    conv_vec = "stage{}_conv{}_vec".format(stage,layer)
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    relu_vec = "stage{}_relu{}_vec".format(stage,layer)
    net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
    from1_layer = relu_vec
    # heat
    conv_heat = "stage{}_conv{}_heat".format(stage,layer)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    relu_heat = "stage{}_relu{}_heat".format(stage,layer)
    net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
    from2_layer = relu_heat

    # vec inception one
    layer = 1
    mid_layer = "stage{}_inception{}_vec".format(stage,layer)
    InceptionReduceLayer(net, from1_layer, mid_layer, channels_3=[64,128], channels_5=[32,64,64],
                         use_out_conv=False, channels_output=0, out_bn=False, inter_bn=True)
    from1_layer = mid_layer
    # heat inception one
    mid_layer = "stage{}_inception{}_heat".format(stage, layer)
    InceptionReduceLayer(net, from2_layer, mid_layer, channels_3=[64, 128], channels_5=[32, 64, 64],
                         use_out_conv=False, channels_output=0, out_bn=False, inter_bn=True)
    from2_layer = mid_layer

    # vec inception two
    layer = 2
    mid_layer = "stage{}_inception{}_vec".format(stage, layer)
    InceptionReduceLayer(net, from1_layer, mid_layer, channels_3=[64, 128], channels_5=[32, 64, 64],
                         use_out_conv=True, channels_output=128, out_bn=True, inter_bn=True)
    from1_layer = mid_layer
    # heat inception one
    mid_layer = "stage{}_inception{}_heat".format(stage, layer)
    InceptionReduceLayer(net, from2_layer, mid_layer, channels_3=[64, 128], channels_5=[32, 64, 64],
                         use_out_conv=True, channels_output=128, out_bn=True, inter_bn=True)
    from2_layer = mid_layer


    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
##############################################################
# Directly compress the StageX:
# Replace one 3x3 conv layer with one 3x1 layer and 1x3 layer.
#####################################################################
def mPose_StageXCompress_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
               use_3_layers=5, use_1_layers=2, short_cut=True, base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    num_channels = 192
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec_w = "stage{}_conv{}_vec_w".format(stage, layer)
        net[conv_vec_w] = L.Convolution(net[from1_layer], num_output=num_channels, pad_w=1,kernel_w=3,kernel_h=1, **kwargs)
        conv_vec = "stage{}_conv{}_vec".format(stage, layer)
        net[conv_vec] = L.Convolution(net[conv_vec_w], num_output=num_channels, pad_h=1, kernel_w=1,kernel_h=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage, layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat_w = "stage{}_conv{}_heat_w".format(stage, layer)
        net[conv_heat_w] = L.Convolution(net[from2_layer], num_output=num_channels, pad_w=1, kernel_w=3,kernel_h=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage, layer)
        net[conv_heat] = L.Convolution(net[conv_heat_w], num_output=num_channels, pad_h=1, kernel_w=1,kernel_h=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage, layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
##############################################################
# Directly compress the StageX:
# Replace one 3x3 conv layer with one 3x1 layer and 1x3 layer.
#####################################################################
def mPose_StageXCompress_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    num_channels = 192
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec_w = "stage{}_conv{}_vec_w".format(stage,layer)
        net[conv_vec_w] = L.Convolution(net[from1_layer], num_output=num_channels, pad_w=1, kernel_w=3,kernel_h=1, **kwargs)
        conv_vec = "stage{}_conv{}_vec".format(stage, layer)
        net[conv_vec] = L.Convolution(net[conv_vec_w], num_output=num_channels, pad_h=1, kernel_w=1, kernel_h=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat_w = "stage{}_conv{}_heat_w".format(stage, layer)
        net[conv_heat_w] = L.Convolution(net[from2_layer], num_output=num_channels, pad_w=1, kernel_w=3,kernel_h=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
        net[conv_heat] = L.Convolution(net[conv_heat_w], num_output=num_channels, pad_h=1, kernel_w=1,kernel_h=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=1, kernel_size=3, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

##############################################################
# Replace the StageX:
# Replace one 3x3 conv layer with 64 1x1 conv and 128 3x3.
#####################################################################
def mPose_StageXReplace_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
               use_3_layers=5, use_1_layers=2, short_cut=True, base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    num_channels = 192
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec_1 = "stage{}_conv{}_vec_1".format(stage, layer)
        net[conv_vec_1] = L.Convolution(net[from1_layer], num_output=64, kernel_size=1, **kwargs)
        relu_vec_1 = "stage{}_relu{}_vec_1".format(stage, layer)
        net[relu_vec_1] = L.ReLU(net[conv_vec_1], in_place=True)
        conv_vec = "stage{}_conv{}_vec".format(stage, layer)
        net[conv_vec] = L.Convolution(net[relu_vec_1], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage, layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat_1 = "stage{}_conv{}_heat_1".format(stage, layer)
        net[conv_heat_1] = L.Convolution(net[from2_layer], num_output=64, kernel_size=1, **kwargs)
        relu_heat_1 = "stage{}_relu{}_heat_1".format(stage, layer)
        net[relu_heat_1] = L.ReLU(net[conv_heat_1], in_place=True)
        conv_heat = "stage{}_conv{}_heat".format(stage, layer)
        net[conv_heat] = L.Convolution(net[relu_heat_1], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage, layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=1, kernel_size=3, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
##############################################################
# Replace the StageX:
# Replace one 3x3 conv layer with 64 1x1 conv and 128 3x3.
#####################################################################
def mPose_StageXReplace_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True, \
                       base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    for layer in range(1, numlayers):
        # vec
        conv_vec_1 = "stage{}_conv{}_vec_1".format(stage,layer)
        net[conv_vec_1] = L.Convolution(net[from1_layer], num_output=64, kernel_size=1, **kwargs)
        relu_vec_1 = "stage{}_relu{}_vec_1".format(stage, layer)
        net[relu_vec_1] = L.ReLU(net[conv_vec_1], in_place=True)
        conv_vec = "stage{}_conv{}_vec".format(stage, layer)
        net[conv_vec] = L.Convolution(net[relu_vec_1], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat_1 = "stage{}_conv{}_heat_1".format(stage, layer)
        net[conv_heat_1] = L.Convolution(net[from2_layer], num_output=64, kernel_size=1, **kwargs)
        relu_heat_1 = "stage{}_relu{}_heat_1".format(stage, layer)
        net[relu_heat_1] = L.ReLU(net[conv_heat_1], in_place=True)
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
        net[conv_heat] = L.Convolution(net[conv_heat_1], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=1, kernel_size=3, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=1, kernel_size=3, **kwargs)
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
def mPose_Stage_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
               use_layers=5, short_cut=True, base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    for layer in range(1, use_layers+1):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    # 1x1 layers
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers+1)
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_vec = "stage{}_relu{}_vec".format(stage,use_layers+1)
    net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
    from1_layer = relu_vec
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+1)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_heat = "stage{}_relu{}_heat".format(stage,use_layers+1)
    net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
    from2_layer = relu_heat
    # output
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers+2)
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=0, kernel_size=1, **kwargs)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+2)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=0, kernel_size=1, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_Stage_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                      mask_vec="vec_mask", mask_heat="heat_mask", \
                      label_vec="vec_label", label_heat="heat_label", \
                      use_layers=5, short_cut=True, base_layer="convf", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    for layer in range(1, use_layers+1):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    # 1x1 layers
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers+1)
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_vec = "stage{}_relu{}_vec".format(stage,use_layers+1)
    net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
    from1_layer = relu_vec
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+1)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_heat = "stage{}_relu{}_heat".format(stage,use_layers+1)
    net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
    from2_layer = relu_heat
    # output
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers+2)
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=38, pad=0, kernel_size=1, **kwargs)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+2)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=19, pad=0, kernel_size=1, **kwargs)
    # loss
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_Stage_Train_BN(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                      mask_vec="vec_mask", mask_heat="heat_mask", \
                      label_vec="vec_label", label_heat="heat_label", \
                      use_layers=5, short_cut=True, base_layer="convf", lr=1, decay=1):
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    for layer in range(1, use_layers):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer)
    	ConvBNUnitLayer(net, from1_layer, conv_vec, use_bn=True, use_relu=True, \
    		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    	from1_layer = conv_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
    	ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=True, use_relu=True, \
    		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    	from2_layer = conv_heat
    # Last for vec & heat
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers)
    ConvBNUnitLayer(net, from1_layer, conv_vec, use_bn=False, use_relu=False, \
		num_output=38, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
    ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=False, use_relu=False, \
		num_output=19, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    # loss
    weight_vec = "weight_stage{}_vec".format(stage)
    weight_heat = "weight_stage{}_heat".format(stage)
    loss_vec = "loss_stage{}_vec".format(stage)
    loss_heat = "loss_stage{}_heat".format(stage)
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_Stage_Test_BN(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                      use_layers=5, short_cut=True, base_layer="convf", lr=1, decay=1):
    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    for layer in range(1, use_layers):
        # vec
        conv_vec = "stage{}_conv{}_vec".format(stage,layer)
    	ConvBNUnitLayer(net, from1_layer, conv_vec, use_bn=True, use_relu=True, \
    		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    	from1_layer = conv_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer)
    	ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=True, use_relu=True, \
    		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    	from2_layer = conv_heat
    # Last for vec & heat
    conv_vec = "stage{}_conv{}_vec".format(stage,use_layers)
    ConvBNUnitLayer(net, from1_layer, conv_vec, use_bn=False, use_relu=False, \
		num_output=38, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
    ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=False, use_relu=False, \
		num_output=19, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPoseNet_COCO_3S_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    strid_convs = [1,1,1,0,0]
    kernel_sizes = [3,3,3,3,5]
    leaky = False
    # net = YoloNetPartCompressDepthwiseE(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=4,strid_conv=strid_convs,
    #                                    kernel_sizes=kernel_sizes,final_pool=False,group_divide=1,lr=0, decay=0,addstrs='_recon')
    # print net.keys()
    # net = YoloNetPartCompressDepthwisePartial(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=4,
    #                                    strid_conv=strid_convs,kernel_sizes=kernel_sizes, final_pool=False, group_divide=1, lr=1, decay=1,addstrs='_recon')

    net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,
                                       strid_conv=strid_convs, final_pool=False,lr=1, decay=1,leaky=leaky)

    # concat conv4_3 & conv5_5
    baselayer = "convf"
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3","conv5_5"], tags=["Ref","Up"],unifiedlayer=baselayer, upsampleMethod="Reorg")
    # Stages
    #
    # net['convf_drop'] = L.SpatialDropout(net[baselayer], in_place=True,dropout_param=dict(dropout_ratio=0.2))
    # baselayer = 'convf_drop'
    use_stage = 3
    # use_3_layers = 7
    # use_1_layers = 0
    # n_channel = 128
    # lrdecay = 1.0
    # group_divide = 1
    # net = mPose_StageXDepthwiseNoLossA_Train(net, from_layer=baselayer, out_layer="concat_stage1",stage=1, use_3_layers=use_3_layers, use_1_layers=use_1_layers, lr=lrdecay,decay=lrdecay,
    #                                         kernel_size=7,num_channels=n_channel, group_divide=group_divide,flag_hasoutput = True,mask_vec="vec_mask",mask_heat="heat_mask",
    #                                         label_vec="vec_label",label_heat="heat_label",base_layer = baselayer,short_cut=True,addstrs='_recon')
    # net = mPose_StageXDepthwiseNoLoss_Train(net, from_layer=baselayer, out_layer="concat_stage1", stage=1,
    #                                          use_3_layers=use_3_layers, use_1_layers=use_1_layers, lr=lrdecay,
    #                                          decay=lrdecay,kernel_size=7, num_channels=n_channel, group_divide=group_divide,
    #                                          flag_hasoutput=True, mask_vec="vec_mask", mask_heat="heat_mask",
    #                                          label_vec="vec_label", label_heat="heat_label", base_layer=baselayer,
    #                                          short_cut=True, addstrs='_recon',flag_hasloss=True)

    # net = mPose_StageXDepthwise_Train(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, mask_vec="vec_mask", mask_heat="heat_mask",
    #                                   label_vec="vec_label",label_heat="heat_label",use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False,
    #                                   base_layer=baselayer,lr=lrdecay, decay=lrdecay, kernel_size=9,num_channels = n_channel,group_divide=1,flag_layer1DW = True,
    #                                   nchan_input=384)
    # net = mPose_StageXDepthwise_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2,mask_vec="vec_mask", mask_heat="heat_mask",
    #                                   label_vec="vec_label",label_heat="heat_label", use_3_layers=use_3_layers, use_1_layers=use_1_layers,short_cut=True,
    #                                    base_layer=baselayer,lr=lrdecay, decay=lrdecay, kernel_size=7, num_channels=n_channel, group_divide=1)
    # net = mPose_StageXDepthwise_TrainA(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3,mask_vec="vec_mask", mask_heat="heat_mask",
    #                                   label_vec="vec_label", label_heat="heat_label", use_3_layers=use_3_layers,use_1_layers=use_1_layers, short_cut=False,
    #                                   base_layer=baselayer, lr=lrdecay, decay=lrdecay, kernel_size=7, num_channels=n_channel, group_divide=1)

    # n_channel = 64
    # group_divide = 8
    # n_group = 8
    # use_3_layers = 7
    # use_1_layers = 0
    # addstrs = '_recon'
    # net = mPose_StageXShuffle_Train(net, from_layer=baselayer, stage=1, use_3_layers=use_3_layers, use_1_layers=use_1_layers,
    #                                 out_layer="concat_stage1", mask_vec="vec_mask", mask_heat="heat_mask",short_cut=True,
    #                                 base_layer=baselayer, lr=1.0, decay=1.0, num_channels=n_channel, n_group=n_group,
    #                                 addstrs=addstrs,flag_hasoutput=True, flag_has_loss=True)
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    net = mPose_StageX_Train(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel)
    net = mPose_StageX_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel)
    net = mPose_StageX_Train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False, \
                           lr=lrdecay, decay=lrdecay, num_channels=n_channel)
    # for Test
    if not train:
        print(net.keys())
        conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers)
        conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_3S_Test(net, from_layer="data", frame_layer="orig_data", **pose_kwargs):
    # Darknet19

    kernel_sizes = [3, 3, 3, 3, 5]
    strid_convs = [1, 1, 1, 0, 0]
    net = YoloNetPartCompressDepthwiseE(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=4,
                                       strid_conv=strid_convs, kernel_sizes=kernel_sizes, final_pool=False, group_divide=1, lr=1, decay=1,addstrs='_recon')
    # net = YoloNetPartCompress(net, from_layer=from_layer, use_bn=True, use_layers=5, use_sub_layers=5, strid_conv=[1,1,1,0,0],final_pool=False, lr=1, decay=1)
    # concat conv4_3 & conv5_5
    # net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False, lr=1, decay=1)
    # concat conv4_3 & conv5_5
    baselayer = "convf"
    net = UnifiedMultiScaleLayers(net, layers=["conv4_4_recon","conv5_8_recon"], tags=["Ref","Up"], \
                                  unifiedlayer=baselayer, upsampleMethod="Reorg")
    # Stages
    use_stage = 1
    use_3_layers = 7
    use_1_layers = 0
    n_channel = 128 # for NoLoss
    # n_channel = 64 # for NoLossA
    # lrdecay = 1.0
    group_divide = 1 # for NoLoss
    # group_divide = 16# 4,8, 16  for NoLossA
    # net = mPose_StageXDepthwise_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1,use_3_layers=use_3_layers, use_1_layers=use_1_layers,
    #                                  short_cut=True,base_layer=baselayer,num_channels=n_channel,kernel_size=7)
    # net = mPose_StageXDepthwise_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2,use_3_layers=use_3_layers, use_1_layers=use_1_layers,
    #                                  short_cut=True, base_layer=baselayer, num_channels=n_channel, kernel_size=7)
    # net = mPose_StageXDepthwise_Test(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3,
    #                                  use_3_layers=use_3_layers, use_1_layers=use_1_layers,
    #                                  short_cut=False, base_layer=baselayer, num_channels=n_channel, kernel_size=7)
    net = mPose_StageXDepthwiseNoLoss_Test(net, from_layer=baselayer, out_layer="concat_stage1",stage=1, use_3_layers=use_3_layers,
                                            use_1_layers=use_1_layers, lr=0, decay=0,
                                            kernel_size=7, num_channels=n_channel, group_divide=group_divide, addstrs='_recon',
                                            flag_hasoutput=True,short_cut=False)
    addstrs = ''
    # use_3_layers = 5
    # use_1_layers = 0
    # n_channel = 64
    # lrdecay = 4.0
    # net = mPose_StageX_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
    #                        use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False, \
    #                        base_layer=baselayer,num_channels=n_channel)
    # net = mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
    #                        use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
    #                        base_layer=baselayer,num_channels=n_channel)
    # net = mPose_StageX_Test(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
    #                        use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False,num_channels=n_channel)

    conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers)
    conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers)



    feaLayers = []
    feaLayers.append(net[conv_heat])
    feaLayers.append(net[conv_vec])
    outlayer = "concat_stage{}".format(use_stage)
    net[outlayer] = L.Concat(*feaLayers, axis=1)
    # Resize
    resize_kwargs = {
        'factor': pose_kwargs.get("resize_factor", 8),
        'scale_gap': pose_kwargs.get("resize_scale_gap", 0.3),
        'start_scale': pose_kwargs.get("resize_start_scale", 1.0),
    }
    net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
    # Nms
    nms_kwargs = {
        'threshold': pose_kwargs.get("nms_threshold", 0.05),
        'max_peaks': pose_kwargs.get("nms_max_peaks", 64),
        'num_parts': pose_kwargs.get("nms_num_parts", 18),
    }
    net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
    # ConnectLimbs
    connect_kwargs = {
        'is_type_coco': pose_kwargs.get("conn_is_type_coco", True),
        'max_person': pose_kwargs.get("conn_max_person", 20),
        'max_peaks_use': pose_kwargs.get("conn_max_peaks_use", 32),
        'iters_pa_cal': pose_kwargs.get("conn_iters_pa_cal", 10),
        'connect_inter_threshold': pose_kwargs.get("conn_connect_inter_threshold", 0.05),
        'connect_inter_min_nums': pose_kwargs.get("conn_connect_inter_min_nums", 8),
        'connect_min_subset_cnt': pose_kwargs.get("conn_connect_min_subset_cnt", 3),
        'connect_min_subset_score': pose_kwargs.get("conn_connect_min_subset_score", 0.3),
    }
    net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
    # VisualizePose
    visual_kwargs = {
        'is_type_coco': pose_kwargs.get("conn_is_type_coco", True),
        'type': pose_kwargs.get("visual_type", P.Visualizepose.VECMAP_ID),
        'visualize': pose_kwargs.get("visual_visualize", True),
        'draw_skeleton': pose_kwargs.get("visual_draw_skeleton", True),
        'print_score': pose_kwargs.get("visual_print_score", False),
        'part_id': pose_kwargs.get("visual_part_id", 0),
        'from_part': pose_kwargs.get("visual_from_part", 0),
        'vec_id': pose_kwargs.get("visual_vec_id", 0),
        'from_vec': pose_kwargs.get("visual_from_vec", 0),
        'pose_threshold': pose_kwargs.get("visual_pose_threshold", 0.05),
        'write_frames': pose_kwargs.get("visual_write_frames", True),
        'output_directory': pose_kwargs.get("visual_output_directory", "images15F"),
    }
    net.finished = L.Visualizepose(net[frame_layer], net.resized_map, net.limbs, visualize_pose_param=visual_kwargs)
    return net

def mPoseNet_BaseNetReconstruct_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    lr = 1
    decay = 1
    strid_convs = [1,1,1,0,0]
    kernel_sizes = [3,3,3,3,5]

    net = YoloNetPartCompressDepthwiseE(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=4,strid_conv=strid_convs,kernel_sizes=kernel_sizes,
                                       final_pool=False, group_divide=1, lr=lr, decay=decay,addstrs='_recon')
    net = UnifiedMultiScaleLayers(net, layers=["conv4_4_recon", "conv5_8_recon"], tags=["Ref", "Up"], unifiedlayer="convf_recon",
                                  upsampleMethod="Reorg")
    baselayer = "convf_recon"
    n_channel = 64
    group_divide = 8
    n_group = 8
    use_3_layers = 7
    addstrs = '_recon'
    net = mPose_StageXShuffle_Train(net, from_layer=baselayer, stage=1,use_3_layers=use_3_layers, short_cut=False, \
                              base_layer=baselayer, lr=1.0, decay=1.0, num_channels=n_channel, n_group=n_group, addstrs=addstrs,
                              flag_hasoutput=False, flag_has_loss=False)
    # net = mPose_StageXDepthwiseNoLossA_Train(net, from_layer=baselayer, stage=1, use_3_layers=7, use_1_layers=0, lr=2.0,
    #                                   decay=2.0, kernel_size=7, num_channels=n_channel, group_divide=group_divide,flag_hasoutput = False,
    #                                          addstrs='_recon')
    print net.keys()
    net = YoloNetPartCompress(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5,
                                       strid_conv=strid_convs, final_pool=False,lr=0, decay=0)
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3", "conv5_5"], tags=["Ref", "Up"],unifiedlayer="convf", upsampleMethod="Reorg")
    baselayer = "convf"
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    net = mPose_StageX_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=True,base_layer=baselayer, num_channels=n_channel)
    net = mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=True,base_layer=baselayer, num_channels=n_channel)
    net = mPose_StageX_Test(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=False,num_channels=n_channel, flag_output=False)

    net['loss1'] = L.EuclideanLoss(net['stage3_relu4_vec'], net['stage1_relu6_vec_recon'], loss_weight=1)
    net['loss2'] = L.EuclideanLoss(net['stage3_relu4_heat'], net['stage1_relu6_heat_recon'], loss_weight=1)
    return net

def mPoseNet_BaseNetReconstructA_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input

    strid_convs = [1,1,1,0,0]
    kernel_sizes = [3,3,3,3,5]
    net = YoloNetPartCompressDepthwiseF(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=4,strid_conv=strid_convs,kernel_sizes=kernel_sizes,
                                       final_pool=False, group_divide=1, lr=1, decay=1,addstrs='_recon')

    net = YoloNetPartCompress(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5,
                                       strid_conv=strid_convs, final_pool=False,lr=0, decay=0)

    net['loss1'] = L.EuclideanLoss(net['conv4_3'], net['conv4_4_recon'], loss_weight=1.0)
    net['loss2'] = L.EuclideanLoss(net['conv5_5'], net['conv5_8_recon'], loss_weight=1.0)
    return net

def mPoseNet_BaseNetReconstructB_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    lr = 1
    decay = 1
    strid_convs = [1,1,1,0,0]
    kernel_sizes = [3,3,3,3,5]

    net = YoloNetPartCompressDepthwiseE(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=4,strid_conv=strid_convs,kernel_sizes=kernel_sizes,
                                       final_pool=False, group_divide=1, lr=lr, decay=decay,addstrs='_recon')
    net = UnifiedMultiScaleLayers(net, layers=["conv4_4_recon", "conv5_8_recon"], tags=["Ref", "Up"], unifiedlayer="convf_recon",
                                  upsampleMethod="Reorg")
    baselayer = "convf_recon"
    n_channel = 128
    group_divide = 1
    use_3_layers = 7
    net = mPose_StageXDepthwiseNoLoss_Train(net, from_layer=baselayer, stage=1, use_3_layers=use_3_layers, use_1_layers=0, lr=2.0,
                                      decay=2.0, kernel_size=7, num_channels=n_channel, group_divide=group_divide,flag_hasoutput = True,
                                            flag_hasloss=False,addstrs='_recon')

    net = YoloNetPartCompress(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5,
                                       strid_conv=strid_convs, final_pool=False,lr=0, decay=0)
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3", "conv5_5"], tags=["Ref", "Up"],unifiedlayer="convf", upsampleMethod="Reorg")
    baselayer = "convf"
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    net = mPose_StageX_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=True,base_layer=baselayer, num_channels=n_channel)
    net = mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=True,base_layer=baselayer, num_channels=n_channel)
    net = mPose_StageX_Test(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3,use_3_layers=use_3_layers,
                            use_1_layers=use_1_layers, short_cut=False,num_channels=n_channel, flag_output=True)

    net['loss1'] = L.EuclideanLoss(net['stage3_conv5_vec'], net['stage1_conv7_vec'], loss_weight=1)
    net['loss2'] = L.EuclideanLoss(net['stage3_conv5_heat'], net['stage1_conv7_heat'], loss_weight=1)
    return net