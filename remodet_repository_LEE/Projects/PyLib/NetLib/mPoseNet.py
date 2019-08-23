# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

from YoloNet import *
from MultiScaleLayer import *
from ConvBNLayer import *

def mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
               use_3_layers=5, use_1_layers=2, short_cut=True, base_layer="convf", lr=1, decay=1):
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

def mPose_StageX_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
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
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+2)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
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
    net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers+2)
    net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
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
		num_output=34, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
    ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=False, use_relu=False, \
		num_output=18, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
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
		num_output=34, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
    ConvBNUnitLayer(net, from2_layer, conv_heat, use_bn=False, use_relu=False, \
		num_output=18, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
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
    net = YoloNetPart(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False, lr=1, decay=1)
    # concat conv4_3 & conv5_5
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3","conv5_5"], tags=["Ref","Up"], \
                                  unifiedlayer="convf", upsampleMethod="Reorg")
    # Stages
    baselayer = "convf"
    use_3_layers = 5
    use_1_layers = 0
    net = mPose_StageX_Train(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer, lr=4, decay=1)
    net = mPose_StageX_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer, lr=4, decay=1)
    net = mPose_StageX_Train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False, \
                           lr=4, decay=1)
    # for Test
    if not train:
        print(net.keys())
        conv_vec = "stage{}_conv{}_vec".format(3,use_3_layers+use_1_layers)
        conv_heat = "stage{}_conv{}_heat".format(3,use_3_layers+use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(use_stages)
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

def mPoseNet_COCO_3S_Test(net, from_layer="data", frame_layer="orig_data", use_bn=True, **pose_kwargs):
    # Darknet19
    net = YoloNetPart(net, from_layer=from_layer, use_bn=use_bn, use_layers=5, use_sub_layers=5, final_pool=False, lr=1, decay=1)
    # concat conv4_3 & conv5_5
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3","conv5_5"], tags=["Ref","Up"], \
                                  unifiedlayer="convf", upsampleMethod="Reorg")
    # Stages
    baselayer = "convf"
    use_3_layers = 5
    use_1_layers = 0
    net = mPose_StageX_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer)
    net = mPose_StageX_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=True, \
                           base_layer=baselayer)
    net = mPose_StageX_Test(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False,
                           base_layer=baselayer)

    # net = mPose_StageX_Test(net, from_layer="concat_stage3", out_layer="concat_stage4", stage=4, \
    #                        use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False)

    conv_vec = "stage{}_conv{}_vec".format(3,use_3_layers+use_1_layers)
    conv_heat = "stage{}_conv{}_heat".format(3,use_3_layers+use_1_layers)
    feaLayers = []
    feaLayers.append(net[conv_heat])
    feaLayers.append(net[conv_vec])
    outlayer = "concat_stage{}".format(3)
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
        'type': pose_kwargs.get("visual_type", P.Visualizepose.POSE),
        'visualize': pose_kwargs.get("visual_visualize", True),
        'draw_skeleton': pose_kwargs.get("visual_draw_skeleton", True),
        'print_score': pose_kwargs.get("visual_print_score", False),
        'part_id': pose_kwargs.get("visual_part_id", 0),
        'from_part': pose_kwargs.get("visual_from_part", 0),
        'vec_id': pose_kwargs.get("visual_vec_id", 0),
        'from_vec': pose_kwargs.get("visual_from_vec", 0),
        'pose_threshold': pose_kwargs.get("visual_pose_threshold", 0.05),
        'write_frames': pose_kwargs.get("visual_write_frames", False),
        'output_directory': pose_kwargs.get("visual_output_directory", ""),
    }
    net.finished = L.Visualizepose(net[frame_layer], net.resized_map, net.limbs, visualize_pose_param=visual_kwargs)
    return net
