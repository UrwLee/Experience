# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

from VggNet import *

def Pose_Stage1_COCO(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    # L1 & L2 for conv1
    net.conv5_1_CPM_L1 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_CPM_L1 = L.ReLU(net.conv5_1_CPM_L1, in_place=True)
    net.conv5_1_CPM_L2 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_CPM_L2 = L.ReLU(net.conv5_1_CPM_L2, in_place=True)
    # L1 & L2 for conv2
    net.conv5_2_CPM_L1 = L.Convolution(net.relu5_1_CPM_L1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_CPM_L1 = L.ReLU(net.conv5_2_CPM_L1, in_place=True)
    net.conv5_2_CPM_L2 = L.Convolution(net.relu5_1_CPM_L2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_CPM_L2 = L.ReLU(net.conv5_2_CPM_L2, in_place=True)
    # L1 & L2 for conv3
    net.conv5_3_CPM_L1 = L.Convolution(net.relu5_2_CPM_L1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_3_CPM_L1 = L.ReLU(net.conv5_3_CPM_L1, in_place=True)
    net.conv5_3_CPM_L2 = L.Convolution(net.relu5_2_CPM_L2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_3_CPM_L2 = L.ReLU(net.conv5_3_CPM_L2, in_place=True)
    # L1 & L2 for conv4
    net.conv5_4_CPM_L1 = L.Convolution(net.relu5_3_CPM_L1, num_output=512, pad=0, kernel_size=1, **kwargs)
    net.relu5_4_CPM_L1 = L.ReLU(net.conv5_4_CPM_L1, in_place=True)
    net.conv5_4_CPM_L2 = L.Convolution(net.relu5_3_CPM_L2, num_output=512, pad=0, kernel_size=1, **kwargs)
    net.relu5_4_CPM_L2 = L.ReLU(net.conv5_4_CPM_L2, in_place=True)
    # L1 & L2 for conv5
    net.conv5_5_CPM_L1 = L.Convolution(net.relu5_4_CPM_L1, num_output=38, pad=0, kernel_size=1, **kwargs)
    net.conv5_5_CPM_L2 = L.Convolution(net.relu5_4_CPM_L2, num_output=19, pad=0, kernel_size=1, **kwargs)
    # concat layers
    fea_layers = []
    fea_layers.append(net.conv5_5_CPM_L1)
    fea_layers.append(net.conv5_5_CPM_L2)
    fea_layers.append(net[from_layer])
    net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def Pose_Stage1_COCO_train(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", \
                           mask_L1="vec_mask", mask_L2="heat_mask", \
                           label_L1="vec_label", label_L2="heat_label", lr=1, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    # L1 & L2 for conv1
    net.conv5_1_CPM_L1 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_CPM_L1 = L.ReLU(net.conv5_1_CPM_L1, in_place=True)
    net.conv5_1_CPM_L2 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_1_CPM_L2 = L.ReLU(net.conv5_1_CPM_L2, in_place=True)
    # L1 & L2 for conv2
    net.conv5_2_CPM_L1 = L.Convolution(net.relu5_1_CPM_L1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_CPM_L1 = L.ReLU(net.conv5_2_CPM_L1, in_place=True)
    net.conv5_2_CPM_L2 = L.Convolution(net.relu5_1_CPM_L2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_2_CPM_L2 = L.ReLU(net.conv5_2_CPM_L2, in_place=True)
    # L1 & L2 for conv3
    net.conv5_3_CPM_L1 = L.Convolution(net.relu5_2_CPM_L1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_3_CPM_L1 = L.ReLU(net.conv5_3_CPM_L1, in_place=True)
    net.conv5_3_CPM_L2 = L.Convolution(net.relu5_2_CPM_L2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu5_3_CPM_L2 = L.ReLU(net.conv5_3_CPM_L2, in_place=True)
    # L1 & L2 for conv4
    net.conv5_4_CPM_L1 = L.Convolution(net.relu5_3_CPM_L1, num_output=512, pad=0, kernel_size=1, **kwargs)
    net.relu5_4_CPM_L1 = L.ReLU(net.conv5_4_CPM_L1, in_place=True)
    net.conv5_4_CPM_L2 = L.Convolution(net.relu5_3_CPM_L2, num_output=512, pad=0, kernel_size=1, **kwargs)
    net.relu5_4_CPM_L2 = L.ReLU(net.conv5_4_CPM_L2, in_place=True)
    # L1 & L2 for conv5
    net.conv5_5_CPM_L1 = L.Convolution(net.relu5_4_CPM_L1, num_output=38, pad=0, kernel_size=1, **kwargs)
    net.conv5_5_CPM_L2 = L.Convolution(net.relu5_4_CPM_L2, num_output=19, pad=0, kernel_size=1, **kwargs)
    # loss_L1 & loss_L2
    net.weight_stage1_L1 = L.Eltwise(net.conv5_5_CPM_L1, net[mask_L1], eltwise_param=dict(operation=P.Eltwise.PROD))
    net.loss_stage1_L1 = L.EuclideanLoss(net.weight_stage1_L1, net[label_L1], loss_weight=1)
    net.weight_stage1_L2 = L.Eltwise(net.conv5_5_CPM_L2, net[mask_L2], eltwise_param=dict(operation=P.Eltwise.PROD))
    net.loss_stage1_L2 = L.EuclideanLoss(net.weight_stage1_L2, net[label_L2], loss_weight=1)
    # concat layers
    fea_layers = []
    fea_layers.append(net.conv5_5_CPM_L1)
    fea_layers.append(net.conv5_5_CPM_L2)
    fea_layers.append(net[from_layer])
    net[out_layer] = L.Concat(*fea_layers, concat_param=dict(axis=1))
    return net

def Pose_StageX_COCO(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, \
                     short_cut=True, base_layer="conv4_4_CPM", lr=4, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    # L1 & L2 for conv1
    conv_L1 = "Mconv1_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[from_layer], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu1_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv1_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[from_layer], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu1_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv2
    conv_L1 = "Mconv2_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu2_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv2_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu2_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv3
    conv_L1 = "Mconv3_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu3_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv3_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu3_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv4
    conv_L1 = "Mconv4_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu4_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv4_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu4_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv5
    conv_L1 = "Mconv5_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu5_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv5_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu5_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv6
    conv_L1 = "Mconv6_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_L1 = "Mrelu6_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv6_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_L2 = "Mrelu6_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv7
    conv_L1 = "Mconv7_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=38, pad=0, kernel_size=1, **kwargs)
    conv_L2 = "Mconv7_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=19, pad=0, kernel_size=1, **kwargs)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_L1])
        fea_layers.append(net[conv_L2])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def Pose_StageX_COCO_train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, \
                           mask_L1="vec_mask", mask_L2="heat_mask", \
                           label_L1="vec_label", label_L2="heat_label", \
                           short_cut=True, base_layer="conv4_4_CPM", lr=4, decay=1):
    kwargs = {
            'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    assert from_layer in net.keys()
    # L1 & L2 for conv1
    conv_L1 = "Mconv1_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[from_layer], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu1_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv1_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[from_layer], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu1_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv2
    conv_L1 = "Mconv2_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu2_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv2_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu2_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv3
    conv_L1 = "Mconv3_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu3_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv3_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu3_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv4
    conv_L1 = "Mconv4_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu4_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv4_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu4_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv5
    conv_L1 = "Mconv5_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L1 = "Mrelu5_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv5_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=3, kernel_size=7, **kwargs)
    relu_L2 = "Mrelu5_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv6
    conv_L1 = "Mconv6_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_L1 = "Mrelu6_stage{}_L1".format(stage)
    net[relu_L1] = L.ReLU(net[conv_L1], in_place=True)
    conv_L2 = "Mconv6_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=128, pad=0, kernel_size=1, **kwargs)
    relu_L2 = "Mrelu6_stage{}_L2".format(stage)
    net[relu_L2] = L.ReLU(net[conv_L2], in_place=True)
    # L1 & L2 for conv7
    conv_L1 = "Mconv7_stage{}_L1".format(stage)
    net[conv_L1] = L.Convolution(net[relu_L1], num_output=38, pad=0, kernel_size=1, **kwargs)
    conv_L2 = "Mconv7_stage{}_L2".format(stage)
    net[conv_L2] = L.Convolution(net[relu_L2], num_output=19, pad=0, kernel_size=1, **kwargs)
    # Loss
    weight_L1 = "weight_stage{}_L1".format(stage)
    weight_L2 = "weight_stage{}_L2".format(stage)
    loss_L1 = "loss_stage{}_L1".format(stage)
    loss_L2 = "loss_stage{}_L2".format(stage)
    net[weight_L1] = L.Eltwise(net[conv_L1], net[mask_L1], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_L1] = L.EuclideanLoss(net[weight_L1], net[label_L1], loss_weight=1)
    net[weight_L2] = L.Eltwise(net[conv_L2], net[mask_L2], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_L2] = L.EuclideanLoss(net[weight_L2], net[label_L2], loss_weight=1)
    # 特征拼接
    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_L1])
        fea_layers.append(net[conv_L2])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

# Define pre-10 layers of VGG19
def VGG19_PoseNet_COCO_Test(net, from_layer="data", frame_layer="orig_data", **pose_kwargs):
    # baseNet-VGG19
    assert from_layer in net.keys()
    net = VGG19Net_Pre10(net, from_layer="data")
    # conv4_3_CPM & conv4_4_CPM
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    # conv4_3_CPM
    net.conv4_3_CPM = L.Convolution(net.relu4_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_CPM = L.ReLU(net.conv4_3_CPM, in_place=True)
    net.conv4_4_CPM = L.Convolution(net.relu4_3_CPM, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu4_4_CPM = L.ReLU(net.conv4_4_CPM, in_place=True)
    # Stage1
    net = Pose_Stage1_COCO(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", lr=1, decay=1)
    # Stage2-6
    net = Pose_StageX_COCO(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO(net, from_layer="concat_stage3", out_layer="concat_stage4", stage=3, short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO(net, from_layer="concat_stage4", out_layer="concat_stage5", stage=4, short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO(net, from_layer="concat_stage5", out_layer="concat_stage6", stage=5, short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO(net, from_layer="concat_stage6", out_layer="concat_stage7", stage=6, short_cut=False, lr=4, decay=1)
    # concat the output layers
    feaLayers = []
    feaLayers.append(net["Mconv7_stage6_L2"])
    feaLayers.append(net["Mconv7_stage6_L1"])
    net["concat_stage7"] = L.Concat(*feaLayers, axis=1)
    # Resize
    resize_kwargs = {
        'factor': pose_kwargs.get("resize_factor", 8),
        'scale_gap': pose_kwargs.get("resize_scale_gap", 0.3),
        'start_scale': pose_kwargs.get("resize_start_scale", 1.0),
    }
    net.resized_map = L.ImResize(net.concat_stage7, name="resize", imresize_param=resize_kwargs)
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
        'visualize': pose_kwargs.get("visual_visualize", True),
        'draw_skeleton': pose_kwargs.get("visual_draw_skeleton", True),
        'print_score': pose_kwargs.get("visual_print_score", False),
        'type': pose_kwargs.get("visual_type", P.Visualizepose.POSE),
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

def VGG19_PoseNet_Stage3_COCO_Test(net, from_layer="data", frame_layer="orig_data", **pose_kwargs):
    # baseNet-VGG19
    assert from_layer in net.keys()
    net = VGG19Net_Pre10(net, from_layer="data")
    # conv4_3_CPM & conv4_4_CPM
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    # conv4_3_CPM
    net.conv4_3_CPM = L.Convolution(net.relu4_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_CPM = L.ReLU(net.conv4_3_CPM, in_place=True)
    net.conv4_4_CPM = L.Convolution(net.relu4_3_CPM, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu4_4_CPM = L.ReLU(net.conv4_4_CPM, in_place=True)
    # Stage1
    net = Pose_Stage1_COCO(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", lr=1, decay=1)
    # Stage2-6
    net = Pose_StageX_COCO(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO(net, from_layer="concat_stage3", out_layer="concat_stage4", stage=3, short_cut=False, lr=4, decay=1)
    # concat the output layers
    feaLayers = []
    feaLayers.append(net["Mconv7_stage3_L2"])
    feaLayers.append(net["Mconv7_stage3_L1"])
    net["concat_stage4"] = L.Concat(*feaLayers, axis=1)
    # Resize
    resize_kwargs = {
        'factor': pose_kwargs.get("resize_factor", 8),
        'scale_gap': pose_kwargs.get("resize_scale_gap", 0.3),
        'start_scale': pose_kwargs.get("resize_start_scale", 1.0),
    }
    net.resized_map = L.ImResize(net.concat_stage4, name="resize", imresize_param=resize_kwargs)
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

def VGG19_PoseNet_COCO_6S_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # Slice for label and mask
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[38,57,95], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[38,57,95,114], axis=1))
    # Label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # baseNet-VGG19
    net = VGG19Net_Pre10(net, from_layer=data_layer)
    # conv4_3_CPM & conv4_4_CPM
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    # conv4_3_CPM
    net.conv4_3_CPM = L.Convolution(net.relu4_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_CPM = L.ReLU(net.conv4_3_CPM, in_place=True)
    net.conv4_4_CPM = L.Convolution(net.relu4_3_CPM, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu4_4_CPM = L.ReLU(net.conv4_4_CPM, in_place=True)
    # Stage1
    net = Pose_Stage1_COCO_train(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", lr=1, decay=1)
    # Stage2-6
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage3", out_layer="concat_stage4", stage=3, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage4", out_layer="concat_stage5", stage=4, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage5", out_layer="concat_stage6", stage=5, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage6", out_layer="concat_stage7", stage=6, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=False, lr=4, decay=1)
    # for Test
    if not train:
        net.vec_out = L.Eltwise(net.vec_mask, net.Mconv7_stage6_L1, eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net.Mconv7_stage6_L2, eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        net["concat_stage7"] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net.concat_stage7, name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 64),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 20),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 32),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.3),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 96*96),
            'eval_iters': pose_test_kwargs.get("eval_test_iters", 10000),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def VGG19_PoseNet_COCO_3S_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # Slice for label and mask
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[38,57,95], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[38,57,95,114], axis=1))
    # Label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # baseNet-VGG19
    net = VGG19Net_Pre10(net, from_layer=data_layer)
    # conv4_3_CPM & conv4_4_CPM
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    # conv4_3_CPM
    net.conv4_3_CPM = L.Convolution(net.relu4_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_CPM = L.ReLU(net.conv4_3_CPM, in_place=True)
    net.conv4_4_CPM = L.Convolution(net.relu4_3_CPM, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu4_4_CPM = L.ReLU(net.conv4_4_CPM, in_place=True)
    # Stage1
    net = Pose_Stage1_COCO_train(net, from_layer="relu4_4_CPM", out_layer="concat_stage2", \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", lr=1, decay=1)
    # Stage2-3
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=2, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=True, base_layer="relu4_4_CPM", lr=4, decay=1)
    net = Pose_StageX_COCO_train(net, from_layer="concat_stage3", out_layer="concat_stage4", stage=3, \
                               mask_L1="vec_mask", mask_L2="heat_mask", \
                               label_L1="vec_label", label_L2="heat_label", \
                               short_cut=False, lr=4, decay=1)
    # for Test
    if not train:
        net.vec_out = L.Eltwise(net.vec_mask, net.Mconv7_stage3_L1, eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net.Mconv7_stage3_L2, eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        net["concat_stage4"] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net.concat_stage4, name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 64),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 20),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 32),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.3),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 96*96),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net
