# -*- coding: utf-8 -*-
import os
import sys
import math
sys.dont_write_bytecode = True
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from PyLib.LayerParam.MultiBoxLossLayerParam import *
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.InceptionLayer import *
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.VggNet import VGG16_BaseNet_ChangeChannel
from PyLib.NetLib.YoloNet import YoloNetPart
from BaseNet import *
from AddC6 import *
from DetectorHeader import *
from DAP_Param import *
import numpy as np
from solverParam import truncvalues
# ##############################################################################
def Deconv(net,from_layer,num_output,group,kernel_size,stride,lr_mult,decay_mult,use_bn,use_scale,use_relu):
    deconv_param = {
        'num_output': num_output,
        'kernel_size': kernel_size,
        'pad': 0,
        'stride': stride,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        'group': group,
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
        'convolution_param': deconv_param
    }
    out_layer = from_layer + "_deconv"
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    base_conv_name = out_layer
    from_layer = out_layer
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
    }
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=lr_mult, decay_mult=0), dict(lr_mult=lr_mult, decay_mult=0)],
        'filler': dict(type='constant', value=1.0),
        'bias_filler': dict(type='constant', value=0.2),
    }
    if use_bn:
        bn_name = '{}_bn'.format(base_conv_name)
        net[bn_name] = L.BatchNorm(net[from_layer], in_place=True, **bn_kwargs)
        from_layer = bn_name
    if use_scale:
        sb_name = '{}_scale'.format(base_conv_name)
        net[sb_name] = L.Scale(net[from_layer], in_place=True, **sb_kwargs)
        from_layer = sb_name
    if use_relu:
        relu_name = '{}_relu'.format(base_conv_name)
        net[relu_name] = L.ReLU(net[from_layer], in_place=True)
# ------------------------------------------------------------------------------
# Final Network
flag_train_withperson = True
def InceptionOfficialLayer(net, from_layer, out_layer, channels_1=1,channels_3=[],channels_5=[],channels_ave=1,inter_bn = True,leaky=False):
    fea_layer = from_layer

    concatlayers = []
    mid_layer = "{}/incep/1x1".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_1, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])
    start_layer = mid_layer
    mid_layer = "{}/incep/1_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True,num_output=channels_3[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_3[1], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/2_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky)
    start_layer = mid_layer
    mid_layer = "{}/incep/5x5".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[1], kernel_size=5, pad=2,
                    stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/pool".format(out_layer)
    net[mid_layer] = L.Pooling(net[fea_layer], pool=P.Pooling.AVE, kernel_size=3, stride=1, pad=1)
    start_layer = mid_layer
    mid_layer = "{}/incep/pool_1x1".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_ave, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky)
    concatlayers.append(net[mid_layer])
    # incep
    layer_name = "{}/incep".format(out_layer)
    name = "{}/incep".format(out_layer)
    net[name] = L.Concat(*concatlayers, name=layer_name, axis=1)

    return net

def FaceBoxAlikeNet(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    lr = 1.0
    decay = 1.0
    from_layer = data_layer
    num_channels = [32,64,128]
    k_sizes = [7,3,3]
    strides = [2,1,1]
    for i in xrange(len(num_channels)):
        add_layer = "conv{}".format(i+1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                    num_output=num_channels[i], kernel_size=k_sizes[i], pad=(k_sizes[i]-1)/2, stride=strides[i], use_scale=True,
                    n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
        # if not i == len(num_channels) - 1:
        add_layer = "pool{}".format(i+1)
        net[add_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
        from_layer = add_layer
    layer_cnt = len(num_channels)
    num_channels = [128,128,128]
    divide_scale = 4
    for i in xrange(len(num_channels)):
        n_chan = num_channels[i]
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        net = InceptionOfficialLayer(net, from_layer, add_layer, channels_1=n_chan/divide_scale, channels_3=[n_chan/8, n_chan/4],
                                     channels_5=[n_chan/8, n_chan/4], channels_ave=n_chan/divide_scale, inter_bn=True, leaky=False)
        from_layer = "conv{}_{}/incep".format(layer_cnt+1,i + 1)
    layer_cnt += 1
    num_channels = [128,128,128]
    for i in xrange(len(num_channels)):
        if i == 0:
            stride = 2
        else:
            stride = 1
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                        num_output=num_channels[i], kernel_size=3, pad=1, stride=stride,
                        use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
    layer_cnt += 1
    num_channels = [128, 128, 128]
    for i in xrange(len(num_channels)):
        if i == 0:
            stride = 2
        else:
            stride = 1
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                        num_output=num_channels[i], kernel_size=3, pad=1, stride=stride,
                        use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
    lr_detnetperson = 1.0
    # Create SSD Header for SSD1
    if flag_train_withperson:
        mbox_1_layers = SsdDetectorHeaders(net, \
                                           net_width=net_width, net_height=net_height, data_layer=data_layer, \
                                           from_layers=ssd_Param_1.get('feature_layers', []), \
                                           num_classes=ssd_Param_1.get("num_classes", 2), \
                                           boxsizes=ssd_Param_1.get("anchor_boxsizes", []), \
                                           aspect_ratios=ssd_Param_1.get("anchor_aspect_ratios", []), \
                                           prior_variance=ssd_Param_1.get("anchor_prior_variance",
                                                                          [0.1, 0.1, 0.2, 0.2]), \
                                           flip=ssd_Param_1.get("anchor_flip", True), \
                                           clip=ssd_Param_1.get("anchor_clip", True), \
                                           normalizations=ssd_Param_1.get("interlayers_normalizations", []), \
                                           use_batchnorm=ssd_Param_1.get("interlayers_use_batchnorm", True), \
                                           inter_layer_channels=ssd_Param_1.get("interlayers_channels_kernels", []), \
                                           use_focus_loss=ssd_Param_1.get("bboxloss_using_focus_loss", False), \
                                           use_dense_boxes=ssd_Param_1.get('bboxloss_use_dense_boxes', False), \
                                           stage=1, lr_mult=lr_detnetperson)
        # make Loss or Detout for SSD1

        if train:
            loss_param = get_loss_param(normalization=ssd_Param_1.get("bboxloss_normalization", P.Loss.VALID))
            mbox_1_layers.append(net[gt_label])
            use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes', False)
            if use_dense_boxes:
                bboxloss_param = {
                    'gt_labels': ssd_Param_1.get('gt_labels', []),
                    'target_labels': ssd_Param_1.get('target_labels', []),
                    'num_classes': ssd_Param_1.get("num_classes", 2),
                    'alias_id': ssd_Param_1.get("alias_id", 0),
                    'loc_loss_type': ssd_Param_1.get("bboxloss_loc_loss_type", P.MultiBoxLoss.SMOOTH_L1),
                    'conf_loss_type': ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.LOGISTIC),
                    'loc_weight': ssd_Param_1.get("bboxloss_loc_weight", 1),
                    'conf_weight': ssd_Param_1.get("bboxloss_conf_weight", 1),
                    'overlap_threshold': ssd_Param_1.get("bboxloss_overlap_threshold", 0.5),
                    'neg_overlap': ssd_Param_1.get("bboxloss_neg_overlap", 0.5),
                    'size_threshold': ssd_Param_1.get("bboxloss_size_threshold", 0.0001),
                    'do_neg_mining': ssd_Param_1.get("bboxloss_do_neg_mining", True),
                    'neg_pos_ratio': ssd_Param_1.get("bboxloss_neg_pos_ratio", 3),
                    'using_focus_loss': ssd_Param_1.get("bboxloss_using_focus_loss", False),
                    'gama': ssd_Param_1.get("bboxloss_focus_gama", 2),
                    'alpha':ssd_Param_1.get("bboxloss_focus_alpha", 0.25),
                    'use_difficult_gt': ssd_Param_1.get("bboxloss_use_difficult_gt", False),
                    'code_type': ssd_Param_1.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
                    'use_prior_for_matching': True,
                    'encode_variance_in_target': False,
                    'flag_noperson': ssd_Param_1.get('flag_noperson', False),
                }
                net["mbox_1_loss"] = L.DenseBBoxLoss(*mbox_1_layers, dense_bbox_loss_param=bboxloss_param, \
                                                     loss_param=loss_param,
                                                     include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                                     propagate_down=[True, True, False, False])
            else:
                bboxloss_param = {
                    'gt_labels': ssd_Param_1.get('gt_labels', []),
                    'target_labels': ssd_Param_1.get('target_labels', []),
                    'num_classes': ssd_Param_1.get("num_classes", 2),
                    'alias_id': ssd_Param_1.get("alias_id", 0),
                    'loc_loss_type': ssd_Param_1.get("bboxloss_loc_loss_type", P.MultiBoxLoss.SMOOTH_L1),
                    'conf_loss_type': ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX),
                    'loc_weight': ssd_Param_1.get("bboxloss_loc_weight", 1),
                    'conf_weight': ssd_Param_1.get("bboxloss_conf_weight", 1),
                    'overlap_threshold': ssd_Param_1.get("bboxloss_overlap_threshold", 0.5),
                    'neg_overlap': ssd_Param_1.get("bboxloss_neg_overlap", 0.5),
                    'size_threshold': ssd_Param_1.get("bboxloss_size_threshold", 0.0001),
                    'do_neg_mining': ssd_Param_1.get("bboxloss_do_neg_mining", True),
                    'neg_pos_ratio': ssd_Param_1.get("bboxloss_neg_pos_ratio", 3),
                    'using_focus_loss': ssd_Param_1.get("bboxloss_using_focus_loss", False),
                    'gama': ssd_Param_1.get("bboxloss_focus_gama", 2),
                    'alpha': ssd_Param_1.get("bboxloss_focus_alpha", 0.25),
                    'use_difficult_gt': ssd_Param_1.get("bboxloss_use_difficult_gt", False),
                    'code_type': ssd_Param_1.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
                    'match_type': P.MultiBoxLoss.PER_PREDICTION,
                    'share_location': True,
                    'use_prior_for_matching': True,
                    'background_label_id': 0,
                    'encode_variance_in_target': False,
                    'map_object_to_agnostic': False,
                }
                net["mbox_1_loss"] = L.BBoxLoss(*mbox_1_layers, bbox_loss_param=bboxloss_param, \
                                                loss_param=loss_param,
                                                include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                                propagate_down=[True, True, False, False])
        else:
            if ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
                reshape_name = "mbox_1_conf_reshape"
                net[reshape_name] = L.Reshape(mbox_1_layers[1], \
                                              shape=dict(dim=[0, -1, ssd_Param_1.get("num_classes", 2)]))
                softmax_name = "mbox_1_conf_softmax"
                net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
                flatten_name = "mbox_1_conf_flatten"
                net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
                mbox_1_layers[1] = net[flatten_name]
            elif ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
                sigmoid_name = "mbox_1_conf_sigmoid"
                net[sigmoid_name] = L.Sigmoid(mbox_1_layers[1])
                mbox_1_layers[1] = net[sigmoid_name]
            else:
                raise ValueError("Unknown conf loss type.")
            # Det-out param
            det_out_param = {
                'num_classes': ssd_Param_1.get("num_classes", 2),
                'target_labels': ssd_Param_1.get('detout_target_labels', []),
                'alias_id': ssd_Param_1.get("alias_id", 0),
                'conf_threshold': ssd_Param_1.get("detout_conf_threshold", 0.01),
                'nms_threshold': ssd_Param_1.get("detout_nms_threshold", 0.45),
                'size_threshold': ssd_Param_1.get("detout_size_threshold", 0.0001),
                'top_k': ssd_Param_1.get("detout_top_k", 30),
                'share_location': True,
                'code_type': P.PriorBox.CENTER_SIZE,
                'background_label_id': 0,
                'variance_encoded_in_target': False,
            }
            use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes', False)
            if use_dense_boxes:
                net.detection_out_1 = L.DenseDetOut(*mbox_1_layers, \
                                                    detection_output_param=det_out_param, \
                                                    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
            else:
                net.detection_out_1 = L.DetOut(*mbox_1_layers, \
                                               detection_output_param=det_out_param, \
                                               include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    # make Loss & Detout for SSD2
    if use_ssd2_for_detection:
        mbox_2_layers = SsdDetectorHeaders(net, \
                                           net_width=net_width, net_height=net_height, data_layer=data_layer, \
                                           from_layers=ssd_Param_2.get('feature_layers', []), \
                                           num_classes=ssd_Param_2.get("num_classes", 2), \
                                           boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
                                           aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios", []), \
                                           prior_variance=ssd_Param_2.get("anchor_prior_variance",
                                                                          [0.1, 0.1, 0.2, 0.2]), \
                                           flip=ssd_Param_2.get("anchor_flip", True), \
                                           clip=ssd_Param_2.get("anchor_clip", True), \
                                           normalizations=ssd_Param_2.get("interlayers_normalizations", []), \
                                           use_batchnorm=ssd_Param_2.get("interlayers_use_batchnorm", True), \
                                           inter_layer_channels=ssd_Param_2.get("interlayers_channels_kernels", []), \
                                           use_focus_loss=ssd_Param_2.get("bboxloss_using_focus_loss", False), \
                                           use_dense_boxes=ssd_Param_2.get('bboxloss_use_dense_boxes', False), \
                                           stage=2)
        # make Loss or Detout for SSD1
        if train:
            loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization", P.Loss.VALID))
            mbox_2_layers.append(net[gt_label])
            use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes', False)
            if use_dense_boxes:
                bboxloss_param = {
                    'gt_labels': ssd_Param_2.get('gt_labels', []),
                    'target_labels': ssd_Param_2.get('target_labels', []),
                    'num_classes': ssd_Param_2.get("num_classes", 2),
                    'alias_id': ssd_Param_2.get("alias_id", 0),
                    'loc_loss_type': ssd_Param_2.get("bboxloss_loc_loss_type", P.MultiBoxLoss.SMOOTH_L1),
                    'conf_loss_type': ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.LOGISTIC),
                    'loc_weight': ssd_Param_2.get("bboxloss_loc_weight", 1),
                    'conf_weight': ssd_Param_2.get("bboxloss_conf_weight", 1),
                    'overlap_threshold': ssd_Param_2.get("bboxloss_overlap_threshold", 0.5),
                    'neg_overlap': ssd_Param_2.get("bboxloss_neg_overlap", 0.5),
                    'size_threshold': ssd_Param_2.get("bboxloss_size_threshold", 0.0001),
                    'do_neg_mining': ssd_Param_2.get("bboxloss_do_neg_mining", True),
                    'neg_pos_ratio': ssd_Param_2.get("bboxloss_neg_pos_ratio", 3),
                    'using_focus_loss': ssd_Param_2.get("bboxloss_using_focus_loss", False),
                    'gama': ssd_Param_1.get("bboxloss_focus_gama", 2),
                    'alpha':ssd_Param_1.get("bboxloss_focus_alpha", 0.25),
                    'use_difficult_gt': ssd_Param_2.get("bboxloss_use_difficult_gt", False),
                    'code_type': ssd_Param_2.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
                    'use_prior_for_matching': True,
                    'encode_variance_in_target': False,
                    'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                }
                net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                                     loss_param=loss_param,
                                                     include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                                     propagate_down=[True, True, False, False])
            else:
                bboxloss_param = {
                    'gt_labels': ssd_Param_2.get('gt_labels', []),
                    'target_labels': ssd_Param_2.get('target_labels', []),
                    'num_classes': ssd_Param_2.get("num_classes", 2),
                    'alias_id': ssd_Param_2.get("alias_id", 0),
                    'loc_loss_type': ssd_Param_2.get("bboxloss_loc_loss_type", P.MultiBoxLoss.SMOOTH_L1),
                    'conf_loss_type': ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX),
                    'loc_weight': ssd_Param_2.get("bboxloss_loc_weight", 1),
                    'conf_weight': ssd_Param_2.get("bboxloss_conf_weight", 1),
                    'overlap_threshold': ssd_Param_2.get("bboxloss_overlap_threshold", 0.5),
                    'neg_overlap': ssd_Param_2.get("bboxloss_neg_overlap", 0.5),
                    'size_threshold': ssd_Param_2.get("bboxloss_size_threshold", 0.0001),
                    'do_neg_mining': ssd_Param_2.get("bboxloss_do_neg_mining", True),
                    'neg_pos_ratio': ssd_Param_2.get("bboxloss_neg_pos_ratio", 3),
                    'using_focus_loss': ssd_Param_2.get("bboxloss_using_focus_loss", False),
                    'gama': ssd_Param_1.get("bboxloss_focus_gama", 2),
                    'alpha':ssd_Param_1.get("bboxloss_focus_alpha", 0.25),
                    'use_difficult_gt': ssd_Param_2.get("bboxloss_use_difficult_gt", False),
                    'code_type': ssd_Param_2.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
                    'match_type': P.MultiBoxLoss.PER_PREDICTION,
                    'share_location': True,
                    'use_prior_for_matching': True,
                    'background_label_id': 0,
                    'encode_variance_in_target': False,
                    'map_object_to_agnostic': False,
                }
                net["mbox_2_loss"] = L.BBoxLoss(*mbox_2_layers, bbox_loss_param=bboxloss_param, \
                                                loss_param=loss_param,
                                                include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                                propagate_down=[True, True, False, False])
        else:
            if ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
                reshape_name = "mbox_2_conf_reshape"
                net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                                              shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes", 2)]))
                softmax_name = "mbox_2_conf_softmax"
                net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
                flatten_name = "mbox_2_conf_flatten"
                net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
                mbox_2_layers[1] = net[flatten_name]
            elif ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
                sigmoid_name = "mbox_2_conf_sigmoid"
                net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
                mbox_2_layers[1] = net[sigmoid_name]
            else:
                raise ValueError("Unknown conf loss type.")
            # Det-out param
            det_out_param = {
                'num_classes': ssd_Param_2.get("num_classes", 2),
                'target_labels': ssd_Param_2.get('detout_target_labels', []),
                'alias_id': ssd_Param_2.get("alias_id", 0),
                'conf_threshold': ssd_Param_2.get("detout_conf_threshold", 0.01),
                'nms_threshold': ssd_Param_2.get("detout_nms_threshold", 0.45),
                'size_threshold': ssd_Param_2.get("detout_size_threshold", 0.0001),
                'top_k': ssd_Param_2.get("detout_top_k", 30),
                'share_location': True,
                'code_type': P.PriorBox.CENTER_SIZE,
                'background_label_id': 0,
                'variance_encoded_in_target': False,
            }
            use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes', False)
            if use_dense_boxes:
                net.detection_out_2 = L.DenseDetOut(*mbox_2_layers, \
                                                    detection_output_param=det_out_param, \
                                                    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
            else:
                net.detection_out_2 = L.DetOut(*mbox_2_layers, \
                                               detection_output_param=det_out_param, \
                                               include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    # EVAL in TEST MODE
    if not train:
        det_eval_param = {
            'gt_labels': eval_Param.get('eval_gt_labels', []),
            'num_classes': eval_Param.get("eval_num_classes", 2),
            'evaluate_difficult_gt': eval_Param.get("eval_difficult_gt", False),
            'boxsize_threshold': eval_Param.get("eval_boxsize_threshold", [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),
            'iou_threshold': eval_Param.get("eval_iou_threshold", [0.9, 0.75, 0.5]),
            'background_label_id': 0,
        }
        if use_ssd2_for_detection:
            det_out_layers = []
            if flag_train_withperson:
                det_out_layers.append(net['detection_out_1'])
            det_out_layers.append(net['detection_out_2'])
            name = 'det_out'
            net[name] = L.Concat(*det_out_layers, axis=2)
            net.det_accu = L.DetEval(net[name], net[gt_label], \
                                     detection_evaluate_param=det_eval_param, \
                                     include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        else:
            net.det_accu = L.DetEval(net['detection_out_1'], net[gt_label], \
                                     detection_evaluate_param=det_eval_param, \
                                     include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net



def FaceBoxAlikeNetFPN(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    lr = 1.0
    decay = 1.0
    from_layer = data_layer
    num_channels = [32,64]
    k_sizes = [7,3]
    strides = [2,2]
    pool_flags = [True,False]
    for i in xrange(len(num_channels)):
        add_layer = "conv{}".format(i+1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                    num_output=num_channels[i], kernel_size=k_sizes[i], pad=(k_sizes[i]-1)/2, stride=strides[i], use_scale=True,
                    n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
        if pool_flags[i]:
            add_layer = "pool{}".format(i+1)
            net[add_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
            from_layer = add_layer
    layer_cnt = len(num_channels)
    num_channels = [256,256,256]
    divide_scale = 4
    for i in xrange(len(num_channels)):
        n_chan = num_channels[i]
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        net = InceptionOfficialLayer(net, from_layer, add_layer, channels_1=n_chan/divide_scale, channels_3=[n_chan/4, n_chan/4],
                                     channels_5=[n_chan/4, n_chan/4], channels_ave=n_chan/divide_scale, inter_bn=True, leaky=False)
        from_layer = "conv{}_{}/incep".format(layer_cnt+1,i + 1)
    layer_cnt += 1
    num_channels = [128,128,128,128,128]
    for i in xrange(len(num_channels)):
        if i == 0:
            stride = 2
        else:
            stride = 1
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                        num_output=num_channels[i], kernel_size=3, pad=1, stride=stride,
                        use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
    layer_cnt += 1
    num_channels = [128, 128, 128,128,128]
    for i in xrange(len(num_channels)):
        if i == 0:
            stride = 2
        else:
            stride = 1
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                        num_output=num_channels[i], kernel_size=3, pad=1, stride=stride,
                        use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
    fpn_chann = 64

    #### 1/16
    from_layer = "conv5_5"
    Deconv(net, from_layer, num_output=fpn_chann, group=1, kernel_size=2, stride=2, lr_mult=lr, decay_mult=decay,
           use_bn=True, use_scale=True, use_relu=False)
    from_layer1 = from_layer + "_deconv"
    from_layer = "conv4_5"
    add_layer = from_layer + "_fpn1x1"
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=fpn_chann, kernel_size=1, pad=0, stride=1,
                    use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
    from_layer2 = add_layer
    add_layer = "conv5_5_fpn"
    net[add_layer] = L.Eltwise(net[from_layer1], net[from_layer2],
                               eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = add_layer
    add_layer = from_layer + "_relu"
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)

    from_layer = add_layer
    add_layer = "featuremap1/16"
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                    num_output=fpn_chann, kernel_size=3, pad=1, stride=1,
                    use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
    ######1/8
    from_layer = add_layer
    Deconv(net, from_layer, num_output=fpn_chann, group=1, kernel_size=2, stride=2, lr_mult=lr, decay_mult=decay,
           use_bn=True, use_scale=True, use_relu=False)
    from_layer1 = from_layer + "_deconv"
    from_layer = "conv3_3/incep"
    add_layer = from_layer + "_fpn1x1"
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=fpn_chann, kernel_size=1, pad=0, stride=1,
                    use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
    from_layer2 = add_layer
    add_layer = "conv3_3_fpn"
    net[add_layer] = L.Eltwise(net[from_layer1], net[from_layer2],
                               eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = add_layer
    add_layer = from_layer + "_relu"
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)

    from_layer = add_layer
    add_layer = "featuremap1/8"
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
                    num_output=fpn_chann, kernel_size=3, pad=1, stride=1,
                    use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
    # ######1/4
    # from_layer = add_layer
    # Deconv(net, from_layer, num_output=fpn_chann, group=1, kernel_size=2, stride=2, lr_mult=lr, decay_mult=decay,
    #        use_bn=True, use_scale=True, use_relu=False)
    # from_layer1 = from_layer + "_deconv"
    # from_layer = "conv3"
    # add_layer = from_layer + "_fpn1x1"
    # ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
    #                 num_output=fpn_chann, kernel_size=1, pad=0, stride=1,
    #                 use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
    # from_layer2 = add_layer
    # add_layer = "conv2_fpn"
    # net[add_layer] = L.Eltwise(net[from_layer1], net[from_layer2],
    #                            eltwise_param=dict(operation=P.Eltwise.SUM))
    # from_layer = add_layer
    # add_layer = from_layer + "_relu"
    # net[add_layer] = L.ReLU(net[from_layer], in_place=True)
    #
    # from_layer = add_layer
    # add_layer = "featuremap1/4"
    # ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=True, leaky=False,
    #                 num_output=fpn_chann, kernel_size=3, pad=1, stride=1,
    #                 use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)



    return net