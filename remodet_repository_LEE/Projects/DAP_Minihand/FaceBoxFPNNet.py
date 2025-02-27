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
from DetectorHeader import *
from DAP_Param import *
import numpy as np
#############################################################################
def getDecovArgs(num_output,lr=1,decay=1):
    return {
    	'param': [dict(lr_mult=lr, decay_mult=decay)],
    	'convolution_param': {
    		'num_output': num_output,
    		'kernel_size': 2,
    		'pad': 0,
    		'stride': 2,
    		'weight_filler': dict(type='xavier'),
    		'bias_term': True,
    		'group': 1,
    		'bias_filler': dict(type='constant', value=0.0),
    	}
    }
# ------------------------------------------------------------------------------
# Final Network
flag_train_withperson = True
def InceptionOfficialLayer(net, from_layer, out_layer, channels_1=1,channels_3=[],channels_5=[],channels_ave=1,inter_bn = True,leaky=False,lr=1,decay=1):
    fea_layer = from_layer

    concatlayers = []
    mid_layer = "{}/incep/1x1".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_1, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    concatlayers.append(net[mid_layer])
    start_layer = mid_layer
    mid_layer = "{}/incep/1_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True,num_output=channels_3[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    start_layer = mid_layer
    mid_layer = "{}/incep/3x3".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_3[1], kernel_size=3, pad=1,
                    stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/2_reduce".format(out_layer)
    ConvBNUnitLayer(net, fea_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[0], kernel_size=1, pad=0,
                    stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    start_layer = mid_layer
    mid_layer = "{}/incep/5x5".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_5[1], kernel_size=5, pad=2,
                    stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    concatlayers.append(net[mid_layer])

    mid_layer = "{}/incep/pool".format(out_layer)
    net[mid_layer] = L.Pooling(net[fea_layer], pool=P.Pooling.AVE, kernel_size=3, stride=1, pad=1)
    start_layer = mid_layer
    mid_layer = "{}/incep/pool_1x1".format(out_layer)
    ConvBNUnitLayer(net, start_layer, mid_layer, use_bn=inter_bn, use_relu=True, num_output=channels_ave, kernel_size=1,
                    pad=0,stride=1, use_scale=True, leaky=leaky,lr_mult=lr, decay_mult=decay)
    concatlayers.append(net[mid_layer])
    # incep
    layer_name = "{}/incep".format(out_layer)
    name = "{}/incep".format(out_layer)
    net[name] = L.Concat(*concatlayers, name=layer_name, axis=1)

    return net

def FaceBoxFPNNet(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    flag_handusefpn = False
    lr = 0
    decay = 0
    use_bn = False
    from_layer = data_layer
    num_channels = [32,64,128]
    k_sizes = [3,3,3]
    strides = [2,2,2]
    for i in xrange(len(num_channels)):
        add_layer = "conv{}".format(i+1)
        ConvBNUnitLayer(net, from_layer, add_layer, use_bn=use_bn, use_relu=True, leaky=False,
                    num_output=num_channels[i], kernel_size=k_sizes[i], pad=(k_sizes[i]-1)/2, stride=strides[i], use_scale=True,
                    n_group=1, lr_mult=lr, decay_mult=decay)
        from_layer = add_layer
        # if not i == len(num_channels) - 1:
        # add_layer = "pool{}".format(i+1)
        # net[add_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
        # from_layer = add_layer
    layer_cnt = len(num_channels)
    num_channels = [192,192,192,192]
    divide_scale = 4
    f4_depth = len(num_channels)
    for i in xrange(len(num_channels)):
        n_chan = num_channels[i]
        add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
        net = InceptionOfficialLayer(net, from_layer, add_layer, channels_1=n_chan/divide_scale, channels_3=[n_chan/8, n_chan/4],
                                     channels_5=[n_chan/8, n_chan/4], channels_ave=n_chan/divide_scale, inter_bn=use_bn, leaky=False,
                                     lr=lr,decay=decay)
        from_layer = "conv{}_{}/incep".format(layer_cnt+1,i + 1)

    if flag_handusefpn:
        layer_cnt += 1
        num_channels = [256,128,256,128,256]
        kernels      = [3,1,3,1,3]
        strides      = [2,1,1,1,1]
        f5_depth = len(num_channels)
        for i in xrange(len(num_channels)):
            add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
            ConvBNUnitLayer(net, from_layer, add_layer, use_bn=use_bn, use_relu=True, leaky=False,
                            num_output=num_channels[i], kernel_size=kernels[i], pad=kernels[i]/2, stride=strides[i],
                            use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
            from_layer = add_layer
        layer_cnt += 1
        num_channels = [128,128,128,128,128]
        kernels      = [3,3,3,3,3]
        strides      = [2,1,1,1,1]
        f6_depth = len(num_channels)
        for i in xrange(len(num_channels)):
            add_layer = "conv{}_{}".format(layer_cnt+1,i + 1)
            ConvBNUnitLayer(net, from_layer, add_layer, use_bn=use_bn, use_relu=True, leaky=False,
                            num_output=num_channels[i], kernel_size=kernels[i], pad=kernels[i]/2, stride=strides[i],
                            use_scale=True, n_group=1, lr_mult=lr, decay_mult=decay)
            from_layer = add_layer
        # ##########################################################################
        # Use FPN
        # f3 -> c6_4
        f3 = 'conv6_{}'.format(f6_depth)
        # f2: f3 -> deconv + c5_3 -> 1x1
        out_layer_1 = f3 + '_deconv'
        net[out_layer_1]=L.Deconvolution(net[f3],**(getDecovArgs(256,lr,decay)))
        f2 = 'conv5_{}'.format(f5_depth)
        out_layer_2 = f2 + '_1x1'
        ConvBNUnitLayer(net, f2, out_layer_2, use_bn=False, use_relu=False, num_output=256, kernel_size=1, pad=0, stride=1, lr_mult=lr,decay_mult=decay)
        out_layer = 'feat5'
        net[out_layer] = L.Eltwise(net[out_layer_2], net[out_layer_1], eltwise_param=dict(operation=P.Eltwise.SUM))
        net['feat5_relu'] = L.ReLU(net['feat5'], in_place=True)
        # f1: f2 -> deconv + c4_4 -> 1x1
        out_layer_1 = out_layer + '_deconv'
        net[out_layer_1]=L.Deconvolution(net['feat5_relu'],**(getDecovArgs(192,lr,decay)))
        f1 = 'conv4_{}/incep'.format(f4_depth)
        out_layer_2 = f1 + '_1x1'
        ConvBNUnitLayer(net, f1, out_layer_2, use_bn=False, use_relu=False, num_output=192, kernel_size=1, pad=0, stride=1, lr_mult=lr,decay_mult=decay)
        out_layer = 'feat4'
        net[out_layer] = L.Eltwise(net[out_layer_2], net[out_layer_1], eltwise_param=dict(operation=P.Eltwise.SUM))
        net['feat4_relu'] = L.ReLU(net['feat4'], in_place=True)
        from_layer = "feat4"
    add_layer = from_layer + "_deconv"
    net[add_layer] = L.Deconvolution(net[from_layer], **(getDecovArgs(64)))
    from_layer = add_layer
    add_layer  = from_layer + "_relu"
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
    print net.keys()
    # make Loss & Detout for SSD2
    mbox_2_layers = SsdDetectorHeaders(net, \
                                       net_width=net_width, net_height=net_height, data_layer=data_layer, \
                                       from_layers=ssd_Param_2.get('feature_layers', []), \
                                       num_classes=ssd_Param_2.get("num_classes", 2), \
                                       boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
                                       aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios", []), \
                                       prior_variance=ssd_Param_2.get("anchor_prior_variance", [0.1, 0.1, 0.2, 0.2]), \
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
                'gama': ssd_Param_2.get("bboxloss_focus_gama", 2),
                'use_difficult_gt': ssd_Param_2.get("bboxloss_use_difficult_gt", False),
                'code_type': ssd_Param_2.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
                'use_prior_for_matching': True,
                'encode_variance_in_target': False,
                'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                'size_threshold_max': ssd_Param_2.get("bboxloss_size_threshold_max", 2),
                'flag_showdebug': ssd_Param_2.get("flag_showdebug", False),
                'flag_forcematchallgt': ssd_Param_2.get("flag_forcematchallgt", False),
                'flag_areamaxcheckinmatch': ssd_Param_2.get("flag_areamaxcheckinmatch", False),
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
                'gama': ssd_Param_2.get("bboxloss_focus_gama", 2),
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
                                            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
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
        net.det_accu = L.DetEval(net['detection_out_2'], net[gt_label], \
                                 detection_evaluate_param=det_eval_param, \
                                 include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net
