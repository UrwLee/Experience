# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, "/home/zhangming/work/minihand/remodet_repository/python")
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
from BaseNet import *
from DetectorHeader import *
from DAP_Param import *
def Deconv(net,from_layer,num_output,group,kernel_size,stride,lr_mult,decay_mult,use_bn,use_scale,use_relu):
    deconv_param = {
        'num_output': num_output,
        'kernel_size': kernel_size,
        'pad': 0,
        'stride': stride,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': True,
        'group': group,
        'bias_filler': dict(type='constant', value=0.0),
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult),dict(lr_mult=2*lr_mult, decay_mult=0)],
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
# ##############################################################################
# Network
def DAP_HandNet(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet: Only contains conv1 & pool1
    # lr_basenet =0
    # use_sub_layers = ()# exmpty means only has conv1 and pooling
    # num_channels = ()
    # output_channels = (0, )
    # channel_scale = 4
    # add_strs = "_recon"
    # net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
    #                              output_channels=output_channels, channel_scale=channel_scale, lr=lr_basenet, decay=lr_basenet,
    #                              add_strs=add_strs)
    # Base of ZhangM
    net = HandBase(net, data_layer=data_layer, use_bn=True)
    # make Loss & Detout for SSD2
    mbox_2_layers = SsdDetectorHeaders(net, \
          net_width=net_width, net_height=net_height, data_layer=data_layer, \
          from_layers=ssd_Param_2.get('feature_layers',[]), \
          num_classes=ssd_Param_2.get("num_classes",2), \
          boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
          aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios",[]), \
          prior_variance = ssd_Param_2.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
          flip=ssd_Param_2.get("anchor_flip",True), \
          clip=ssd_Param_2.get("anchor_clip",True), \
          normalizations=ssd_Param_2.get("interlayers_normalizations",[]), \
          use_batchnorm=ssd_Param_2.get("interlayers_use_batchnorm",True), \
          inter_layer_channels=ssd_Param_2.get("interlayers_channels_kernels",[]), \
          use_focus_loss=ssd_Param_2.get("bboxloss_using_focus_loss",False), \
          use_dense_boxes=ssd_Param_2.get('bboxloss_use_dense_boxes',False), \
          stage=2)
     # make Loss or Detout for SSD1
    if train:
         loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization",P.Loss.VALID))
         mbox_2_layers.append(net[gt_label])
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
         if use_dense_boxes:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'use_prior_for_matching':True,
                 'encode_variance_in_target': False,
                 'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                 'size_threshold_max':ssd_Param_2.get("bboxloss_size_threshold_max",2),
                 'flag_showdebug':ssd_Param_2.get("flag_showdebug",False),
                 'flag_forcematchallgt':ssd_Param_2.get("flag_forcematchallgt",False),
                 'flag_areamaxcheckinmatch':ssd_Param_2.get("flag_areamaxcheckinmatch",False),
             }
             net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                     loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                     propagate_down=[True, True, False, False])
         else:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'match_type':P.MultiBoxLoss.PER_PREDICTION,
                 'share_location':True,
                 'use_prior_for_matching':True,
                 'background_label_id':0,
                 'encode_variance_in_target': False,
                 'map_object_to_agnostic':False,
             }
             net["mbox_2_loss"] = L.BBoxLoss(*mbox_2_layers, bbox_loss_param=bboxloss_param, \
                         loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                         propagate_down=[True, True, False, False])
    else:
         if ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
             reshape_name = "mbox_2_conf_reshape"
             net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                     shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes",2)]))
             softmax_name = "mbox_2_conf_softmax"
             net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
             flatten_name = "mbox_2_conf_flatten"
             net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
             mbox_2_layers[1] = net[flatten_name]
         elif ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
             sigmoid_name = "mbox_2_conf_sigmoid"
             net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
             mbox_2_layers[1] = net[sigmoid_name]
         else:
             raise ValueError("Unknown conf loss type.")
         # Det-out param
         det_out_param = {
             'num_classes':ssd_Param_2.get("num_classes",2),
             'target_labels': ssd_Param_2.get('detout_target_labels',[]),
             'alias_id':ssd_Param_2.get("alias_id",0),
             'conf_threshold':ssd_Param_2.get("detout_conf_threshold",0.01),
             'nms_threshold':ssd_Param_2.get("detout_nms_threshold",0.45),
             'size_threshold':ssd_Param_2.get("detout_size_threshold",0.0001),
             'top_k':ssd_Param_2.get("detout_top_k",30),
             'share_location':True,
             'code_type':P.PriorBox.CENTER_SIZE,
             'background_label_id':0,
             'variance_encoded_in_target':False,
         }
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
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
            'gt_labels': eval_Param.get('eval_gt_labels',[]),
            'num_classes':eval_Param.get("eval_num_classes",2),
            'evaluate_difficult_gt':eval_Param.get("eval_difficult_gt",False),
            'boxsize_threshold':eval_Param.get("eval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]),
            'iou_threshold':eval_Param.get("eval_iou_threshold",[0.9,0.75,0.5]),
            'background_label_id':0,
        }
        net.det_accu = L.DetEval(net['detection_out_2'], net[gt_label], \
                	  detection_evaluate_param=det_eval_param, \
                	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net

# ##############################################################################
# Network
def DAP_HandNet_MultiScale(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet: Only contains conv1 & pool1
    # For ResNet_Start
    # lr_basenet =0
    # use_sub_layers = (6,)
    # num_channels = (144,)
    # output_channels = (128,)
    # channel_scale = 4
    # add_strs = "_recon"
    # net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
    #                              output_channels=output_channels, channel_scale=channel_scale, lr=lr_basenet, decay=lr_basenet,
    #                              add_strs=add_strs)
    # For ResNet_End
    use_bn = False
    lr_mult = 0
    use_global_stats = None
    channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192))
    strides = (True, True, True, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3))
    pool_last = (False, False, False, False)
    net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides, use_bn=use_bn,
                                      kernels=kernels, freeze_layers=[], pool_last=pool_last, lr_mult=lr_mult,
                                      decay_mult=lr_mult,
                                      use_global_stats=use_global_stats)
    use_bn = False
    init_xavier = False
    from_layer = "conv1"
    out_layer = 'conv2_hand'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=False, leaky=False, lr_mult=1,
                    decay_mult=1,init_xavier=init_xavier)

    from_layer = "conv4_5"
    Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
           use_bn=use_bn, use_scale=use_bn, use_relu=False)

    out_layer = "hand_multiscale"
    net[out_layer] = L.Eltwise(net["conv2_hand"], net["conv4_5_deconv"], eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = out_layer
    out_layer = from_layer + "_relu"
    net[out_layer] = L.ReLU(net[from_layer], in_place=True)
    # make Loss & Detout for SSD2
    mbox_2_layers = SsdDetectorHeaders(net, \
          net_width=net_width, net_height=net_height, data_layer=data_layer, \
          from_layers=ssd_Param_2.get('feature_layers',[]), \
          num_classes=ssd_Param_2.get("num_classes",2), \
          boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
          aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios",[]), \
          prior_variance = ssd_Param_2.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
          flip=ssd_Param_2.get("anchor_flip",True), \
          clip=ssd_Param_2.get("anchor_clip",True), \
          normalizations=ssd_Param_2.get("interlayers_normalizations",[]), \
          use_batchnorm=ssd_Param_2.get("interlayers_use_batchnorm",True), \
          inter_layer_channels=ssd_Param_2.get("interlayers_channels_kernels",[]), \
          use_focus_loss=ssd_Param_2.get("bboxloss_using_focus_loss",False), \
          use_dense_boxes=ssd_Param_2.get('bboxloss_use_dense_boxes',False), \
          stage=2)
     # make Loss or Detout for SSD1
    if train:
         loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization",P.Loss.VALID))
         mbox_2_layers.append(net[gt_label])
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
         if use_dense_boxes:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'use_prior_for_matching':True,
                 'encode_variance_in_target': False,
                 'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                 'size_threshold_max':ssd_Param_2.get("bboxloss_size_threshold_max",2),
                 'flag_showdebug':ssd_Param_2.get("flag_showdebug",False),
                 'flag_forcematchallgt':ssd_Param_2.get("flag_forcematchallgt",False),
                 'flag_areamaxcheckinmatch':ssd_Param_2.get("flag_areamaxcheckinmatch",False),
             }
             net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                     loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                     propagate_down=[True, True, False, False])
         else:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'match_type':P.MultiBoxLoss.PER_PREDICTION,
                 'share_location':True,
                 'use_prior_for_matching':True,
                 'background_label_id':0,
                 'encode_variance_in_target': False,
                 'map_object_to_agnostic':False,
             }
             net["mbox_2_loss"] = L.BBoxLoss(*mbox_2_layers, bbox_loss_param=bboxloss_param, \
                         loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                         propagate_down=[True, True, False, False])
    else:
         if ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
             reshape_name = "mbox_2_conf_reshape"
             net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                     shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes",2)]))
             softmax_name = "mbox_2_conf_softmax"
             net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
             flatten_name = "mbox_2_conf_flatten"
             net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
             mbox_2_layers[1] = net[flatten_name]
         elif ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
             sigmoid_name = "mbox_2_conf_sigmoid"
             net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
             mbox_2_layers[1] = net[sigmoid_name]
         else:
             raise ValueError("Unknown conf loss type.")
         # Det-out param
         det_out_param = {
             'num_classes':ssd_Param_2.get("num_classes",2),
             'target_labels': ssd_Param_2.get('detout_target_labels',[]),
             'alias_id':ssd_Param_2.get("alias_id",0),
             'conf_threshold':ssd_Param_2.get("detout_conf_threshold",0.01),
             'nms_threshold':ssd_Param_2.get("detout_nms_threshold",0.45),
             'size_threshold':ssd_Param_2.get("detout_size_threshold",0.0001),
             'top_k':ssd_Param_2.get("detout_top_k",30),
             'share_location':True,
             'code_type':P.PriorBox.CENTER_SIZE,
             'background_label_id':0,
             'variance_encoded_in_target':False,
         }
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
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
            'gt_labels': eval_Param.get('eval_gt_labels',[]),
            'num_classes':eval_Param.get("eval_num_classes",2),
            'evaluate_difficult_gt':eval_Param.get("eval_difficult_gt",False),
            'boxsize_threshold':eval_Param.get("eval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]),
            'iou_threshold':eval_Param.get("eval_iou_threshold",[0.9,0.75,0.5]),
            'background_label_id':0,
        }
        net.det_accu = L.DetEval(net['detection_out_2'], net[gt_label], \
                	  detection_evaluate_param=det_eval_param, \
                	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net

# ##############################################################################
# Network
def DAP_HandNet_MultiScale2_6_3_7Deconv(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet: Only contains conv1 & pool1

    lr_basenet =0
    use_sub_layers = (6,7)
    num_channels = (144,288)
    output_channels = (128,0)
    channel_scale = 4
    add_strs = "_recon"
    flag_with_deconv = True
    net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                                 output_channels=output_channels, channel_scale=channel_scale, lr=lr_basenet, decay=lr_basenet,
                                 flag_with_deconv=flag_with_deconv,add_strs=add_strs)
    use_bn = True
    from_layer = "conv1_recon"
    out_layer = 'conv2_hand'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=1,
                    decay_mult=1)
    if flag_with_deconv:
        from_layer = "conv3_{}{}_Add".format(use_sub_layers[1], add_strs) + "_deconv"
        feature_layers = []
        feature_layers.append(net["conv2_6_recon_relu"])
        feature_layers.append(net[from_layer])
        add_layer = "hand_concat"
        net[add_layer] = L.Concat(*feature_layers, axis=1)
        from_layer = add_layer
    else:
        from_layer = "conv2_6_recon_relu"
    Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
           use_bn=True, use_scale=True, use_relu=False)

    out_layer = "hand_multiscale"
    net[out_layer] = L.Eltwise(net["conv2_hand"], net[from_layer + "_deconv"], eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = out_layer
    out_layer = from_layer + "_relu"
    net[out_layer] = L.ReLU(net[from_layer], in_place=True)
    # make Loss & Detout for SSD2
    mbox_2_layers = SsdDetectorHeaders(net, \
          net_width=net_width, net_height=net_height, data_layer=data_layer, \
          from_layers=ssd_Param_2.get('feature_layers',[]), \
          num_classes=ssd_Param_2.get("num_classes",2), \
          boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
          aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios",[]), \
          prior_variance = ssd_Param_2.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
          flip=ssd_Param_2.get("anchor_flip",True), \
          clip=ssd_Param_2.get("anchor_clip",True), \
          normalizations=ssd_Param_2.get("interlayers_normalizations",[]), \
          use_batchnorm=ssd_Param_2.get("interlayers_use_batchnorm",True), \
          inter_layer_channels=ssd_Param_2.get("interlayers_channels_kernels",[]), \
          use_focus_loss=ssd_Param_2.get("bboxloss_using_focus_loss",False), \
          use_dense_boxes=ssd_Param_2.get('bboxloss_use_dense_boxes',False), \
          stage=2)
     # make Loss or Detout for SSD1
    if train:
         loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization",P.Loss.VALID))
         mbox_2_layers.append(net[gt_label])
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
         if use_dense_boxes:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'use_prior_for_matching':True,
                 'encode_variance_in_target': False,
                 'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                 'size_threshold_max':ssd_Param_2.get("bboxloss_size_threshold_max",2),
                 'flag_showdebug':ssd_Param_2.get("flag_showdebug",False),
                 'flag_forcematchallgt':ssd_Param_2.get("flag_forcematchallgt",False),
                 'flag_areamaxcheckinmatch':ssd_Param_2.get("flag_areamaxcheckinmatch",False),
             }
             net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                     loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                     propagate_down=[True, True, False, False])
         else:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'match_type':P.MultiBoxLoss.PER_PREDICTION,
                 'share_location':True,
                 'use_prior_for_matching':True,
                 'background_label_id':0,
                 'encode_variance_in_target': False,
                 'map_object_to_agnostic':False,
             }
             net["mbox_2_loss"] = L.BBoxLoss(*mbox_2_layers, bbox_loss_param=bboxloss_param, \
                         loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                         propagate_down=[True, True, False, False])
    else:
         if ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
             reshape_name = "mbox_2_conf_reshape"
             net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                     shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes",2)]))
             softmax_name = "mbox_2_conf_softmax"
             net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
             flatten_name = "mbox_2_conf_flatten"
             net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
             mbox_2_layers[1] = net[flatten_name]
         elif ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
             sigmoid_name = "mbox_2_conf_sigmoid"
             net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
             mbox_2_layers[1] = net[sigmoid_name]
         else:
             raise ValueError("Unknown conf loss type.")
         # Det-out param
         det_out_param = {
             'num_classes':ssd_Param_2.get("num_classes",2),
             'target_labels': ssd_Param_2.get('detout_target_labels',[]),
             'alias_id':ssd_Param_2.get("alias_id",0),
             'conf_threshold':ssd_Param_2.get("detout_conf_threshold",0.01),
             'nms_threshold':ssd_Param_2.get("detout_nms_threshold",0.45),
             'size_threshold':ssd_Param_2.get("detout_size_threshold",0.0001),
             'top_k':ssd_Param_2.get("detout_top_k",30),
             'share_location':True,
             'code_type':P.PriorBox.CENTER_SIZE,
             'background_label_id':0,
             'variance_encoded_in_target':False,
         }
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
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
            'gt_labels': eval_Param.get('eval_gt_labels',[]),
            'num_classes':eval_Param.get("eval_num_classes",2),
            'evaluate_difficult_gt':eval_Param.get("eval_difficult_gt",False),
            'boxsize_threshold':eval_Param.get("eval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]),
            'iou_threshold':eval_Param.get("eval_iou_threshold",[0.9,0.75,0.5]),
            'background_label_id':0,
        }
        net.det_accu = L.DetEval(net['detection_out_2'], net[gt_label], \
                	  detection_evaluate_param=det_eval_param, \
                	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net


# ##############################################################################
# Network
def HandNet_DarkBase(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    use_bn = False
    lr_mult = 0
    use_global_stats = None
    channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192))
    strides = (True, True, True, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3))
    pool_last = (False, False, False, True)
    net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides, use_bn=use_bn,
                                      kernels=kernels, freeze_layers=[], pool_last=pool_last, lr_mult=lr_mult,
                                      decay_mult=lr_mult,
                                      use_global_stats=use_global_stats)
    flag_with_deconv = True
    flag_eltwise = False
    from_layer = "conv4_5"
    if flag_with_deconv:
        Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
               use_bn=True, use_scale=True, use_relu=False)
    print net.keys()
    if flag_eltwise:
        use_bn = True
        from_layer = "conv1"
        out_layer = 'conv2_hand'
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
                        num_output=64, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=1,
                        decay_mult=1)

        out_layer = "hand_multiscale"
        net[out_layer] = L.Eltwise(net["conv2_hand"], net["conv4_3_deconv"],
                                   eltwise_param=dict(operation=P.Eltwise.SUM))
        from_layer = out_layer
        out_layer = from_layer + "_relu"
        net[out_layer] = L.ReLU(net[from_layer], in_place=True)

    # make Loss & Detout for SSD2
    mbox_2_layers = SsdDetectorHeaders(net, \
          net_width=net_width, net_height=net_height, data_layer=data_layer, \
          from_layers=ssd_Param_2.get('feature_layers',[]), \
          num_classes=ssd_Param_2.get("num_classes",2), \
          boxsizes=ssd_Param_2.get("anchor_boxsizes", []), \
          aspect_ratios=ssd_Param_2.get("anchor_aspect_ratios",[]), \
          prior_variance = ssd_Param_2.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
          flip=ssd_Param_2.get("anchor_flip",True), \
          clip=ssd_Param_2.get("anchor_clip",True), \
          normalizations=ssd_Param_2.get("interlayers_normalizations",[]), \
          use_batchnorm=ssd_Param_2.get("interlayers_use_batchnorm",True), \
          inter_layer_channels=ssd_Param_2.get("interlayers_channels_kernels",[]), \
          use_focus_loss=ssd_Param_2.get("bboxloss_using_focus_loss",False), \
          use_dense_boxes=ssd_Param_2.get('bboxloss_use_dense_boxes',False), \
          stage=2)
     # make Loss or Detout for SSD1
    if train:
         loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization",P.Loss.VALID))
         mbox_2_layers.append(net[gt_label])
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
         if use_dense_boxes:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'use_prior_for_matching':True,
                 'encode_variance_in_target': False,
                 'flag_noperson': ssd_Param_2.get('flag_noperson', False),
                 'size_threshold_max':ssd_Param_2.get("bboxloss_size_threshold_max",2),
                 'flag_showdebug':ssd_Param_2.get("flag_showdebug",False),
                 'flag_forcematchallgt':ssd_Param_2.get("flag_forcematchallgt",False),
                 'flag_areamaxcheckinmatch':ssd_Param_2.get("flag_areamaxcheckinmatch",False),
             }
             net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                     loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                     propagate_down=[True, True, False, False])
         else:
             bboxloss_param = {
                 'gt_labels': ssd_Param_2.get('gt_labels',[]),
                 'target_labels': ssd_Param_2.get('target_labels',[]),
                 'num_classes':ssd_Param_2.get("num_classes",2),
                 'alias_id':ssd_Param_2.get("alias_id",0),
                 'loc_loss_type':ssd_Param_2.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                 'conf_loss_type':ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                 'loc_weight':ssd_Param_2.get("bboxloss_loc_weight",1),
                 'conf_weight':ssd_Param_2.get("bboxloss_conf_weight",1),
                 'overlap_threshold':ssd_Param_2.get("bboxloss_overlap_threshold",0.5),
                 'neg_overlap':ssd_Param_2.get("bboxloss_neg_overlap",0.5),
                 'size_threshold':ssd_Param_2.get("bboxloss_size_threshold",0.0001),
                 'do_neg_mining':ssd_Param_2.get("bboxloss_do_neg_mining",True),
                 'neg_pos_ratio':ssd_Param_2.get("bboxloss_neg_pos_ratio",3),
                 'using_focus_loss':ssd_Param_2.get("bboxloss_using_focus_loss",False),
                 'gama':ssd_Param_2.get("bboxloss_focus_gama",2),
                 'use_difficult_gt':ssd_Param_2.get("bboxloss_use_difficult_gt",False),
                 'code_type':ssd_Param_2.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                 'match_type':P.MultiBoxLoss.PER_PREDICTION,
                 'share_location':True,
                 'use_prior_for_matching':True,
                 'background_label_id':0,
                 'encode_variance_in_target': False,
                 'map_object_to_agnostic':False,
             }
             net["mbox_2_loss"] = L.BBoxLoss(*mbox_2_layers, bbox_loss_param=bboxloss_param, \
                         loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                         propagate_down=[True, True, False, False])
    else:
         if ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
             reshape_name = "mbox_2_conf_reshape"
             net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                     shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes",2)]))
             softmax_name = "mbox_2_conf_softmax"
             net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
             flatten_name = "mbox_2_conf_flatten"
             net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
             mbox_2_layers[1] = net[flatten_name]
         elif ssd_Param_2.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
             sigmoid_name = "mbox_2_conf_sigmoid"
             net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
             mbox_2_layers[1] = net[sigmoid_name]
         else:
             raise ValueError("Unknown conf loss type.")
         # Det-out param
         det_out_param = {
             'num_classes':ssd_Param_2.get("num_classes",2),
             'target_labels': ssd_Param_2.get('detout_target_labels',[]),
             'alias_id':ssd_Param_2.get("alias_id",0),
             'conf_threshold':ssd_Param_2.get("detout_conf_threshold",0.01),
             'nms_threshold':ssd_Param_2.get("detout_nms_threshold",0.45),
             'size_threshold':ssd_Param_2.get("detout_size_threshold",0.0001),
             'top_k':ssd_Param_2.get("detout_top_k",30),
             'share_location':True,
             'code_type':P.PriorBox.CENTER_SIZE,
             'background_label_id':0,
             'variance_encoded_in_target':False,
         }
         use_dense_boxes = ssd_Param_2.get('bboxloss_use_dense_boxes',False)
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
            'gt_labels': eval_Param.get('eval_gt_labels',[]),
            'num_classes':eval_Param.get("eval_num_classes",2),
            'evaluate_difficult_gt':eval_Param.get("eval_difficult_gt",False),
            'boxsize_threshold':eval_Param.get("eval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]),
            'iou_threshold':eval_Param.get("eval_iou_threshold",[0.9,0.75,0.5]),
            'background_label_id':0,
        }
        net.det_accu = L.DetEval(net['detection_out_2'], net[gt_label], \
                	  detection_evaluate_param=det_eval_param, \
                	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net