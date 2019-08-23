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
from BaseNet import *
from AddC6 import *
from DetectorHeader import *
from DAP_Param import *
from DAPData import lr_basenet
# ##############################################################################
# ------------------------------------------------------------------------------
# Final Network
def DAPNet(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    use_sub_layers = (6, 7)
    num_channels = (144, 288)
    output_channels = (128, 0)
    channel_scale = 4
    add_strs = "_recon"
    flag_withparamname=True
    net = ResidualVariant_Base_A_base(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                          output_channels=output_channels,channel_scale=channel_scale,lr=lr_basenet, decay=1, add_strs=add_strs,flag_withparamname=flag_withparamname)
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "conv3_7_recon_relu"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1.0,n_group=1)
    featuremap1 = ["pool1_recon","conv2_6_recon_relu"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap1"
    UnifiedMultiScaleLayers(net,layers=featuremap1, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Concat FM2
    featuremap2 = ["conv2_6_recon_relu","conv3_7_recon_relu"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap2"
    UnifiedMultiScaleLayers(net,layers=featuremap2, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Concat FM3
    c6_layer = 'conv6_{}'.format(len(Conv6_Param['conv6_output']))
    featuremap3 = ["conv3_7_recon_relu",c6_layer]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap3"
    UnifiedMultiScaleLayers(net,layers=featuremap3, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Create SSD Header for SSD1
    lr_mult = 1
    decay_mult = 1.0
    mbox_1_layers = SsdDetectorHeaders(net, \
         net_width=net_width, net_height=net_height, data_layer=data_layer, \
         from_layers=ssd_Param_1.get('feature_layers',[]), \
         num_classes=ssd_Param_1.get("num_classes",2), \
         boxsizes=ssd_Param_1.get("anchor_boxsizes", []), \
         aspect_ratios=ssd_Param_1.get("anchor_aspect_ratios",[]), \
         prior_variance = ssd_Param_1.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
         flip=ssd_Param_1.get("anchor_flip",True), \
         clip=ssd_Param_1.get("anchor_clip",True), \
         normalizations=ssd_Param_1.get("interlayers_normalizations",[]), \
         use_batchnorm=ssd_Param_1.get("interlayers_use_batchnorm",True), \
         inter_layer_channels=ssd_Param_1.get("interlayers_channels_kernels",[]), \
         use_focus_loss=ssd_Param_1.get("bboxloss_using_focus_loss",False), \
         use_dense_boxes=ssd_Param_1.get('bboxloss_use_dense_boxes',False), \
         stage=1,lr_mult=lr_mult, decay_mult=decay_mult)
    # make Loss or Detout for SSD1
    if train:
        loss_param = get_loss_param(normalization=ssd_Param_1.get("bboxloss_normalization",P.Loss.VALID))
        mbox_1_layers.append(net[gt_label])
        use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes',False)
        if use_dense_boxes:
            bboxloss_param = {
                'gt_labels': ssd_Param_1.get('gt_labels',[]),
                'target_labels': ssd_Param_1.get('target_labels',[]),
                'num_classes':ssd_Param_1.get("num_classes",2),
                'alias_id':ssd_Param_1.get("alias_id",0),
                'loc_loss_type':ssd_Param_1.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                'conf_loss_type':ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                'loc_weight':ssd_Param_1.get("bboxloss_loc_weight",1),
                'conf_weight':ssd_Param_1.get("bboxloss_conf_weight",1),
                'overlap_threshold':ssd_Param_1.get("bboxloss_overlap_threshold",0.5),
                'neg_overlap':ssd_Param_1.get("bboxloss_neg_overlap",0.5),
                'size_threshold':ssd_Param_1.get("bboxloss_size_threshold",0.0001),
                'do_neg_mining':ssd_Param_1.get("bboxloss_do_neg_mining",True),
                'neg_pos_ratio':ssd_Param_1.get("bboxloss_neg_pos_ratio",3),
                'using_focus_loss':ssd_Param_1.get("bboxloss_using_focus_loss",False),
                'gama':ssd_Param_1.get("bboxloss_focus_gama",2),
                'use_difficult_gt':ssd_Param_1.get("bboxloss_use_difficult_gt",False),
                'code_type':ssd_Param_1.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                'use_prior_for_matching':True,
                'encode_variance_in_target': False,
                'flag_noperson':ssd_Param_1.get('flag_noperson',False),
            }
            net["mbox_1_loss"] = L.DenseBBoxLoss(*mbox_1_layers, dense_bbox_loss_param=bboxloss_param, \
                                    loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                    propagate_down=[True, True, False, False])
        else:
            bboxloss_param = {
                'gt_labels': ssd_Param_1.get('gt_labels',[]),
                'target_labels': ssd_Param_1.get('target_labels',[]),
                'num_classes':ssd_Param_1.get("num_classes",2),
                'alias_id':ssd_Param_1.get("alias_id",0),
                'loc_loss_type':ssd_Param_1.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                'conf_loss_type':ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                'loc_weight':ssd_Param_1.get("bboxloss_loc_weight",1),
                'conf_weight':ssd_Param_1.get("bboxloss_conf_weight",1),
                'overlap_threshold':ssd_Param_1.get("bboxloss_overlap_threshold",0.5),
                'neg_overlap':ssd_Param_1.get("bboxloss_neg_overlap",0.5),
                'size_threshold':ssd_Param_1.get("bboxloss_size_threshold",0.0001),
                'do_neg_mining':ssd_Param_1.get("bboxloss_do_neg_mining",True),
                'neg_pos_ratio':ssd_Param_1.get("bboxloss_neg_pos_ratio",3),
                'using_focus_loss':ssd_Param_1.get("bboxloss_using_focus_loss",False),
                'gama':ssd_Param_1.get("bboxloss_focus_gama",2),
                'use_difficult_gt':ssd_Param_1.get("bboxloss_use_difficult_gt",False),
                'code_type':ssd_Param_1.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                'match_type':P.MultiBoxLoss.PER_PREDICTION,
                'share_location':True,
                'use_prior_for_matching':True,
                'background_label_id':0,
                'encode_variance_in_target': False,
                'map_object_to_agnostic':False,
            }
            net["mbox_1_loss"] = L.BBoxLoss(*mbox_1_layers, bbox_loss_param=bboxloss_param, \
                        loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                        propagate_down=[True, True, False, False])
    else:
        if ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_1_conf_reshape"
            net[reshape_name] = L.Reshape(mbox_1_layers[1], \
                    shape=dict(dim=[0, -1, ssd_Param_1.get("num_classes",2)]))
            softmax_name = "mbox_1_conf_softmax"
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_1_conf_flatten"
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_1_layers[1] = net[flatten_name]
        elif ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_1_conf_sigmoid"
            net[sigmoid_name] = L.Sigmoid(mbox_1_layers[1])
            mbox_1_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
        # Det-out param
        det_out_param = {
            'num_classes':ssd_Param_1.get("num_classes",2),
            'target_labels': ssd_Param_1.get('detout_target_labels',[]),
            'alias_id':ssd_Param_1.get("alias_id",0),
            'conf_threshold':ssd_Param_1.get("detout_conf_threshold",0.01),
            'nms_threshold':ssd_Param_1.get("detout_nms_threshold",0.45),
            'size_threshold':ssd_Param_1.get("detout_size_threshold",0.0001),
            'top_k':ssd_Param_1.get("detout_top_k",30),
            'share_location':True,
            'code_type':P.PriorBox.CENTER_SIZE,
            'background_label_id':0,
            'variance_encoded_in_target':False,
        }
        use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes',False)
        if use_dense_boxes:
            net.detection_out_1 = L.DenseDetOut(*mbox_1_layers, \
        	  	detection_output_param=det_out_param, \
        	  	include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        else:
            net.detection_out_1 = L.DetOut(*mbox_1_layers, \
    	  		detection_output_param=det_out_param, \
    	  		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    # make Loss & Detout for SSD2
    lr_mult = 1.0
    decay_mult = 1.0
    if use_ssd2_for_detection:
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
              stage=2,lr_mult=lr_mult, decay_mult=decay_mult)
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
        if use_ssd2_for_detection:
            det_out_layers = []
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

# Final Network
def DAPNetVGGDark(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    flag_withparamname = True
    net = VGGDarkNet(net, data_layer="data", flag_withparamname=flag_withparamname)
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "pool5"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=1, decay_mult=1.0,n_group=1)
    featuremap1 = ["conv2","conv3_3"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap1"
    UnifiedMultiScaleLayers(net,layers=featuremap1, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Concat FM2
    featuremap2 = ["conv3_3","conv4_5"]
    tags = ["Ref","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap2"
    UnifiedMultiScaleLayers(net,layers=featuremap2, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Concat FM3
    c6_layer = 'conv6_{}'.format(len(Conv6_Param['conv6_output']))
    featuremap3 = ["pool5",c6_layer]
    tags = ["Ref","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap3"
    UnifiedMultiScaleLayers(net,layers=featuremap3, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Create SSD Header for SSD1
    lr_mult = 1
    decay_mult = 1.0
    mbox_1_layers = SsdDetectorHeaders(net, \
         net_width=net_width, net_height=net_height, data_layer=data_layer, \
         from_layers=ssd_Param_1.get('feature_layers',[]), \
         num_classes=ssd_Param_1.get("num_classes",2), \
         boxsizes=ssd_Param_1.get("anchor_boxsizes", []), \
         aspect_ratios=ssd_Param_1.get("anchor_aspect_ratios",[]), \
         prior_variance = ssd_Param_1.get("anchor_prior_variance",[0.1,0.1,0.2,0.2]), \
         flip=ssd_Param_1.get("anchor_flip",True), \
         clip=ssd_Param_1.get("anchor_clip",True), \
         normalizations=ssd_Param_1.get("interlayers_normalizations",[]), \
         use_batchnorm=ssd_Param_1.get("interlayers_use_batchnorm",True), \
         inter_layer_channels=ssd_Param_1.get("interlayers_channels_kernels",[]), \
         use_focus_loss=ssd_Param_1.get("bboxloss_using_focus_loss",False), \
         use_dense_boxes=ssd_Param_1.get('bboxloss_use_dense_boxes',False), \
         stage=1,lr_mult=lr_mult, decay_mult=decay_mult)
    # make Loss or Detout for SSD1
    if train:
        loss_param = get_loss_param(normalization=ssd_Param_1.get("bboxloss_normalization",P.Loss.VALID))
        mbox_1_layers.append(net[gt_label])
        use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes',False)
        if use_dense_boxes:
            bboxloss_param = {
                'gt_labels': ssd_Param_1.get('gt_labels',[]),
                'target_labels': ssd_Param_1.get('target_labels',[]),
                'num_classes':ssd_Param_1.get("num_classes",2),
                'alias_id':ssd_Param_1.get("alias_id",0),
                'loc_loss_type':ssd_Param_1.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                'conf_loss_type':ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
                'loc_weight':ssd_Param_1.get("bboxloss_loc_weight",1),
                'conf_weight':ssd_Param_1.get("bboxloss_conf_weight",1),
                'overlap_threshold':ssd_Param_1.get("bboxloss_overlap_threshold",0.5),
                'neg_overlap':ssd_Param_1.get("bboxloss_neg_overlap",0.5),
                'size_threshold':ssd_Param_1.get("bboxloss_size_threshold",0.0001),
                'do_neg_mining':ssd_Param_1.get("bboxloss_do_neg_mining",True),
                'neg_pos_ratio':ssd_Param_1.get("bboxloss_neg_pos_ratio",3),
                'using_focus_loss':ssd_Param_1.get("bboxloss_using_focus_loss",False),
                'gama':ssd_Param_1.get("bboxloss_focus_gama",2),
                'use_difficult_gt':ssd_Param_1.get("bboxloss_use_difficult_gt",False),
                'code_type':ssd_Param_1.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                'use_prior_for_matching':True,
                'encode_variance_in_target': False,
                'flag_noperson':ssd_Param_1.get('flag_noperson',False),
            }
            net["mbox_1_loss"] = L.DenseBBoxLoss(*mbox_1_layers, dense_bbox_loss_param=bboxloss_param, \
                                    loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                    propagate_down=[True, True, False, False])
        else:
            bboxloss_param = {
                'gt_labels': ssd_Param_1.get('gt_labels',[]),
                'target_labels': ssd_Param_1.get('target_labels',[]),
                'num_classes':ssd_Param_1.get("num_classes",2),
                'alias_id':ssd_Param_1.get("alias_id",0),
                'loc_loss_type':ssd_Param_1.get("bboxloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
                'conf_loss_type':ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX),
                'loc_weight':ssd_Param_1.get("bboxloss_loc_weight",1),
                'conf_weight':ssd_Param_1.get("bboxloss_conf_weight",1),
                'overlap_threshold':ssd_Param_1.get("bboxloss_overlap_threshold",0.5),
                'neg_overlap':ssd_Param_1.get("bboxloss_neg_overlap",0.5),
                'size_threshold':ssd_Param_1.get("bboxloss_size_threshold",0.0001),
                'do_neg_mining':ssd_Param_1.get("bboxloss_do_neg_mining",True),
                'neg_pos_ratio':ssd_Param_1.get("bboxloss_neg_pos_ratio",3),
                'using_focus_loss':ssd_Param_1.get("bboxloss_using_focus_loss",False),
                'gama':ssd_Param_1.get("bboxloss_focus_gama",2),
                'use_difficult_gt':ssd_Param_1.get("bboxloss_use_difficult_gt",False),
                'code_type':ssd_Param_1.get("bboxloss_code_type",P.PriorBox.CENTER_SIZE),
                'match_type':P.MultiBoxLoss.PER_PREDICTION,
                'share_location':True,
                'use_prior_for_matching':True,
                'background_label_id':0,
                'encode_variance_in_target': False,
                'map_object_to_agnostic':False,
            }
            net["mbox_1_loss"] = L.BBoxLoss(*mbox_1_layers, bbox_loss_param=bboxloss_param, \
                        loss_param=loss_param,include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                        propagate_down=[True, True, False, False])
    else:
        if ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_1_conf_reshape"
            net[reshape_name] = L.Reshape(mbox_1_layers[1], \
                    shape=dict(dim=[0, -1, ssd_Param_1.get("num_classes",2)]))
            softmax_name = "mbox_1_conf_softmax"
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_1_conf_flatten"
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_1_layers[1] = net[flatten_name]
        elif ssd_Param_1.get("bboxloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_1_conf_sigmoid"
            net[sigmoid_name] = L.Sigmoid(mbox_1_layers[1])
            mbox_1_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
        # Det-out param
        det_out_param = {
            'num_classes':ssd_Param_1.get("num_classes",2),
            'target_labels': ssd_Param_1.get('detout_target_labels',[]),
            'alias_id':ssd_Param_1.get("alias_id",0),
            'conf_threshold':ssd_Param_1.get("detout_conf_threshold",0.01),
            'nms_threshold':ssd_Param_1.get("detout_nms_threshold",0.45),
            'size_threshold':ssd_Param_1.get("detout_size_threshold",0.0001),
            'top_k':ssd_Param_1.get("detout_top_k",30),
            'share_location':True,
            'code_type':P.PriorBox.CENTER_SIZE,
            'background_label_id':0,
            'variance_encoded_in_target':False,
        }
        use_dense_boxes = ssd_Param_1.get('bboxloss_use_dense_boxes',False)
        if use_dense_boxes:
            net.detection_out_1 = L.DenseDetOut(*mbox_1_layers, \
        	  	detection_output_param=det_out_param, \
        	  	include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        else:
            net.detection_out_1 = L.DetOut(*mbox_1_layers, \
    	  		detection_output_param=det_out_param, \
    	  		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    # make Loss & Detout for SSD2
    lr_mult = 1.0
    decay_mult = 1.0
    if use_ssd2_for_detection:
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
              stage=2,lr_mult=lr_mult, decay_mult=decay_mult)
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
        if use_ssd2_for_detection:
            det_out_layers = []
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