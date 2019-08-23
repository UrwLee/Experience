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

# ##############################################################################
# ------------------------------------------------------------------------------
# Final Network
flag_train_withperson = True
flag_train_withhand = False
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

def MultiScaleEltLayer(net,layers = [],kernels =[], strides = [],out_layer = "",num_channels = 128,lr=1.0,decay=1.0,add_str = "",use_bn = True,flag_withparamname=False):
    assert len(layers) == len(kernels) == len(strides)
    feat_layers = []
    for i in xrange(len(layers)):
        f_layer = layers[i]
        o_layer = f_layer  + "_adap" + add_str
        k = kernels[i]
        ConvBNUnitLayer(net, f_layer, o_layer, use_bn=use_bn, use_relu=False,
                        num_output=num_channels, kernel_size=k, pad=(k-1)/2, stride=strides[i], use_scale=True, leaky=False, lr_mult=lr,
                        decay_mult=decay,flag_withparamname=flag_withparamname)
        feat_layers.append(net[o_layer])
    net[out_layer] = L.Eltwise(*feat_layers, eltwise_param=dict(operation=P.Eltwise.SUM))
    relu_name = out_layer + "_relu"
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)


def DAPNet(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    lr_basenet = 1.0
    # BaseNet
    use_sub_layers = (6, 7)
    num_channels = (144, 288)
    output_channels = (128, 0)
    channel_scale = 4
    add_strs = "_recon"
    net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                          output_channels=output_channels,channel_scale=channel_scale,lr=lr_basenet, decay=1, add_strs=add_strs,)
    if flag_train_withhand:
        use_bn = True
        from_layer = "pool1_recon"
        out_layer = 'conv2_hand'
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
                        num_output=32, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=False, lr_mult=1,
                        decay_mult=1)
        from_layer = "conv2_4_recon_relu"
        Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
               use_bn=True, use_scale=True,use_relu=True)
        feature_layers = []
        feature_layers.append(net["conv2_hand"])
        feature_layers.append(net["conv2_4_recon_relu_deconv"])
        add_layer = "featuremap0"
        net[add_layer] = L.Concat(*feature_layers, axis=1)

    lr_detnetperson = 1.0
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "conv3_7_recon_relu"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=True,lr_mult=lr_detnetperson, decay_mult=1,n_group=1)
    # Concat FM1 & FM2 & FM3 for Detection
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
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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
def DAPNetVGGReduce(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    flag_use_dark = False
    use_bn = False
    if flag_use_dark:
        net = YoloNetPart(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5, final_pool=True, lr=1,
                          decay=1)#1.47G, 13.7M
    else:
        # c = ((32,), (32,), (32, 32, 128), (64, 64, 128), (128, 128, 256))#1.74G, 19.43M
        # net = VGG16_BaseNet_ChangeChannel(net, "data", channels=c)
        # channels = ((32,), (32,), (64, 32, 128), (128, 64, 128, 64, 256), (256, 128, 256, 128, 256))
        channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
        strides = (True, True, True, False, False)
        kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
        pool_last = (False,False,False,True,True)
        net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last)


    lr_detnetperson = 1.0
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "pool5"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=lr_detnetperson, decay_mult=1,n_group=1)
    # Concat FM1 & FM2 & FM3 for Detection
    featuremap1 = ["pool2","pool3"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap1"
    UnifiedMultiScaleLayers(net,layers=featuremap1, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    # Concat FM2
    featuremap2 = ["pool3","pool4"]
    tags = ["Down","Ref"]
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
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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

def DAPNetVGGReduceNoConcat(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    flag_use_dark = False
    use_bn = False
    lr_mult = 1.0
    use_global_stats = None
    if flag_use_dark:
        net = YoloNetPart(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5, final_pool=True, lr=1,
                          decay=1)#1.47G, 13.7M
    else:
        # c = ((32,), (32,), (32, 32, 128), (64, 64, 128), (128, 128, 256))#1.74G, 19.43M
        # net = VGG16_BaseNet_ChangeChannel(net, "data", channels=c)
        # channels = ((32,), (32,), (64, 32, 128), (128, 64, 128, 64, 256), (256, 128, 256, 128, 256))
        channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
        strides = (True, True, True, False, False)
        kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
        pool_last = (False,False,False,True,True)
        net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last,lr_mult=lr_mult,decay_mult=lr_mult,
                                          use_global_stats = use_global_stats)


    lr_detnetperson = 1.0
    use_global_stats = None
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "pool5"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=lr_detnetperson, decay_mult=lr_detnetperson,n_group=1,use_global_stats=use_global_stats)

    # Create SSD Header for SSD1
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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
# Final Network
def DAPNet_hand_pool1(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    lr_basenet =0
    # BaseNet
    use_sub_layers = ()# exmpty means only has conv1 and pooling
    num_channels = ()
    output_channels = (0, )
    channel_scale = 4
    add_strs = "_recon"
    net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                                 output_channels=output_channels, channel_scale=channel_scale, lr=lr_basenet, decay=lr_basenet,
                                 add_strs=add_strs)
    print net.keys()
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
         # mbox_2_layers.append(net[data_layer])
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
                 'flag_areamaxcheckinmatch':ssd_Param_2.get("flag_areamaxcheckinmatch",True),
             }
             net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                     loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                     propagate_down=[True, True, False, False,False])
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

def DAPNetVGGReduceEltWise(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    flag_use_dark = False
    use_bn = False
    lr_mult = 0.1
    use_global_stats = None
    flag_withparamname = True
    if flag_use_dark:
        net = YoloNetPart(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5, final_pool=True, lr=1,
                          decay=1)#1.47G, 13.7M
    else:
        # c = ((32,), (32,), (32, 32, 128), (64, 64, 128), (128, 128, 256))#1.74G, 19.43M
        # net = VGG16_BaseNet_ChangeChannel(net, "data", channels=c)
        # channels = ((32,), (32,), (64, 32, 128), (128, 64, 128, 64, 256), (256, 128, 256, 128, 256))
        channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
        strides = (True, True, True, False, False)
        kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
        pool_last = (False,False,False,True,True)
        net = VGG16_BaseNet_ChangeChannel(net, from_layer=data_layer, channels=channels, strides=strides,use_bn=use_bn,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last,lr_mult=lr_mult,decay_mult=lr_mult,
                                          use_global_stats = use_global_stats,flag_withparamname=flag_withparamname)


    lr_detnetperson = 1.0
    use_global_stats = None
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    out_layer = "pool5"
    net = addconv6(net, from_layer=out_layer, use_bn=use_bn, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=lr_detnetperson, decay_mult=lr_detnetperson,n_group=1,use_global_stats=use_global_stats,flag_withparamname=flag_withparamname)
    layers = ["conv3_3","conv4_5"]
    kernels = [3,3]
    strides = [1,1]
    out_layer = "featuremap1"
    num_channels = 128
    add_str = "feat1"
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer, num_channels=num_channels, lr=1.0, decay=1.0,use_bn=use_bn,add_str=add_str,flag_withparamname=flag_withparamname)
    layers = ["conv4_5", "conv5_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap2"
    num_channels = 128
    add_str = "feat2"
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,use_bn=use_bn,add_str=add_str,flag_withparamname=flag_withparamname)
    layers = ["conv5_5", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    add_str = "feat3"
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,use_bn=use_bn,add_str=add_str,flag_withparamname=flag_withparamname)
    # Create SSD Header for SSD1
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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
def DAPNetResNetEltWise(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    lr_basenet = 1.0
    # BaseNet
    use_sub_layers = (6, 7)
    num_channels = (144, 288)
    output_channels = (0, 0)
    channel_scale = 4
    add_strs = "_recon"
    net = ResidualVariant_Base_B(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                                 output_channels=output_channels, channel_scale=channel_scale, lr=lr_basenet, decay=1,
                                 add_strs=add_strs, )


    lr_detnetperson = 1.0
    use_global_stats = None
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    from_layer = "conv3_7_recon_Add"
    net = addconv6(net, from_layer=from_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=True,lr_mult=lr_detnetperson, decay_mult=lr_detnetperson,n_group=1,use_global_stats=use_global_stats)

    layers = ["pool1_recon","conv2_6_recon_Add"]
    kernels = [3,3]
    strides = [2,1]
    out_layer = "featuremap1"
    num_channels = 128
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer, num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat1")
    layers = ["conv2_6_recon_Add", "conv3_7_recon_Add"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap2"
    num_channels = 256
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat2")
    layers = ["conv3_7_recon_Add", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat3")
    # # Create SSD Header for SSD1
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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

def DAPNetDarkPoseEltWise(net, train=True, data_layer="data", gt_label="label", \
           net_width=512, net_height=288):
    # BaseNet
    lr_basenet = 0.1
    # BaseNet
    strid_convs = [1, 1, 1, 0, 0]
    net = YoloNetPartCompressReduceConv5(net, use_layers=5, use_sub_layers=5, use_bn=True,strid_conv=strid_convs,lr=lr_basenet, decay=1.0)


    lr_detnetperson = 1.0
    use_global_stats = None
    # Add Conv6
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    from_layer = "conv5_5"
    net = addconv6(net, from_layer=from_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=True,lr_mult=lr_detnetperson, decay_mult=lr_detnetperson,n_group=1,use_global_stats=use_global_stats)

    layers = ["conv2","conv4_3"]
    kernels = [3,3]
    strides = [2,1]
    out_layer = "featuremap1"
    num_channels = 128
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer, num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat1")
    layers = ["conv4_3", "conv5_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap2"
    num_channels = 128
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat2")
    layers = ["conv5_5", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=1.0, decay=1.0,add_str="_feat3")
    # # Create SSD Header for SSD1
    if flag_train_withperson:
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
         stage=1,lr_mult=lr_detnetperson)
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