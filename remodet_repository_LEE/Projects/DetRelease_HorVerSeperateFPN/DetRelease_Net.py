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
from AddC6 import *
from TPDUtils import *
from DetectorHeader import *
from DetNet_Param import *
from DetRelease_Data import *
from DetRelease_General import *
def Deconv(net,from_layer,num_output,group,kernel_size,stride,lr_mult,decay_mult,use_bn,use_scale,use_relu,add_str = "",deconv_name = "_Upsample"):
    deconv_param = {
        'num_output': num_output,
        'kernel_size': kernel_size,
        'pad': 0,
        'stride': stride,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0.0),
        'bias_term': True,
        'group': group,
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
        'convolution_param': deconv_param
    }
    out_layer = from_layer + deconv_name
    net[out_layer] = L.Deconvolution(net[from_layer + add_str], **kwargs_deconv)
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
        from_layer = relu_name
    return out_layer

def MultiScaleEltLayer(net,layers = [],kernels =[], strides = [],out_layer = "",num_channels = 128,lr=1.0,decay=1.0,add_str = "",use_bn = True,flag_withparamname=False):
    assert len(layers) == len(kernels) == len(strides)
    feat_layers = []
    for i in xrange(len(layers)):
        f_layer = layers[i]
        o_layer = f_layer  + "_adapfeat" + out_layer[-1]
        k = kernels[i]
        ConvBNUnitLayer(net, f_layer + add_str, o_layer, use_bn=use_bn, use_relu=False,
                        num_output=num_channels, kernel_size=k, pad=(k-1)/2, stride=strides[i], use_scale=True, leaky=False, lr_mult=lr,
                        decay_mult=decay,flag_withparamname=flag_withparamname,pose_string=add_str)
        feat_layers.append(net[o_layer + add_str])
    net[out_layer + add_str] = L.Eltwise(*feat_layers, eltwise_param=dict(operation=P.Eltwise.SUM))
    relu_name = out_layer + "_relu" + add_str
    net[relu_name] = L.ReLU(net[out_layer + add_str], in_place=True)
def DetRelease_FirstBodyPartPoseNet(train=True):

    ##Step1: Create Data for Body_Part Detection of 16:9, 9:16 and Pose Estimation
    ##Step2: Create BaseNet for three subnets until conv5_5
    ##Step3: Create Conv6 for Body_Part Detection for Detection subnets(16:9 and 9:16)
    ##Step4: Create featuremap1,featuremap2,featuremap3 for Detection subnet_16:9
    ##Step5: Create featuremap1,featuremap2,featuremap3 for Detection subnet_9:16
    ##Step6: Create Header and Body Loss for  subnet_16:9
    ##Step7: Create Header and Part Loss for  subnet_16:9
    ##Step8: Create Header and Body Loss for  subnet_9:16
    ##Step9: Create Header and Part Loss for  subnet_9:16
    ##Step10:Create Pose Estimation convf and stage loss
    net = caffe.NetSpec()
    ##Step1: Create Data for Body_Part Detection of 16:9, 9:16 and Pose Estimation
    net = get_DAPDataLayer(net, train=train, batchsize=batch_size,data_name = "data",label_name = "label",flag_169=flag_169_global)
    if train:
        net = get_poseDataLayer(net, train=train, batch_size=batch_size,data_name="data_pose", label_name="label_pose")
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net["label_pose"], ntop=4, slice_param=dict(slice_point=[34, 52, 86], axis=1))
        net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))

    ##Step2: Create BaseNet for three subnets until conv5_5
    use_bn = False
    channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
    strides = (True, True, True, False, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
    pool_last = (False,False,False,True,True)
    net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last,flag_withparamname=True,add_string='',
                                use_bn=use_bn,lr_mult=lr_conv1_conv5,decay_mult=1.0,use_global_stats=None)
    if train:
        pool_last = (False, False, False, True, False)
        net = VGG16_BaseNet_ChangeChannel(net, from_layer="data_pose", channels=channels, strides=strides,
                                          kernels=kernels, freeze_layers=[], pool_last=pool_last, flag_withparamname=True,
                                          add_string='_pose', use_bn=use_bn, lr_mult=lr_conv1_conv5, decay_mult=1.0,
                                          use_global_stats=None)

    ##Step3: Create Conv6 for Body_Part Detection for Detection subnets(16:9 and 9:16)
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    from_layer = "pool5"
    net = addconv6(net, from_layer=from_layer, use_bn=use_bn, conv6_output=conv6_output, \
                   conv6_kernal_size=conv6_kernal_size, pre_name="conv6", start_pool=False, lr_mult=lr_conv6_adap,
                   decay_mult=1, n_group=1, flag_withparamname=False)
    ##Step4:Create featuremap1,featuremap2,featuremap3 for Detection subnet_16:9
    #layers = ["conv3_3", "conv4_5"]
    #kernels = [3, 3]
    #strides = [1, 1]
    #out_layer = "featuremap1"
    #num_channels = 128
    #add_str = ""
    #MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
    #                   num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
    #                   flag_withparamname=False)
    #layers = ["conv4_5", "conv5_5"]
    #kernels = [3, 3]
    #strides = [2, 1]
    #out_layer = "featuremap2"
    #num_channels = 128
    #add_str = ""
    #MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
    #                   num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
    #                   flag_withparamname=False)
    layers = ["conv5_5", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=False)
    add_str = ""
    net,topdown16 = tdm(net,'conv5_5','conv6_5',2,freeze = False)
    net,topdown8 = tdm(net,'conv4_5','featuremap2',1,freeze = False)
    ##Step6:Create Header and Body Loss for  subnet_16:9
    data_layer = "data"
    gt_label = "label"
    if flag_169_global:
        net_width = 512
        net_height = 288
    else:
        net_width = 288
        net_height = 512
    ssd_Param_1 = get_ssd_Param_1(flag_169=flag_169_global,bboxloss_loc_weight = bboxloss_loc_weight_body,bboxloss_conf_weight=bboxloss_conf_weight_body)
    mbox_1_layers = SsdDetectorHeaders(net, \
                                       net_width=net_width, net_height=net_height, data_layer=data_layer, \
                                       from_layers=ssd_Param_1.get('feature_layers', []), \
                                       num_classes=ssd_Param_1.get("num_classes", 2), \
                                       boxsizes=ssd_Param_1.get("anchor_boxsizes", []), \
                                       aspect_ratios=ssd_Param_1.get("anchor_aspect_ratios", []), \
                                       prior_variance=ssd_Param_1.get("anchor_prior_variance", [0.1, 0.1, 0.2, 0.2]), \
                                       flip=ssd_Param_1.get("anchor_flip", True), \
                                       clip=ssd_Param_1.get("anchor_clip", True), \
                                       normalizations=ssd_Param_1.get("interlayers_normalizations", []), \
                                       use_batchnorm=ssd_Param_1.get("interlayers_use_batchnorm", True), \
                                       inter_layer_channels=ssd_Param_1.get("interlayers_channels_kernels", []), \
                                       use_focus_loss=ssd_Param_1.get("bboxloss_using_focus_loss", False), \
                                       use_dense_boxes=ssd_Param_1.get('bboxloss_use_dense_boxes', False), \
                                       stage=1, lr_mult=lr_inter_loss,flag_withparamname=False,add_str=add_str, AnchorFixed = AnchorFixed)
    ##Step7:Create Header and Part Loss for  subnet_16:9
    ssd_Param_2 = get_ssd_Param_2(flag_169=flag_169_global, bboxloss_loc_weight=bboxloss_loc_weight_part,bboxloss_conf_weight=bboxloss_conf_weight_part)
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
                                       stage=2, lr_mult=lr_inter_loss,flag_withparamname=False, add_str=add_str)

    if train:
        loss_param = get_loss_param(normalization=ssd_Param_1.get("bboxloss_normalization", P.Loss.VALID))
        mbox_1_layers.append(net[gt_label])

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
            'use_difficult_gt': ssd_Param_1.get("bboxloss_use_difficult_gt", False),
            'code_type': ssd_Param_1.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
            'flag_noperson':ssd_Param_1.get("flag_noperson", False),
            'match_type': P.MultiBoxLoss.PER_PREDICTION,
            'share_location': True,
            'use_prior_for_matching': True,
            'background_label_id': 0,
            'encode_variance_in_target': False,
            'map_object_to_agnostic': False,
            'matchtype_anchorgt':ssd_Param_1.get("matchtype_anchorgt", "REMOVELARGMARGIN"),
            'margin_ratio':ssd_Param_1.get("margin_ratio", 0.25),
            'sigma_angtdist':ssd_Param_1.get("sigma_angtdist", 0.1),
	    'only_w':ssd_Param_1.get("only_w",False)
        }
        if body_loss_type == "BBoxLoss":
            net["mbox_1_loss"] = L.BBoxLoss(*mbox_1_layers, bbox_loss_param=bboxloss_param, \
                                            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                            propagate_down=[True, True, False, False])
        else:
            net["mbox_1_loss"] = L.BBoxLossWTIOUCKCOVER(*mbox_1_layers, bbox_loss_param=bboxloss_param, \
                                            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                            propagate_down=[True, True, False, False])



        loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization", P.Loss.VALID))
        mbox_2_layers.append(net[gt_label])

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
        }
        net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                             loss_param=loss_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                             propagate_down=[True, True, False, False])

        ##Step10:Create Pose Estimation convf and stage loss
        from_layer = "conv5_5"
        add_str = "_pose"
        num_output = 128
        group = 1
        kernel_size = 2
        stride = 2
        use_bn = False
        use_scale = False
        use_relu = False
        out_layer1 = Deconv(net, from_layer, num_output, group, kernel_size, stride, lr_pose, 1.0, use_bn, use_scale,use_relu,add_str)
        from_layer = "conv4_5"
        out_layer2 = from_layer + "_adap"
        kernel_size = 3
        ConvBNUnitLayer(net, from_layer + add_str, out_layer2, use_bn=use_bn, use_relu=False,
                        num_output=num_output, kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1, use_scale=use_scale,
                        leaky=False, lr_mult=lr_pose,decay_mult=1.0)
        feat_layers = []
        feat_layers.append(net[out_layer1])
        feat_layers.append(net[out_layer2])
        out_layer = "convf"
        net[out_layer] = L.Eltwise(*feat_layers, eltwise_param=dict(operation=P.Eltwise.SUM))

        # relu_name = out_layer + "_relu"
        # net[relu_name] = L.ReLU(net[out_layer], in_place=True)
        use_stage = 3
        use_3_layers = 5
        use_1_layers = 0
        n_channel = 64

        kernel_size = 3
        baselayer = "convf"
        flag_output_sigmoid = False
        for stage in range(use_stage):
            if stage == 0:
                from_layer = baselayer
            else:
                from_layer = "concat_stage{}".format(stage)
            outlayer = "concat_stage{}".format(stage + 1)
            if stage == use_stage - 1:
                short_cut = False
            else:
                short_cut = True
            net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1,
                                     mask_vec="vec_mask", mask_heat="heat_mask", \
                                     label_vec="vec_label", label_heat="heat_label", \
                                     use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                                     base_layer=baselayer, lr=lr_pose, decay=1.0, num_channels=n_channel,
                                     kernel_size=kernel_size, flag_sigmoid=flag_output_sigmoid,loss_weight=0.1)
    else:

        if ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_1_conf_reshape" + add_str
            net[reshape_name] = L.Reshape(mbox_1_layers[1], \
                                          shape=dict(dim=[0, -1, ssd_Param_1.get("num_classes", 2)]))
            softmax_name = "mbox_1_conf_softmax" + add_str
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_1_conf_flatten" + add_str
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_1_layers[1] = net[flatten_name]
        elif ssd_Param_1.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_1_conf_sigmoid" + add_str
            net[sigmoid_name] = L.Sigmoid(mbox_1_layers[1])
            mbox_1_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
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
        net["detection_out_1" + add_str] = L.DetOut(*mbox_1_layers, \
                                                    detection_output_param=det_out_param, \
                                                    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        ##Step7:Create Part Header and sigmoid conf for subnet_16:9

        if ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_2_conf_reshape" + add_str
            net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                                          shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes", 2)]))
            softmax_name = "mbox_2_conf_softmax" + add_str
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_2_conf_flatten" + add_str
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_2_layers[1] = net[flatten_name]
        elif ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_2_conf_sigmoid" + add_str
            net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
            mbox_2_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
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

        net["detection_out_2" + add_str] = L.DenseDetOut(*mbox_2_layers, \
                                                         detection_output_param=det_out_param, \
                                                         include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        ##Step8:Create evaluation part for subnet_16:9
        eval_Param = get_eval_Param([0, 1, 3])
        det_eval_param = {
            'gt_labels': eval_Param.get('eval_gt_labels', []),
            'num_classes': eval_Param.get("eval_num_classes", 2),
            'evaluate_difficult_gt': eval_Param.get("eval_difficult_gt", False),
            'boxsize_threshold': eval_Param.get("eval_boxsize_threshold", [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),
            'iou_threshold': eval_Param.get("eval_iou_threshold", [0.9, 0.75, 0.5]),
            'background_label_id': 0,
        }
        det_out_layers = []
        det_out_layers.append(net['detection_out_1' + add_str])
        det_out_layers.append(net['detection_out_2' + add_str])
        name = 'det_out' + add_str
        net[name] = L.Concat(*det_out_layers, axis=2)
        net["det_accu" + add_str] = L.DetEval(net[name], net[gt_label], \
                                              detection_evaluate_param=det_eval_param, \
                                              include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    return net

def DetRelease_SecondPartAllNet(train=True):


    net = caffe.NetSpec()
    if train:
        ##Step1: Create Data for Body_Part Detection of 16:9, 9:16 and Pose Estimation
        net = get_DAPDataLayer(net, train=train, batchsize=batch_size,data_name = "data",label_name = "label",flag_169=flag_169_global)
        net = get_MinihandDataLayer(net, train=train, data_name="data_minihand", label_name="label_minihand", flag_169=flag_169_global)
    else:
        net = get_DAPDataLayer(net, train=train, batchsize=batch_size, data_name="data", label_name="label",flag_169=flag_169_global)
    ##Step2: Create BaseNet for three subnets until conv5_5
    lr_mult = 0.0
    decay_mult = 1.0
    use_bn = False
    channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
    strides = (True, True, True, False, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
    pool_last = (False,False,False,True,True)
    net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last,flag_withparamname=True,add_string='',
                                use_bn=use_bn,lr_mult=lr_conv1_conv5,decay_mult=1.0,use_global_stats=None)
    if train:
        channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192))
        strides = (True, True, True, False)
        kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3))
        pool_last = (False, False, False, False)
        net = VGG16_BaseNet_ChangeChannel(net, from_layer="data_minihand", channels=channels, strides=strides,
                                          kernels=kernels, freeze_layers=[], pool_last=pool_last, flag_withparamname=True,
                                          add_string='_minihand',use_bn=use_bn, lr_mult=lr_conv1_conv5, decay_mult=1.0, use_global_stats=None)
    ##Step3: Create Conv6 for Body_Part Detection for Detection subnets(16:9 and 9:16)
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    from_layer = "pool5"
    net = addconv6(net, from_layer=from_layer, use_bn=use_bn, conv6_output=conv6_output, \
                   conv6_kernal_size=conv6_kernal_size, pre_name="conv6", start_pool=False, lr_mult=lr_conv6_adap,
                   decay_mult=1, n_group=1, flag_withparamname=False)
    ##Step4:Create featuremap1,featuremap2,featuremap3 for Detection subnet_16:9
    layers = ["conv3_3", "conv4_5"]
    kernels = [3, 3]
    strides = [1, 1]
    out_layer = "featuremap1"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    layers = ["conv4_5", "conv5_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap2"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    layers = ["conv5_5", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    ##Step6:Create Header and Body Loss for  subnet_16:9
    add_str = ""
    data_layer = "data" + add_str

    if flag_169_global:
        net_width = 512
        net_height = 288
    else:
        net_width = 512
        net_height = 288

    ##Step7:Create Header and Part Loss for  subnet_16:9
    ssd_Param_2 = get_ssd_Param_2(flag_169=flag_169_global, bboxloss_loc_weight=bboxloss_loc_weight_part, bboxloss_conf_weight=bboxloss_conf_weight_part)
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
                                       stage=2,flag_withparamname=False,add_str=add_str,lr_mult=lr_inter_loss)

    use_bn = False
    init_xavier = False
    if train:
        add_str = "_minihand"
    else:
        add_str = ""
    from_layer = "conv1" + add_str
    out_layer = 'conv2_hand'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=False, leaky=False, lr_mult=1,
                    decay_mult=1, init_xavier=init_xavier)

    from_layer = "conv4_5"
    Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
           use_bn=use_bn, use_scale=use_bn, use_relu=False, add_str=add_str,deconv_name="_miniUpsample")

    out_layer = "mini_multiscale"
    net[out_layer] = L.Eltwise(net["conv2_hand"], net["conv4_5" + "_miniUpsample"],
                               eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = out_layer
    out_layer = from_layer + "_relu"
    net[out_layer] = L.ReLU(net[from_layer], in_place=True)

    data_layer = "data" + add_str
    ssd_Param_3 = get_ssd_Param_3(flag_169_global, bboxloss_loc_weight=bboxloss_loc_weight_part, bboxloss_conf_weight=bboxloss_conf_weight_part)
    mbox_3_layers = SsdDetectorHeaders(net, \
                                       net_width=net_width, net_height=net_height, data_layer=data_layer, \
                                       from_layers=ssd_Param_3.get('feature_layers', []), \
                                       num_classes=ssd_Param_3.get("num_classes", 2), \
                                       boxsizes=ssd_Param_3.get("anchor_boxsizes", []), \
                                       aspect_ratios=ssd_Param_3.get("anchor_aspect_ratios", []), \
                                       prior_variance=ssd_Param_3.get("anchor_prior_variance",
                                                                      [0.1, 0.1, 0.2, 0.2]), \
                                       flip=ssd_Param_3.get("anchor_flip", True), \
                                       clip=ssd_Param_3.get("anchor_clip", True), \
                                       normalizations=ssd_Param_3.get("interlayers_normalizations", []), \
                                       use_batchnorm=ssd_Param_3.get("interlayers_use_batchnorm", True), \
                                       inter_layer_channels=ssd_Param_3.get("interlayers_channels_kernels", []), \
                                       use_focus_loss=ssd_Param_3.get("bboxloss_using_focus_loss", False), \
                                       use_dense_boxes=ssd_Param_3.get('bboxloss_use_dense_boxes', False), \
                                       stage=3,lr_mult=lr_inter_loss)
    if train:
        gt_label = "label"
        loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization", P.Loss.VALID))
        mbox_2_layers.append(net[gt_label])

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
        }
        net["mbox_2_loss"] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                             loss_param=loss_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                             propagate_down=[True, True, False, False])

        gt_label = "label_minihand"
        loss_param = get_loss_param(normalization=ssd_Param_3.get("bboxloss_normalization", P.Loss.VALID))
        mbox_3_layers.append(net[gt_label])
        bboxloss_param = {
            'gt_labels': ssd_Param_3.get('gt_labels', []),
            'target_labels': ssd_Param_3.get('target_labels', []),
            'num_classes': ssd_Param_3.get("num_classes", 2),
            'alias_id': ssd_Param_3.get("alias_id", 0),
            'loc_loss_type': ssd_Param_3.get("bboxloss_loc_loss_type", P.MultiBoxLoss.SMOOTH_L1),
            'conf_loss_type': ssd_Param_3.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX),
            'loc_weight': ssd_Param_3.get("bboxloss_loc_weight", 1),
            'conf_weight': ssd_Param_3.get("bboxloss_conf_weight", 1),
            'overlap_threshold': ssd_Param_3.get("bboxloss_overlap_threshold", 0.5),
            'neg_overlap': ssd_Param_3.get("bboxloss_neg_overlap", 0.5),
            'size_threshold': ssd_Param_3.get("bboxloss_size_threshold", 0.0001),
            'do_neg_mining': ssd_Param_3.get("bboxloss_do_neg_mining", True),
            'neg_pos_ratio': ssd_Param_3.get("bboxloss_neg_pos_ratio", 3),
            'using_focus_loss': ssd_Param_3.get("bboxloss_using_focus_loss", False),
            'gama': ssd_Param_3.get("bboxloss_focus_gama", 2),
            'use_difficult_gt': ssd_Param_3.get("bboxloss_use_difficult_gt", False),
            'code_type': ssd_Param_3.get("bboxloss_code_type", P.PriorBox.CENTER_SIZE),
            'flag_noperson': ssd_Param_3.get('flag_noperson', False),
            'match_type': P.MultiBoxLoss.PER_PREDICTION,
            'share_location': True,
            'use_prior_for_matching': True,
            'background_label_id': 0,
            'encode_variance_in_target': False,
            'map_object_to_agnostic': False,
        }
        net["mbox_3_loss"] = L.BBoxLoss(*mbox_3_layers, bbox_loss_param=bboxloss_param, \
                                             loss_param=loss_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                             propagate_down=[True, True, False, False])

    else:

        if ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_2_conf_reshape" + add_str
            net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                                          shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes", 2)]))
            softmax_name = "mbox_2_conf_softmax" + add_str
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_2_conf_flatten" + add_str
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_2_layers[1] = net[flatten_name]
        elif ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_2_conf_sigmoid" + add_str
            net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
            mbox_2_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
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

        net["detection_out_2"] = L.DenseDetOut(*mbox_2_layers, \
                                                         detection_output_param=det_out_param, \
                                                         include=dict(phase=caffe_pb2.Phase.Value('TEST')))

        if ssd_Param_3.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_3_conf_reshape" + add_str
            net[reshape_name] = L.Reshape(mbox_3_layers[1], \
                                          shape=dict(dim=[0, -1, ssd_Param_3.get("num_classes", 2)]))
            softmax_name = "mbox_3_conf_softmax" + add_str
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_3_conf_flatten" + add_str
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_3_layers[1] = net[flatten_name]
        elif ssd_Param_3.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_3_conf_sigmoid" + add_str
            net[sigmoid_name] = L.Sigmoid(mbox_3_layers[1])
            mbox_3_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
        det_out_param = {
            'num_classes': ssd_Param_3.get("num_classes", 2),
            'target_labels': ssd_Param_3.get('detout_target_labels', []),
            'alias_id': ssd_Param_3.get("alias_id", 0),
            'conf_threshold': ssd_Param_3.get("detout_conf_threshold", 0.01),
            'nms_threshold': ssd_Param_3.get("detout_nms_threshold", 0.45),
            'size_threshold': ssd_Param_3.get("detout_size_threshold", 0.0001),
            'top_k': ssd_Param_3.get("detout_top_k", 30),
            'share_location': True,
            'code_type': P.PriorBox.CENTER_SIZE,
            'background_label_id': 0,
            'variance_encoded_in_target': False,
        }

        net["detection_out_3"] = L.DenseDetOut(*mbox_3_layers, \
                                                         detection_output_param=det_out_param, \
                                                         include=dict(phase=caffe_pb2.Phase.Value('TEST')))

        ##Step8:Create evaluation part for subnet_16:9
        eval_Param = get_eval_Param([1, 3])
        det_eval_param = {
            'gt_labels': eval_Param.get('eval_gt_labels', []),
            'num_classes': eval_Param.get("eval_num_classes", 2),
            'evaluate_difficult_gt': eval_Param.get("eval_difficult_gt", False),
            'boxsize_threshold': eval_Param.get("eval_boxsize_threshold", [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),
            'iou_threshold': eval_Param.get("eval_iou_threshold", [0.9, 0.75, 0.5]),
            'background_label_id': 0,
        }
        det_out_layers = []
        det_out_layers.append(net['detection_out_2' + add_str])
        det_out_layers.append(net['detection_out_3' + add_str])
        name = 'det_out' + add_str
        net[name] = L.Concat(*det_out_layers, axis=2)
        net["det_accu" + add_str] = L.DetEval(net[name], net["label"], \
                                               detection_evaluate_param=det_eval_param, \
                                               include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net

def DetRelease_SecondPartAllNetMiniHandFace(train=True):


    net = caffe.NetSpec()
    ##Step1: Create Data for Body_Part Detection of 16:9, 9:16 and Pose Estimation
    if train:
        net = get_DAPDataLayer(net, train=train, batchsize=batch_size, data_name="data_pd", label_name="label_pd",
                               flag_169=flag_169_global)
        net = get_MinihandDataLayer(net, train=train, data_name="data_minihand", label_name="label_minihand", flag_169=flag_169_global)

        data = []
        data.append(net["data_minihand"])
        data.append(net["data_pd"])
        net["data"] = L.Concat(*data, axis=0)
        label = []
        label.append(net["label_minihand"])
        label.append(net["label_pd"])
        net["label"] = L.Concat(*label, axis=2)
    else:
        net = get_DAPDataLayer(net, train=train, batchsize=batch_size, data_name="data", label_name="label",flag_169=flag_169_global)

    ##Step2: Create BaseNet for three subnets until conv5_5
    use_bn = False
    channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
    strides = (True, True, True, False, False)
    kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
    pool_last = (False,False,False,True,True)
    net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
                                          kernels=kernels,freeze_layers=[], pool_last=pool_last,flag_withparamname=True,add_string='',
                                use_bn=use_bn,lr_mult=lr_conv1_conv5,decay_mult=1.0,use_global_stats=None)
    ##Step3: Create Conv6 for Body_Part Detection for Detection subnets(16:9 and 9:16)
    conv6_output = Conv6_Param.get('conv6_output',[])
    conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
    from_layer = "pool5"
    net = addconv6(net, from_layer=from_layer, use_bn=use_bn, conv6_output=conv6_output, \
                   conv6_kernal_size=conv6_kernal_size, pre_name="conv6", start_pool=False, lr_mult=lr_conv6_adap,
                   decay_mult=1, n_group=1, flag_withparamname=False)
    ##Step4:Create featuremap1,featuremap2,featuremap3 for Detection subnet_16:9
    layers = ["conv3_3", "conv4_5"]
    kernels = [3, 3]
    strides = [1, 1]
    out_layer = "featuremap1"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    layers = ["conv4_5", "conv5_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap2"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    layers = ["conv5_5", "conv6_5"]
    kernels = [3, 3]
    strides = [2, 1]
    out_layer = "featuremap3"
    num_channels = 128
    add_str = ""
    MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
                       num_channels=num_channels, lr=lr_conv6_adap, decay=1.0, use_bn=use_bn, add_str=add_str,
                       flag_withparamname=True)
    use_bn = False
    init_xavier = False
    from_layer = "conv1"
    out_layer = 'conv2_mini'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=False, leaky=False, lr_mult=1,
                    decay_mult=1, init_xavier=init_xavier)

    from_layer = "conv4_5"
    Deconv(net, from_layer, num_output=64, group=1, kernel_size=2, stride=2, lr_mult=1.0, decay_mult=1.0,
           use_bn=use_bn, use_scale=use_bn, use_relu=False, add_str="", deconv_name="_miniUpsample")

    out_layer = "mini_multiscale"
    net[out_layer] = L.Eltwise(net["conv2_mini"], net["conv4_5" + "_miniUpsample"],
                               eltwise_param=dict(operation=P.Eltwise.SUM))
    from_layer = out_layer
    out_layer = from_layer + "_relu"
    net[out_layer] = L.ReLU(net[from_layer], in_place=True)
    ##Step6:Create Header and Body Loss for  subnet_16:9
    data_layer = "data"
    gt_label = "label"
    if flag_169_global:
        net_width = 512
        net_height = 288
    else:
        net_width = 512
        net_height = 288

    ##Step7:Create Header and Part Loss for  subnet_16:9
    ssd_Param_2 = get_ssd_Param_4(flag_169=flag_169_global, bboxloss_loc_weight=bboxloss_loc_weight_part, bboxloss_conf_weight=bboxloss_conf_weight_part)
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
                                       stage=2,flag_withparamname=False,add_str=add_str,lr_mult=lr_inter_loss)


    if train:

        loss_param = get_loss_param(normalization=ssd_Param_2.get("bboxloss_normalization", P.Loss.VALID))
        mbox_2_layers.append(net[gt_label])

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
        }
        net["mbox_2_loss" + add_str] = L.DenseBBoxLoss(*mbox_2_layers, dense_bbox_loss_param=bboxloss_param, \
                                             loss_param=loss_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                             propagate_down=[True, True, False, False])


    else:

        if ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_2_conf_reshape" + add_str
            net[reshape_name] = L.Reshape(mbox_2_layers[1], \
                                          shape=dict(dim=[0, -1, ssd_Param_2.get("num_classes", 2)]))
            softmax_name = "mbox_2_conf_softmax" + add_str
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_2_conf_flatten" + add_str
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_2_layers[1] = net[flatten_name]
        elif ssd_Param_2.get("bboxloss_conf_loss_type", P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_2_conf_sigmoid" + add_str
            net[sigmoid_name] = L.Sigmoid(mbox_2_layers[1])
            mbox_2_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
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

        net["detection_out_2"] = L.DenseDetOut(*mbox_2_layers, \
                                                         detection_output_param=det_out_param, \
                                                         include=dict(phase=caffe_pb2.Phase.Value('TEST')))

        ##Step8:Create evaluation part for subnet_16:9
        eval_Param = get_eval_Param([1, 3])
        det_eval_param = {
            'gt_labels': eval_Param.get('eval_gt_labels', []),
            'num_classes': eval_Param.get("eval_num_classes", 2),
            'evaluate_difficult_gt': eval_Param.get("eval_difficult_gt", False),
            'boxsize_threshold': eval_Param.get("eval_boxsize_threshold", [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),
            'iou_threshold': eval_Param.get("eval_iou_threshold", [0.9, 0.75, 0.5]),
            'background_label_id': 0,
        }
        net["det_accu"] = L.DetEval(net["detection_out_2"], net[gt_label], \
                                               detection_evaluate_param=det_eval_param, \
                                               include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net
def mPose_StageX_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True,base_layer="convf", lr=1.0, decay=1.0,num_channels = 128,flag_sigmoid = False,
                       kernel_size=3,addstrs = '',flag_change_layer=False,flag_hasoutput=True,flag_hasloss=True,id_layer_until=0, relu_layer_until = False,loss_weight=1.0):
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
            net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=loss_weight)
            net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
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
