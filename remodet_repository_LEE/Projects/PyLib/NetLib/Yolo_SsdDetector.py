# -*- coding: utf-8 -*-
import os
import sys
import caffe
import math

sys.dont_write_bytecode = True

sys.path.append('../')

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from DetectHeaderLayer import *
from ConvBNLayer import *

from VggNet import VGG16Net
from ResNet import ResNet101Net, ResNet152Net, ResNet50Net
from PvaNet import PvaNet
from YoloNet import YoloNet
from GoogleNet import Google_IP_V3_Net
from MultiScaleLayer import *

from PyLib.LayerParam.MultiBoxLossLayerParam import *
from PyLib.LayerParam.DetectionOutLayerParam import *
from PyLib.LayerParam.DetectionEvalLayerParam import *
from PyLib.LayerParam.McBoxLossLayerParam import *
from PyLib.LayerParam.DetectionMcOutLayerParam import *

# 在BaseNet顶层添加额外的卷积层
# 3/1/1卷积核： 尺度不变
def AddTopExtraConvLayers(net, use_pool=False, use_batchnorm=True, num_layers=0, channels=512, feature_layers=[]):
    # Add additional convolutional layers.
    last_layer = net.keys()[-1]
    from_layer = last_layer

    if use_pool:
        poolname = "{}_pool".format(last_layer)
        net[poolname] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2)
        from_layer = poolname

    use_relu = True
    if num_layers > 0:
        for i in range(0, num_layers):
            out_layer = "{}_extra_conv{}".format(last_layer, i+1)
            ConvBNUnitLayer(net, from_layer, out_layer, use_batchnorm, use_relu, channels, 3, 1, 1)
            from_layer = out_layer

    # 最后一层添加进层列表
    feature_layers.append(from_layer)

    return net, feature_layers

# 创建yolo检测器
def Yolo_SsdDetectorHeaders(net, \
              boxsizes=[], \
              net_width=300, net_height=300, \
              data_layer="data", num_classes=2, \
              from_layers=[], \
              use_batchnorm=True, \
              prior_variance = [0.1,0.1,0.2,0.2], \
              normalizations=[], \
              aspect_ratios=[], \
              flip=True, clip=False, \
              inter_layer_channels=[], \
              kernel_size=3,pad=1):
    """
    使用各个特征层创建SSD检测器。
    min_ratio: baseNet最后一层特征层的boxsize。
    max_ratio: 最后一层（GlobalPool）的boxsize。
    from_layers: 用于检测的特征层
    normalizations: 特征层是否先norm。
    inter_layer_channels: 特征层是否增加使用3/1/1的卷积中间层
    kernel_size/pad: 检测器的参数

    返回：
    [loc_layer]-> box估计
    [conf_layer]-> 分类估计
    [priorbox_layer]-> anchor位置
    [optional][objectness_layer]-> 是否有物体的分类估计
    """
    assert from_layers, "Feature layers must be provided."

    prior_width = [0.95]
    prior_height = [0.95]
    for sizeitem in boxsizes:
        basesize = sizeitem
        for ratioitem in aspect_ratios:
            w = basesize * math.sqrt(ratioitem)
            h = basesize / math.sqrt(ratioitem)
            #if clip:
            w = min(w,1.0)
            h = min(h,1.0)
            prior_width.append(w)
            prior_height.append(h)

    width_list = []
    height_list = []
    width_list.append(prior_width)
    height_list.append(prior_height)
    #print width_list
    #print height_list
    mbox_layers = MultiLayersDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                                            from_layers=from_layers, \
                                            normalizations=normalizations, \
                                            use_batchnorm=use_batchnorm, \
                                            prior_variance = prior_variance, \
                                            pro_widths=width_list, pro_heights=height_list, \
                                            aspect_ratios=aspect_ratios, \
                                            flip=flip, clip=clip, \
                                            inter_layer_channels=inter_layer_channels, \
                                            kernel_size=kernel_size, pad=pad)
    return mbox_layers


def Yolo_SsdDetector(net, train=True, data_layer="data", gt_label="label", \
                net_width=300, net_height=300, basenet="Res50",\
                visualize=False, extra_data="data", eval_enable=True, use_layers=2,**yolo_ssd_param):
    """
    创建YOLO检测器。
    train: TRAIN /TEST
    data_layer/gt_label: 数据输入和label输入。
    net_width/net_height: 网络的输入尺寸
    basenet: "vgg"/"res101"/"res50"/pva
    yoloparam: yolo检测器使用的参数列表。
    """
    # BaseNetWork
    # 构建基础网络，选择特征Layer
    final_layer_channels = 0
    if basenet == "VGG":
        net = VGG16Net(net, from_layer=data_layer, need_fc=False)
        final_layer_channels = 512
        # conv4_3 -> 1/8
        # conv5_3 -> 1/16
        if use_layers == 2:
            base_feature_layers = ['conv5_3']
        elif use_layers == 3:
            base_feature_layers = ['conv4_3', 'conv5_3']
        else:
            base_feature_layers = []
        # define added layers onto the top-layer
        add_layers = extra_top_layers
        add_channels = extra_top_depth
        if add_layers > 0:
            final_layer_channels = add_channels
        net, feature_layers = AddTopExtraConvLayers(net, use_pool=True, \
            use_batchnorm=True, num_layers=add_layers, channels=add_channels, \
            feature_layers=base_feature_layers)
    elif basenet == "Res101":
        net = ResNet101Net(net, from_layer=data_layer, use_pool5=False)
        final_layer_channels = 2048
        # res3b3-> 1/8
        # res4b22 -> 1/16
        # res5c -> 1/32
        if use_layers == 2:
            base_feature_layers = ['res4b22']
        elif use_layers == 3:
            base_feature_layers = ['res3b3', 'res4b22']
        else:
            base_feature_layers = []
        # define added layers onto the top-layer
        add_layers = extra_top_layers
        add_channels = extra_top_depth
        if add_layers > 0:
            final_layer_channels = add_channels
        net, feature_layers = AddTopExtraConvLayers(net, use_pool=False, \
            use_batchnorm=True, num_layers=add_layers, channels=add_channels, \
            feature_layers=base_feature_layers)
    elif basenet == "Res50":
        net = ResNet50Net(net, from_layer=data_layer, use_pool5=False)
        final_layer_channels = 2048
        # res3d-> 1/8
        # res4f -> 1/16
        # res5c -> 1/32
        if use_layers == 2:
            base_feature_layers = ['res4f']
        elif use_layers == 3:
            base_feature_layers = ['res3d', 'res4f']
        else:
            base_feature_layers = []
        # define added layers onto the top-layer
        add_layers = extra_top_layers
        add_channels = extra_top_depth
        if add_layers > 0:
            final_layer_channels = add_channels
        net, feature_layers = AddTopExtraConvLayers(net, use_pool=False, \
            use_batchnorm=True, num_layers=add_layers, channels=add_channels, \
            feature_layers=base_feature_layers)
    elif basenet == "PVA":
        net = PvaNet(net, from_layer=data_layer)
        final_layer_channels = 384
        if use_layers == 2:
            base_feature_layers = ['conv5_1/incep/pre', 'conv5_4']
        elif use_layers == 3:
            base_feature_layers = ['conv4_1/incep/pre', 'conv5_1/incep/pre', 'conv5_4']
        else:
            base_feature_layers = ['conv5_4']
        # Note: we do not add extra top layers for pvaNet
        feature_layers = base_feature_layers
    elif basenet == "Yolo":
        net = YoloNet(net, from_layer=data_layer)
        final_layer_channels = 1024
        if use_layers == 2:
            base_feature_layers = ['conv5_5', 'conv6_6']
        elif use_layers == 3:
            base_feature_layers = ['conv4_3','conv5_5', 'conv6_6']
        else:
            base_feature_layers = ['conv6_6']
        # Note: we do not add extra top layers for YoloNet
        feature_layers = base_feature_layers
    else:
        raise ValueError("only VGG16, Res50/101, PVA and Yolo are supported in current version.")

    # concat the feature_layers
    num_layers = len(feature_layers)
    if num_layers == 1:
        tags = ["Ref"]
    elif num_layers == 2:
        tags = ["Down","Ref"]
        down_methods = [["Reorg"]]
    else:
        if basenet == "Yolo":
            tags = ["Down","Down","Ref"]
            down_methods = [["MaxPool","Reorg"],["Reorg"]]          
        else: 
            tags = ["Down","Ref","Up"]
            down_methods = [["Reorg"]]
    # if use VGG, Norm may be used.
    # the interlayers can also be used if needed.
    # upsampleChannels must be the channels of Layers added onto the top.
    UnifiedMultiScaleLayers(net,layers=feature_layers, tags=tags, \
                            unifiedlayer="msfMap", dnsampleMethod=down_methods, \
                            upsampleMethod="Deconv", \
                            upsampleChannels=final_layer_channels)

    mbox_layers = Yolo_SsdDetectorHeaders(net, \
         boxsizes=yolo_ssd_param.get("multilayers_boxsizes", []), \
         net_width=net_width, \
         net_height=net_height, \
         data_layer=data_layer, \
         num_classes=yolo_ssd_param.get("num_classes",2), \
         from_layers=["msfMap"], \
         use_batchnorm=yolo_ssd_param.get("multilayers_use_batchnorm",True), \
         prior_variance = yolo_ssd_param.get("multilayers_prior_variance",[0.1,0.1,0.2,0.2]), \
         normalizations=yolo_ssd_param.get("multilayers_normalizations",[]), \
         aspect_ratios=yolo_ssd_param.get("multilayers_aspect_ratios",[]), \
         flip=yolo_ssd_param.get("multilayers_flip",False), \
         clip=yolo_ssd_param.get("multilayers_clip",False), \
         inter_layer_channels=yolo_ssd_param.get("multilayers_inter_layer_channels",[]), \
         kernel_size=yolo_ssd_param.get("multilayers_kernel_size",3), \
         pad=yolo_ssd_param.get("multilayers_pad",1))

    if train == True:
        # create loss
        multiboxloss_param = get_multiboxloss_param( \
           loc_loss_type=yolo_ssd_param.get("multiloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1), \
           conf_loss_type=yolo_ssd_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX), \
           loc_weight=yolo_ssd_param.get("multiloss_loc_weight",1), \
           conf_weight=yolo_ssd_param.get("multiloss_conf_weight",1), \
           num_classes=yolo_ssd_param.get("num_classes",2), \
           share_location=yolo_ssd_param.get("multiloss_share_location",True), \
           match_type=yolo_ssd_param.get("multiloss_match_type",P.MultiBoxLoss.PER_PREDICTION), \
           overlap_threshold=yolo_ssd_param.get("multiloss_overlap_threshold",0.5), \
           use_prior_for_matching=yolo_ssd_param.get("multiloss_use_prior_for_matching",True), \
           background_label_id=yolo_ssd_param.get("multiloss_background_label_id",0), \
           use_difficult_gt=yolo_ssd_param.get("multiloss_use_difficult_gt",False), \
           do_neg_mining=yolo_ssd_param.get("multiloss_do_neg_mining",True), \
           neg_pos_ratio=yolo_ssd_param.get("multiloss_neg_pos_ratio",3), \
           neg_overlap=yolo_ssd_param.get("multiloss_neg_overlap",0.5), \
           code_type=yolo_ssd_param.get("multiloss_code_type",P.PriorBox.CENTER_SIZE), \
           encode_variance_in_target=yolo_ssd_param.get("multiloss_encode_variance_in_target",False), \
           map_object_to_agnostic=yolo_ssd_param.get("multiloss_map_object_to_agnostic",False), \
           name_to_label_file=yolo_ssd_param.get("multiloss_name_to_label_file",""))
        loss_param = get_loss_param(normalization=yolo_ssd_param.get("multiloss_normalization",P.Loss.VALID))
        mbox_layers.append(net[gt_label])
        net["mbox_loss"] = L.MultiBoxLoss(*mbox_layers, \
                                          multibox_loss_param=multiboxloss_param, \
                                          loss_param=loss_param, \
                                          include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                          propagate_down=[True, True, False, False])
        return net
    else:
        # create conf softmax layer
        # mbox_layers[1]
        if yolo_ssd_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
            reshape_name = "mbox_conf_reshape"
            net[reshape_name] = L.Reshape(mbox_layers[1], \
                    shape=dict(dim=[0, -1, yolo_ssd_param.get("num_classes",2)]))
            softmax_name = "mbox_conf_softmax"
            net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
            flatten_name = "mbox_conf_flatten"
            net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
            mbox_layers[1] = net[flatten_name]
        elif yolo_ssd_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
            sigmoid_name = "mbox_conf_sigmoid"
            net[sigmoid_name] = L.Sigmoid(mbox_layers[1])
            mbox_layers[1] = net[sigmoid_name]
        else:
            raise ValueError("Unknown conf loss type.")
        det_out_param = get_detection_out_param( \
            num_classes=yolo_ssd_param.get("num_classes",2), \
            share_location=yolo_ssd_param.get("multiloss_share_location",True), \
            background_label_id=yolo_ssd_param.get("multiloss_background_label_id",0), \
            code_type=yolo_ssd_param.get("multiloss_code_type",P.PriorBox.CENTER_SIZE), \
            variance_encoded_in_target=yolo_ssd_param.get("multiloss_encode_variance_in_target",False), \
            conf_threshold=yolo_ssd_param.get("detectionout_conf_threshold",0.01), \
            nms_threshold=yolo_ssd_param.get("detectionout_nms_threshold",0.45), \
            boxsize_threshold=yolo_ssd_param.get("detectionout_boxsize_threshold",0.001), \
            top_k=yolo_ssd_param.get("detectionout_top_k",30), \
            visualize=yolo_ssd_param.get("detectionout_visualize",False), \
            visual_conf_threshold=yolo_ssd_param.get("detectionout_visualize_conf_threshold", 0.5), \
            visual_size_threshold=yolo_ssd_param.get("detectionout_visualize_size_threshold", 0), \
            display_maxsize=yolo_ssd_param.get("detectionout_display_maxsize",1000), \
            line_width=yolo_ssd_param.get("detectionout_line_width",4), \
            color=yolo_ssd_param.get("detectionout_color",[[0,255,0],]))
        if visualize:
            mbox_layers.append(net[extra_data])
            
        net.detection_out = L.DetectionOutput(*mbox_layers, \
            detection_output_param=det_out_param, \
            include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        if not visualize and eval_enable:
            # create eval layer
            det_eval_param = get_detection_eval_param( \
                 num_classes=yolo_ssd_param.get("num_classes",2), \
                 background_label_id=yolo_ssd_param.get("multiloss_background_label_id",0), \
                 evaluate_difficult_gt=yolo_ssd_param.get("detectioneval_evaluate_difficult_gt",False), \
                 boxsize_threshold=yolo_ssd_param.get("detectioneval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]), \
                 iou_threshold=yolo_ssd_param.get("detectioneval_iou_threshold",[0.9,0.75,0.5]), \
                 name_size_file=yolo_ssd_param.get("detectioneval_name_size_file",""))
            net.detection_eval = L.DetectionEvaluate(net.detection_out, net[gt_label], \
                  detection_evaluate_param=det_eval_param, \
                  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        if not eval_enable:
            net.slience = L.Silence(net.detection_out, ntop=0, \
                include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        return net
