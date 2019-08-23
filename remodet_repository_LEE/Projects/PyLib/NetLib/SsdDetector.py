d# -*- coding: utf-8 -*-
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
from GoogleNet import Google_IP_V3_Net
from PvaNet import PvaNet
from YoloNet import YoloNet

from MultiScaleLayer import *

from PyLib.LayerParam.MultiBoxLossLayerParam import *
from PyLib.LayerParam.MultiMcBoxLossLayerParam import *
from PyLib.LayerParam.DetectionOutLayerParam import *
from PyLib.LayerParam.DetectionEvalLayerParam import *

def AddSsdExtraConvLayers(net, use_batchnorm=True, feature_layers=[], add_layers=3, \
                          first_channels=256, second_channels=512):
    """
    创建SSD顶部的特征层。

    use_batchnorm: 是否使用BN层。
    feature_layers: 用于检测的特征层列表。
    """
    assert add_layers > 0

    use_relu = True

    # Add additional convolutional layers.
    last_layer = net.keys()[-1]
    from_layer = last_layer

    for i in xrange(1, add_layers+1):
      out_layer = "{}_extra_conv{}_1".format(last_layer, i)
      ConvBNUnitLayer(net, from_layer, out_layer, use_batchnorm, use_relu, first_channels, 1, 0, 1)
      from_layer = out_layer

      out_layer = "{}_extra_conv{}_2".format(last_layer, i)
      ConvBNUnitLayer(net, from_layer, out_layer, use_batchnorm, use_relu, second_channels, 3, 1, 2)
      from_layer = out_layer

      feature_layers.append(out_layer)

    # Add global pooling layer.
    name = net.keys()[-1]
    output_name = "pool_all"
    net[output_name] = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)
    feature_layers.append(output_name)

    return net, feature_layers

def SsdDetectorHeaders(net, \
              boxsizes=[], \
              min_ratio=15, max_ratio=90, \
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
    # assert len(from_layers) > 2, "Feature layers must be greater than 2."
    # step = int(math.floor((max_ratio - min_ratio) / (len(from_layers) - 2)))
    # min_sizes = []
    # max_sizes = []
    # # first layer
    # min_scale = net_width * min_ratio / 200.
    # max_scale = net_width * min_ratio / 100.
    # min_sizes.append([min_scale])
    # max_sizes.append([max_scale])
    # # last layers
    # for ratio in xrange(min_ratio, max_ratio + 1, step):
    #     min_scale = net_width * ratio / 100.
    #     max_scale = net_width * (ratio + step) / 100.
    #     min_sizes.append([min_scale])
    #     max_sizes.append([max_scale])

    pro_widths = [0.0308326,0.08135806,0.1165203,0.1981385,0.21726013,0.31657174,0.44452094,0.50671953,0.82039316]
    pro_heights = [0.06718339,0.17866732,0.42563996,0.61681194,0.29476581,0.73479042,0.42085125,0.81367556,0.84938404]
    # for i in range(len(boxsizes)):
    #   boxsizes_per_layer = boxsizes[i]
    #   pro_widths_per_layer = []
    #   pro_heights_per_layer = []
    #   for j in range(len(boxsizes_per_layer)):
    #     boxsize = boxsizes_per_layer[j]
    #     #print aspect_ratios
    #     aspect_ratio = aspect_ratios[0]
    #     if not len(aspect_ratios) == 1:
    #       aspect_ratio = aspect_ratios[i][j]
    #     for each_aspect_ratio in aspect_ratio:
    #         w = boxsize * math.sqrt(each_aspect_ratio)
    #         h = boxsize / math.sqrt(each_aspect_ratio)
    #         w = min(w,1.0)
    #         h = min(h,1.0)
    #         pro_widths_per_layer.append(w)
    #         pro_heights_per_layer.append(h)
    #   pro_widths.append(pro_widths_per_layer)
    #   pro_heights.append(pro_heights_per_layer)
    #print pro_widths
    #print pro_heights



    mbox_layers = MultiLayersDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                                            from_layers=from_layers, \
                                            normalizations=normalizations, \
                                            use_batchnorm=use_batchnorm, \
                                            prior_variance = prior_variance, \
                                            pro_widths=pro_widths, pro_heights=pro_heights, \
                                            aspect_ratios=aspect_ratios, \
                                            flip=flip, clip=clip, \
                                            inter_layer_channels=inter_layer_channels, \
                                            kernel_size=kernel_size, pad=pad)
    return mbox_layers

def SsdDetector(net, train=True, data_layer="data", gt_label="label", \
                net_width=300, net_height=300, basenet="VGG", \
                visualize=False, extra_data="data", eval_enable=True, **ssdparam):
    """
    创建SSD检测器。
    train: TRAIN /TEST
    data_layer/gt_label: 数据输入和label输入。
    net_width/net_height: 网络的输入尺寸
    num_classes: 估计分类的数量。
    basenet: "vgg"/"res101"，特征网络
    ssdparam: ssd检测器使用的参数列表。

    返回：整个SSD检测器网络。
    """
    # BaseNetWork
    if basenet == "VGG":
        net = VGG16Net(net, from_layer=data_layer, fully_conv=True, reduced=True, \
                dilated=True, dropout=False)
        base_feature_layers = ['conv4_3', 'fc7']
        add_layers = 3
        first_channels = 256
        second_channels = 512
    elif basenet == "Res101":
        net = ResNet101Net(net, from_layer=data_layer, use_pool5=False)
        # 1/8, 1/16, 1/32
        base_feature_layers = ['res3b3', 'res4b22', 'res5c']
        add_layers = 2
        first_channels = 256
        second_channels = 512
    elif basenet == "Res50":
        net = ResNet50Net(net, from_layer=data_layer, use_pool5=False)
        base_feature_layers = ['res3d', 'res4f', 'res5c']
        add_layers = 2
        first_channels = 256
        second_channels = 512
    elif basenet == "PVA":
        net = PvaNet(net, from_layer=data_layer)
        # 1/8, 1/16, 1/32
        base_feature_layers = ['conv4_1/incep/pre', 'conv5_1/incep/pre', 'conv5_4']
        add_layers = 2
        first_channels = 256
        second_channels = 512
    elif basenet == "Yolo":
        net = YoloNet(net, from_layer=data_layer)
        base_feature_layers = ssdparam.get("multilayers_feature_map",[])
        # add_layers = 2
        # first_channels = 256
        # second_channels = 512
        feature_layers = base_feature_layers

    else:
        raise ValueError("only VGG16, Res50/101 and PVANet are supported in current version.")

    result = []
    for item in feature_layers:
      if len(item) == 1:
        result.append(item[0])
        continue
      name = ""
      for layers in item:
        name += layers
      tags = ["Down","Ref"]
      down_methods = [["Reorg"]]
      UnifiedMultiScaleLayers(net,layers=item, tags=tags, \
                            unifiedlayer=name, dnsampleMethod=down_methods)
      result.append(name)
    feature_layers = result

    # Add extra layers
    # extralayers_use_batchnorm=True, extralayers_lr_mult=1, \
    # net, feature_layers = AddSsdExtraConvLayers(net, \
    #     use_batchnorm=ssdparam.get("extralayers_use_batchnorm",False), \
    #     feature_layers=base_feature_layers, add_layers=add_layers, \
    #     first_channels=first_channels, second_channels=second_channels)
    # create ssd detector deader
    mbox_layers = SsdDetectorHeaders(net, \
         min_ratio=ssdparam.get("multilayers_min_ratio",15), \
         max_ratio=ssdparam.get("multilayers_max_ratio",90), \
         boxsizes=ssdparam.get("multilayers_boxsizes", []), \
         net_width=net_width, \
         net_height=net_height, \
         data_layer=data_layer, \
         num_classes=ssdparam.get("num_classes",2), \
         from_layers=feature_layers, \
         use_batchnorm=ssdparam.get("multilayers_use_batchnorm",True), \
         prior_variance = ssdparam.get("multilayers_prior_variance",[0.1,0.1,0.2,0.2]), \
         normalizations=ssdparam.get("multilayers_normalizations",[]), \
         aspect_ratios=ssdparam.get("multilayers_aspect_ratios",[]), \
         flip=ssdparam.get("multilayers_flip",True), \
         clip=ssdparam.get("multilayers_clip",False), \
         inter_layer_channels=ssdparam.get("multilayers_inter_layer_channels",[]), \
         kernel_size=ssdparam.get("multilayers_kernel_size",3), \
         pad=ssdparam.get("multilayers_pad",1))
    if train == True:
        loss_param = get_loss_param(normalization=ssdparam.get("multiloss_normalization",P.Loss.VALID))
        mbox_layers.append(net[gt_label])
        # create loss
        if not ssdparam["combine_yolo_ssd"]:
          multiboxloss_param = get_multiboxloss_param( \
             loc_loss_type=ssdparam.get("multiloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1), \
             conf_loss_type=ssdparam.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX), \
             loc_weight=ssdparam.get("multiloss_loc_weight",1), \
             conf_weight=ssdparam.get("multiloss_conf_weight",1), \
             num_classes=ssdparam.get("num_classes",2), \
             share_location=ssdparam.get("multiloss_share_location",True), \
             match_type=ssdparam.get("multiloss_match_type",P.MultiBoxLoss.PER_PREDICTION), \
             overlap_threshold=ssdparam.get("multiloss_overlap_threshold",0.5), \
             use_prior_for_matching=ssdparam.get("multiloss_use_prior_for_matching",True), \
             background_label_id=ssdparam.get("multiloss_background_label_id",0), \
             use_difficult_gt=ssdparam.get("multiloss_use_difficult_gt",False), \
             do_neg_mining=ssdparam.get("multiloss_do_neg_mining",True), \
             neg_pos_ratio=ssdparam.get("multiloss_neg_pos_ratio",3), \
             neg_overlap=ssdparam.get("multiloss_neg_overlap",0.5), \
             code_type=ssdparam.get("multiloss_code_type",P.PriorBox.CENTER_SIZE), \
             encode_variance_in_target=ssdparam.get("multiloss_encode_variance_in_target",False), \
             map_object_to_agnostic=ssdparam.get("multiloss_map_object_to_agnostic",False), \
             name_to_label_file=ssdparam.get("multiloss_name_to_label_file",""))

          net["mbox_loss"] = L.MultiBoxLoss(*mbox_layers, \
                                            multibox_loss_param=multiboxloss_param, \
                                            loss_param=loss_param, \
                                            include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                            propagate_down=[True, True, False, False])
        else:
          multimcboxloss_param = get_multimcboxloss_param( \
             loc_loss_type=ssdparam.get("multiloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1), \
             loc_weight=ssdparam.get("multiloss_loc_weight",1), \
             conf_weight=ssdparam.get("multiloss_conf_weight",1), \
             num_classes=ssdparam.get("num_classes",2), \
             share_location=ssdparam.get("multiloss_share_location",True), \
             match_type=ssdparam.get("multiloss_match_type",P.MultiBoxLoss.PER_PREDICTION), \
             overlap_threshold=ssdparam.get("multiloss_overlap_threshold",0.5), \
             use_prior_for_matching=ssdparam.get("multiloss_use_prior_for_matching",True), \
             background_label_id=ssdparam.get("multiloss_background_label_id",0), \
             use_difficult_gt=ssdparam.get("multiloss_use_difficult_gt",False), \
             do_neg_mining=ssdparam.get("multiloss_do_neg_mining",True), \
             neg_pos_ratio=ssdparam.get("multiloss_neg_pos_ratio",3), \
             neg_overlap=ssdparam.get("multiloss_neg_overlap",0.5), \
             code_type=ssdparam.get("multiloss_code_type",P.PriorBox.CENTER_SIZE), \
             encode_variance_in_target=ssdparam.get("multiloss_encode_variance_in_target",False), \
             map_object_to_agnostic=ssdparam.get("multiloss_map_object_to_agnostic",False), \
             name_to_label_file=ssdparam.get("multiloss_name_to_label_file",""),\
             rescore=ssdparam.get("multiloss_rescore",True),\
             object_scale=ssdparam.get("multiloss_object_scale",1),\
             noobject_scale=ssdparam.get("multiloss_noobject_scale",1),\
             class_scale=ssdparam.get("multiloss_class_scale",1),\
             loc_scale=ssdparam.get("multiloss_loc_scale",1))
          net["mbox_loss"] = L.MultiMcBoxLoss(*mbox_layers, \
                                            multimcbox_loss_param=multimcboxloss_param, \
                                            loss_param=loss_param, \
                                            include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                            propagate_down=[True, True, False, False])

        return net
    else:
        # create conf softmax layer
        # mbox_layers[1]
        if not ssdparam["combine_yolo_ssd"]:
            if ssdparam.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
                reshape_name = "mbox_conf_reshape"
                net[reshape_name] = L.Reshape(mbox_layers[1], \
                        shape=dict(dim=[0, -1, ssdparam.get("num_classes",2)]))
                softmax_name = "mbox_conf_softmax"
                net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
                flatten_name = "mbox_conf_flatten"
                net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
                mbox_layers[1] = net[flatten_name]
            elif ssdparam.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
                sigmoid_name = "mbox_conf_sigmoid"
                net[sigmoid_name] = L.Sigmoid(mbox_layers[1])
                mbox_layers[1] = net[sigmoid_name]
            else:
                raise ValueError("Unknown conf loss type.")
        det_out_param = get_detection_out_param( \
            num_classes=ssdparam.get("num_classes",2), \
            share_location=ssdparam.get("multiloss_share_location",True), \
            background_label_id=ssdparam.get("multiloss_background_label_id",0), \
            code_type=ssdparam.get("multiloss_code_type",P.PriorBox.CENTER_SIZE), \
            variance_encoded_in_target=ssdparam.get("multiloss_encode_variance_in_target",False), \
            conf_threshold=ssdparam.get("detectionout_conf_threshold",0.01), \
            nms_threshold=ssdparam.get("detectionout_nms_threshold",0.45), \
            boxsize_threshold=ssdparam.get("detectionout_boxsize_threshold",0.001), \
            top_k=ssdparam.get("detectionout_top_k",30), \
            visualize=ssdparam.get("detectionout_visualize",False), \
            visual_conf_threshold=ssdparam.get("detectionout_visualize_conf_threshold", 0.5), \
            visual_size_threshold=ssdparam.get("detectionout_visualize_size_threshold", 0), \
            display_maxsize=ssdparam.get("detectionout_display_maxsize",1000), \
            line_width=ssdparam.get("detectionout_line_width",4), \
            color=ssdparam.get("detectionout_color",[[0,255,0],]))
        if visualize:
            mbox_layers.append(net[extra_data])
        if not ssdparam["combine_yolo_ssd"]:
            net.detection_out = L.DetectionOutput(*mbox_layers, \
    	  		detection_output_param=det_out_param, \
    	  		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        else:
            net.detection_out = L.DetectionMultiMcOutput(*mbox_layers, \
                detection_output_param=det_out_param, \
                include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        if not visualize and eval_enable:
            # create eval layer
            det_eval_param = get_detection_eval_param( \
                 num_classes=ssdparam.get("num_classes",2), \
                 background_label_id=ssdparam.get("multiloss_background_label_id",0), \
                 evaluate_difficult_gt=ssdparam.get("detectioneval_evaluate_difficult_gt",False), \
                 boxsize_threshold=ssdparam.get("detectioneval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]), \
                 iou_threshold=ssdparam.get("detectioneval_iou_threshold",[0.9,0.75,0.5]), \
                 name_size_file=ssdparam.get("detectioneval_name_size_file",""))
            net.detection_eval = L.DetectionEvaluate(net.detection_out, net[gt_label], \
            	  detection_evaluate_param=det_eval_param, \
            	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        if not eval_enable:
            net.slience = L.Silence(net.detection_out, ntop=0, \
                include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        return net
