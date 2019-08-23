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
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.InceptionLayer import *
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.DetectHeaderLayer import *
from PyLib.LayerParam.MultiBoxLossLayerParam import *
# ResNet-Unit
def ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel,bridge = False,num_channel_change = 0,
                     flag_hasresid = True,channel_scale = 4,check_macc = False,lr=1,decay=1):
    add_layer = name_prefix + '_1x1Conv'
    ConvBNUnitLayer(net, base_layer, add_layer, use_bn=True, use_relu=True,
					num_output=num_channel/channel_scale, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=False,lr_mult=lr,
                    decay_mult=decay)
    from_layer = add_layer
    add_layer = name_prefix + '_3x3Conv'
    if num_channel_change != 0:
        num_channel = num_channel_change
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=num_channel, kernel_size=3, pad=1, stride=stride, use_scale=True,
                    n_group=1,lr_mult=lr,
                    decay_mult=decay)
    if flag_hasresid:
        from_layer = add_layer
        if stride == 2:
            feature_layers = []
            feature_layers.append(net[from_layer])
            add_layer = name_prefix + '_AVEPool'
            net[add_layer] = L.Pooling(net[base_layer], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)
            feature_layers.append(net[add_layer])
            add_layer = name_prefix + '_Concat'
            net[add_layer] = L.Concat(*feature_layers, axis=1)
        else:
            add_layer1 = from_layer
            if bridge:
                from_layer = base_layer
                add_layer = name_prefix + '_bridge'
                ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                                num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,lr_mult=lr,
                    decay_mult=decay)
                add_layer2 = add_layer
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
# ResNet
def ResidualVariant_Base_A(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,num_channel_deconv = 128,lr=1,decay=1,add_strs=""):
    out_layer = 'conv1' + add_strs
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=lr,
                    decay_mult=decay)
    from_layer = out_layer
    out_layer = 'pool1' + add_strs
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    for layer in xrange(0, len(use_sub_layers)):
        num_channel_layer = num_channels[layer]
        output_channel_layer = output_channels[layer]
        for sublayer in xrange(use_sub_layers[layer]):
            base_layer = out_layer
            name_prefix = 'conv{}_{}'.format(layer + 2, sublayer + 1) + add_strs
            if sublayer == 0:
                stride = 2
            else:
                stride = 1
            if sublayer == 1:
                bridge = True
            else:
                bridge = False
            if not output_channel_layer == 0 and sublayer == use_sub_layers[layer] - 1:
                num_channel_change = output_channel_layer
                bridge = True
            else:
                num_channel_change = 0
            ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel_layer, bridge=bridge, num_channel_change=num_channel_change,
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False,lr=lr, decay=decay)
            out_layer = name_prefix + '_relu'
    return net
# SSDHeaders
def DenseSsdDetectorHeaders(net, \
              boxsizes=[], \
              net_width=300, net_height=300, \
              data_layer="data", num_classes=2, \
              from_layers=[], \
              use_batchnorm=True, \
              prior_variance = [0.1,0.1,0.2,0.2], \
              normalizations=[], \
              aspect_ratios=[], \
              flip=True, clip=True, \
              inter_layer_channels=[], \
              kernel_size=3,pad=1,\
              use_focus_loss=False):
    assert from_layers, "Feature layers must be provided."
    pro_widths=[]
    pro_heights=[]
    for i in range(len(boxsizes)):
      boxsizes_per_layer = boxsizes[i]
      pro_widths_per_layer = []
      pro_heights_per_layer = []
      for j in range(len(boxsizes_per_layer)):
        boxsize = boxsizes_per_layer[j]
        aspect_ratio = aspect_ratios[0]
        if not len(aspect_ratios) == 1:
          aspect_ratio = aspect_ratios[i][j]
        for each_aspect_ratio in aspect_ratio:
            w = boxsize * math.sqrt(each_aspect_ratio)
            h = boxsize / math.sqrt(each_aspect_ratio)
            w = min(w,1.0)
            h = min(h,1.0)
            pro_widths_per_layer.append(w)
            pro_heights_per_layer.append(h)
      pro_widths.append(pro_widths_per_layer)
      pro_heights.append(pro_heights_per_layer)

    mbox_layers = MultiLayersDenseDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                                            from_layers=from_layers, \
                                            normalizations=normalizations, \
                                            use_batchnorm=use_batchnorm, \
                                            prior_variance = prior_variance, \
                                            pro_widths=pro_widths, pro_heights=pro_heights, \
                                            flip=flip, clip=clip, \
                                            inter_layer_channels=inter_layer_channels, \
                                            kernel_size=kernel_size, pad=pad, \
                                            use_focus_loss=use_focus_loss)
    return mbox_layers

# Conv6
def addconv6(net, from_layer="conv5", use_bn=True, conv6_output=[],conv6_kernal_size=[],start_pool=True,leaky=False,lr_mult=1, decay_mult=1,pre_name="conv6",post_name="",n_group=1):
	if start_pool:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
		from_layer = out_layer
	assert len(conv6_output) == len(conv6_kernal_size)
	for i in range(len(conv6_output)):
		out_layer = '{}_{}{}'.format(pre_name,i+1,post_name)
		num_output = conv6_output[i]
		kernel_size = conv6_kernal_size[i]
		if kernel_size == 3:
			pad = 1
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=leaky,lr_mult=lr_mult, decay_mult=decay_mult,n_group=n_group)
		else:
			pad = 0
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=leaky,lr_mult=lr_mult, decay_mult=decay_mult,n_group=1)
		from_layer = out_layer
	return net

# Final Network
def SsdDetector(net, train=True, data_layer="data", gt_label="label", \
                net_width=512, net_height=288, **ssdparam):
    # Conv6
    conv6_output = ssdparam.get("multilayers_conv6_output",[])
    conv6_kernal_size = ssdparam.get("multilayers_conv6_kernal_size",[])
    use_sub_layers = (6, 7)
    num_channels = (144, 288)
    output_channels = (128, 0)
    channel_scale = 4
    add_strs = "_recon"
    net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                          output_channels=output_channels,channel_scale=channel_scale,lr=1, decay=1, add_strs=add_strs,)
    # Conv6
    out_layer = "conv3_7_recon_relu"
    net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=conv6_output, \
        conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1,n_group=1)
    # Concat FM1
    feature_layers = []
    featuremap1 = ["pool1_recon","conv2_6_recon_relu"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap1"
    UnifiedMultiScaleLayers(net,layers=featuremap1, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    feature_layers.append(out_layer)
    # Concat FM2
    featuremap2 = ["conv2_6_recon_relu","conv3_7_recon_relu"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap2"
    UnifiedMultiScaleLayers(net,layers=featuremap2, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    feature_layers.append(out_layer)
    # Concat FM3
    featuremap3 = ["conv3_7_recon_relu","conv6_5"]
    tags = ["Down","Ref"]
    down_methods = [["MaxPool"]]
    out_layer = "featuremap3"
    UnifiedMultiScaleLayers(net,layers=featuremap3, tags=tags, unifiedlayer=out_layer, dnsampleMethod=down_methods)
    feature_layers.append(out_layer)
    # Create SSD Header
    mbox_layers = DenseSsdDetectorHeaders(net, \
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
         clip=ssdparam.get("multilayers_clip",True), \
         inter_layer_channels=ssdparam.get("multilayers_inter_layer_channels",[]), \
         kernel_size=ssdparam.get("multilayers_kernel_size",3), \
         pad=ssdparam.get("multilayers_pad",1), \
         use_focus_loss=ssdparam.get("multiloss_using_focus_loss",False))
    # Loss & Det-eval
    if train:
        loss_param = get_loss_param(normalization=ssdparam.get("multiloss_normalization",P.Loss.VALID))
        mbox_layers.append(net[gt_label])
        densebboxloss_param = {
            'loc_loss_type':ssdparam.get("multiloss_loc_loss_type",P.MultiBoxLoss.SMOOTH_L1),
            'conf_loss_type':ssdparam.get("multiloss_conf_loss_type",P.MultiBoxLoss.LOGISTIC),
            'loc_weight':ssdparam.get("multiloss_loc_weight",1),
            'conf_weight':ssdparam.get("multiloss_conf_weight",1),
            'num_classes':ssdparam.get("num_classes",2),
            'overlap_threshold':ssdparam.get("multiloss_overlap_threshold",0.5),
            'use_prior_for_matching':True,
            'use_difficult_gt':ssdparam.get("multiloss_use_difficult_gt",False),
            'do_neg_mining':ssdparam.get("multiloss_do_neg_mining",True),
            'neg_pos_ratio':ssdparam.get("multiloss_neg_pos_ratio",3),
            'neg_overlap':ssdparam.get("multiloss_neg_overlap",0.5),
            'code_type':ssdparam.get("multiloss_code_type",P.PriorBox.CENTER_SIZE),
            'encode_variance_in_target': False,
            'size_threshold':ssdparam.get("multiloss_size_threshold",0.0001),
            'alias_id':ssdparam.get("multiloss_alias_id",0),
            'using_focus_loss':ssdparam.get("multiloss_using_focus_loss",False),
            'gama':ssdparam.get("multiloss_focus_gama",2),
        }
        net["mbox_loss"] = L.DenseBBoxLoss(*mbox_layers, \
                                dense_bbox_loss_param=densebboxloss_param, \
                                loss_param=loss_param, \
                                include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), \
                                propagate_down=[True, True, False, False])
    else:
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
        # Det-out param
        densedet_out_param = {
            'num_classes':ssdparam.get("num_classes",2),
            'share_location':ssdparam.get("multiloss_share_location",True),
            'background_label_id':0,
            'code_type':ssdparam.get("multiloss_code_type",P.PriorBox.CENTER_SIZE),
            'variance_encoded_in_target':False,
            'conf_threshold':ssdparam.get("detectionout_conf_threshold",0.01),
            'nms_threshold':ssdparam.get("detectionout_nms_threshold",0.45),
            'size_threshold':ssdparam.get("detectionout_size_threshold",0.0001),
            'top_k':ssdparam.get("detectionout_top_k",30),
            'alias_id':ssdparam.get("multiloss_alias_id",0),
        }
        net.detection_out = L.DenseDetOut(*mbox_layers, \
    	  		detection_output_param=densedet_out_param, \
    	  		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        # Det-eval
        det_eval_param = {
            'num_classes':ssdparam.get("num_classes",2),
            'background_label_id':0,
            'evaluate_difficult_gt':ssdparam.get("detectioneval_evaluate_difficult_gt",False),
            'boxsize_threshold':ssdparam.get("detectioneval_boxsize_threshold",[0,0.01,0.05,0.1,0.15,0.2,0.25]),
            'iou_threshold':ssdparam.get("detectioneval_iou_threshold",[0.9,0.75,0.5]),
        }
        net.det_accu = L.DetEval(net.detection_out, net[gt_label], \
            	  detection_evaluate_param=det_eval_param, \
            	  include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net
