# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from ConvBNLayer import *
from MultiScaleLayer import *
# ============================Layers Configuration==============================
# ==============================================================================
# BaseNet
conv_stage_name = ['conv1','conv2','conv3','conv4','conv5']
use_stride_conv = [True,True,True,False,False]
base_use_bn = False
# Stage-1
stage1_layers  = [32]
stage1_kernels = [3]
stage1_pads    = [1]
# Stage-2
stage2_layers  = [64]
stage2_kernels = [3]
stage2_pads    = [1]
# Stage-3
stage3_layers  = [128,64,128]
stage3_kernels = [3,1,3]
stage3_pads    = [1,0,1]
# Stage-4
stage4_layers  = [256,128,256]
stage4_kernels = [3,1,3]
stage4_pads    = [1,0,1]
# Stage-5
stage5_layers  = [512,256,512,256,512]
stage5_kernels = [3,1,3,1,3]
stage5_pads    = [1,0,1,0,1]
# ==============================================================================
# ==============================================================================
# @def
def ConvStage(net, from_layer="data", stage=1, layers=[], kernels=[], pads=[], \
			  use_bn=True, use_pool=True, lr=1, decay=1):
	num_layer = len(layers)
	assert len(layers) == len(kernels)
	assert len(layers) == len(pads)
	assert num_layer > 0, "Stage {} has not been defined." % stage
	start_layer = from_layer
	for i in xrange(1,num_layer):
		out_layer = "{}_{}".format(conv_stage_name[stage-1],i)
		ConvBNUnitLayer(net, start_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=layers[i-1], kernel_size=kernels[i-1], pad=pads[i-1], \
			stride=1, use_scale=True,lr_mult=lr, decay_mult=decay)
		start_layer = out_layer
	# last layer
	if num_layer == 1:
		out_layer = "{}".format(conv_stage_name[stage-1])
	else:
		out_layer = "{}_{}".format(conv_stage_name[stage-1],num_layer)
	if not use_stride_conv[stage-1]:
		ConvBNUnitLayer(net, start_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=layers[num_layer-1], kernel_size=kernels[num_layer-1], pad=pads[num_layer-1], \
			stride=1, use_scale=True,lr_mult=lr, decay_mult=decay)
		start_layer = out_layer
		if use_pool:
			out_layer = "pool{}".format(stage)
			net[out_layer] = L.Pooling(net[start_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
	else:
		ConvBNUnitLayer(net, start_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=layers[num_layer-1], kernel_size=kernels[num_layer-1], pad=pads[num_layer-1], \
			stride=2, use_scale=True,lr_mult=lr, decay_mult=decay)

# ReID BaseNet
def ReIDBaseNet(net, from_layer="data", use_bn=True, use_conv6=False, lr=1, decay=1):
	# Stage-1
	ConvStage(net, from_layer=from_layer, stage=1, layers=stage1_layers, kernels=stage1_kernels, \
	 		  pads=stage1_pads, use_bn=use_bn, lr=lr, decay=decay)
	# Stage-2
	start_layer = "pool1"
	if use_stride_conv[0]:
		if len(stage1_layers) == 1:
			start_layer = "{}".format(conv_stage_name[0])
		else:
			start_layer = "{}_{}".format(conv_stage_name[0],len(stage1_layers))
	ConvStage(net, from_layer=start_layer, stage=2, layers=stage2_layers, kernels=stage2_kernels, \
	 		  pads=stage2_pads, use_bn=use_bn, lr=lr, decay=decay)
	# Stage-3
	start_layer = "pool2"
	if use_stride_conv[1]:
		if len(stage2_layers) == 1:
			start_layer = "{}".format(conv_stage_name[1])
		else:
			start_layer = "{}_{}".format(conv_stage_name[1],len(stage2_layers))
	ConvStage(net, from_layer=start_layer, stage=3, layers=stage3_layers, kernels=stage3_kernels, \
	 		  pads=stage3_pads, use_bn=use_bn, lr=lr, decay=decay)
	# Stage-4
	start_layer = "pool3"
	if use_stride_conv[2]:
		if len(stage3_layers) == 1:
			start_layer = "{}".format(conv_stage_name[2])
		else:
			start_layer = "{}_{}".format(conv_stage_name[2],len(stage3_layers))
	ConvStage(net, from_layer=start_layer, stage=4, layers=stage4_layers, kernels=stage4_kernels, \
	 		  pads=stage4_pads, use_bn=use_bn, lr=lr, decay=decay)
	# Stage-5
	start_layer = "pool4"
	if use_stride_conv[3]:
		if len(stage4_layers) == 1:
			start_layer = "{}".format(conv_stage_name[3])
		else:
			start_layer = "{}_{}".format(conv_stage_name[3],len(stage4_layers))
	if use_conv6:
		ConvStage(net, from_layer=start_layer, stage=5, layers=stage5_layers, kernels=stage5_kernels, \
		 		  pads=stage5_pads, use_bn=use_bn, lr=lr, decay=decay)
	else:
		ConvStage(net, from_layer=start_layer, stage=5, layers=stage5_layers, kernels=stage5_kernels, \
		 		  pads=stage5_pads, use_bn=use_bn, use_pool=False, lr=lr, decay=decay)
	return net

# ReID Extern Layers (Additional)
def ReIDExtLayers(net, from_layer="convf", label_layer="label", net_input_width=432, net_input_height=324, train=True, lr=1, decay=1):
	# roi_data_layer -> [ROI_POOLING + LABEL]
	# roi_pooling_layer -> (10,10) (0.0625(1/16))
	# we use [conv4_3(reorg) + conv5_5] as convf
	# use stride_conv to get conv6_1
	# -> conv6_2 -> conv6_3 -> (stride_conv) conv7_1 -> conv7_2 -> avg_pool
	# -> FC (256) -> Normalize
	# -> LabeledMatch / UnlabeledMatch (use label)
	# use scale / concat to get {L+Q} array
	# -> softmaxWithLoss & accuracy (train)
   	assert from_layer in net.keys()
	#  Roi_Data_Layer
	roi_data_kwargs = {
		'net_input_width':net_input_width,
		'net_input_height':net_input_height
	}
	net.roi_pool, net.roi_label = L.RoiData(net[label_layer],ntop=2,roi_data_param=roi_data_kwargs)
	# Roi_Pooling_Layer
	roi_pool_kwargs = {
		'pooled_h': 10,
		'pooled_w': 10,
		'spatial_scale': 0.0625,
	}
	net.rpf = L.ROIPooling(net[from_layer],net.roi_pool,roi_pooling_param=roi_pool_kwargs)
	# ConvLayers
	# conv6
	ConvBNUnitLayer(net, "rpf", "reid_c61", use_bn=False, use_relu=True, \
					num_output=256, kernel_size=3, pad=1,stride=2)
	ConvBNUnitLayer(net, "reid_c61", "reid_c62", use_bn=False, use_relu=True, \
					num_output=256, kernel_size=3, pad=1,stride=1)
	# ConvBNUnitLayer(net, "reid_c62", "reid_c63", use_bn=False, use_relu=True, \
	# 				num_output=256, kernel_size=3, pad=1,stride=1)
	# conv7
	ConvBNUnitLayer(net, "reid_c62", "reid_c71", use_bn=False, use_relu=True, \
					num_output=256, kernel_size=3, pad=1,stride=2)
	ConvBNUnitLayer(net, "reid_c71", "reid_c72", use_bn=False, use_relu=True, \
					num_output=256, kernel_size=3, pad=1,stride=1)
	# avg_pool
    	net.avgpool = L.Pooling(net["reid_c72"], pool=P.Pooling.AVE, global_pooling=True)
	# FC & Norm
    	fc_kwargs = {
        	'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
        	'weight_filler': dict(type='gaussian', std=0.005),
        	'bias_filler': dict(type='constant', value=0)}
	net.fp = L.InnerProduct(net.avgpool, num_output=256, **fc_kwargs)
    	net.fpn = L.Normalize(net.fp)
	# Match
	labelMatch_kwargs = {
		'num_classes': 5532,
		'momentum': 0.5,
	}
	net.labeled_match, net.gt = L.LabeledMatch(net.fpn, net.roi_label, ntop=2, labeled_match_param=labelMatch_kwargs)
	unlabelMatch_kwargs = {
		'queue_size': 5000,
	}
	net.unlabeled_match = L.UnlabeledMatch(net.fpn, net.roi_label, unlabeled_match_param=unlabelMatch_kwargs)
	# scale
	power_kwargs = {'scale': 10}
	net.labeled_match_scale = L.Power(net.labeled_match, **power_kwargs)
	net.unlabeled_match_scale = L.Power(net.unlabeled_match, **power_kwargs)
	# concat: cosine similarity
	net.cosine = L.Concat(net.labeled_match_scale,net.unlabeled_match_scale, axis=1)
	if train:
		# softmaxWithLoss
		loss_kwargs = {
			'ignore_label': -1,
			'normalize': True,
		}
		net.loss = L.SoftmaxWithLoss(net.cosine,net.gt,propagate_down=[True,False],loss_weight=[1],loss_param=loss_kwargs)
	else:
		# accuracy
		accu_kwargs = {
			'ignore_label': -1,
			'top_k': 1,
		}
		net.accuracy = L.AccuracyReid(net.cosine,net.gt,accuracy_param=accu_kwargs)
    	return net

def ReIDNet(net, data_layer="data", label_layer="label", net_input_width=432, net_input_height=324):
	# basenet
	net = ReIDBaseNet(net, from_layer=data_layer, use_bn=base_use_bn, lr=0, decay=0)
	# concat features
	net = UnifiedMultiScaleLayers(net, layers=['conv4_3','conv5_5'], tags=["Down","Ref"], unifiedlayer="convf", dnsampleMethod=[["Reorg"]])
	# ExtraLayers
	net = ReIDExtLayers(net, from_layer="convf", label_layer=label_layer, net_input_width=net_input_width, net_input_height=net_input_height, lr=1, decay=1)

    	return net

def ReIDNet_Test(net, data_layer="data", label_layer="label", net_input_width=432, net_input_height=324):
	# basenet
	net = ReIDBaseNet(net, from_layer=data_layer, use_bn=base_use_bn, lr=0, decay=0)
	# concat features
	net = UnifiedMultiScaleLayers(net, layers=['conv4_3','conv5_5'], tags=["Down","Ref"], unifiedlayer="convf", dnsampleMethod=[["Reorg"]])
	# ExtraLayers
	net = ReIDExtLayers(net, from_layer="convf", label_layer=label_layer, net_input_width=net_input_width, net_input_height=net_input_height, train=False, lr=1, decay=1)

    	return net
