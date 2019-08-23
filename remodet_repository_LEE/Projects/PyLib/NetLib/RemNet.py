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
conv_stage_name = ['conv1','conv2','conv3','conv4','conv5','rconv6']
use_stride_conv = [True,True,False,False,False]
base_use_bn = True
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
# Stages
# VecMap
vec_channels = [64,64,64,64,34]
vec_kernels  = [3,3,3,3,3]
vec_pads     = [1,1,1,1,1]
# HeatMap
heat_channels = [64,64,64,64,18]
heat_kernels  = [3,3,3,3,3]
heat_pads     = [1,1,1,1,1]
# layers
use_layers = len(vec_channels)
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
			stride=1, use_scale=True)
		start_layer = out_layer
	# last layer
	if num_layer == 1:
		out_layer = "{}".format(conv_stage_name[stage-1])
	else:
		out_layer = "{}_{}".format(conv_stage_name[stage-1],num_layer)
	if not use_stride_conv[stage-1]:
		ConvBNUnitLayer(net, start_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=layers[num_layer-1], kernel_size=kernels[num_layer-1], pad=pads[num_layer-1], \
			stride=1, use_scale=True)
		start_layer = out_layer
		if use_pool:
			out_layer = "pool{}".format(stage)
			net[out_layer] = L.Pooling(net[start_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
	else:
		ConvBNUnitLayer(net, start_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=layers[num_layer-1], kernel_size=kernels[num_layer-1], pad=pads[num_layer-1], \
			stride=2, use_scale=True)

def RemBaseNet(net, from_layer="data", use_bn=True, use_conv6=False, lr=1, decay=1) :
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

def RemPoseStage_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask", \
                       label_vec="vec_label", label_heat="heat_label", \
                       short_cut=True, base_layer="convf", lr=1, decay=1):
        kwargs = {
           'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
        assert from_layer in net.keys()
	assert len(vec_channels) == len(heat_channels)
	assert len(vec_channels) == len(vec_kernels)
	assert len(vec_channels) == len(vec_pads)
	assert len(heat_channels) == len(heat_pads)
	assert len(heat_channels) == len(heat_kernels)
        from1_layer = from_layer
        from2_layer = from_layer
        for layer in range(1, use_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=vec_channels[layer-1], \
				pad=vec_pads[layer-1], kernel_size=vec_kernels[layer-1], **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=heat_channels[layer-1], \
			 	pad=heat_pads[layer-1], kernel_size=heat_kernels[layer-1], **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=vec_channels[use_layers-1], \
			pad=vec_pads[use_layers-1], kernel_size=vec_kernels[use_layers-1], **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=heat_channels[use_layers-1], \
		 	pad=heat_pads[use_layers-1], kernel_size=heat_kernels[use_layers-1], **kwargs)
        weight_vec = "weight_stage{}_vec".format(stage)
        weight_heat = "weight_stage{}_heat".format(stage)
        loss_vec = "loss_stage{}_vec".format(stage)
        loss_heat = "loss_stage{}_heat".format(stage)
        net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
        net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
        return net

def RemPoseStage_Test(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                      short_cut=True, base_layer="convf", lr=1, decay=1):
        kwargs = {
           'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
        assert from_layer in net.keys()
	assert len(vec_channels) == len(heat_channels)
	assert len(vec_channels) == len(vec_kernels)
	assert len(vec_channels) == len(vec_pads)
	assert len(heat_channels) == len(heat_pads)
	assert len(heat_channels) == len(heat_kernels)
        from1_layer = from_layer
        from2_layer = from_layer
        for layer in range(1, use_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,layer)
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=vec_channels[layer-1], \
				pad=vec_pads[layer-1], kernel_size=vec_kernels[layer-1], **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,layer)
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,layer)
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=heat_channels[layer-1], \
			 	pad=heat_pads[layer-1], kernel_size=heat_kernels[layer-1], **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,layer)
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_layers)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=vec_channels[use_layers-1], \
			pad=vec_pads[use_layers-1], kernel_size=vec_kernels[use_layers-1], **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_layers)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=heat_channels[use_layers-1], \
		 	pad=heat_pads[use_layers-1], kernel_size=heat_kernels[use_layers-1], **kwargs)
        # 特征拼接
        if short_cut:
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
        return net

def RemPoseNet_Train(net, data_layer="data", label_layer="label"):
        # input
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
		# label
        net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
		# BaseNet
        net = RemBaseNet(net, from_layer=data_layer, use_bn=base_use_bn, use_conv6=False, lr=1, decay=1)
		# Stage-5
        stage_5 = "{}_{}".format(conv_stage_name[4],len(stage5_layers))
	if use_stride_conv[4]:
		stage_5 = "{}_{}".format(conv_stage_name[4],len(stage5_layers)-1)
	# Stage-4
	stage_4 = "{}_{}".format(conv_stage_name[3],len(stage4_layers))
	if use_stride_conv[3]:
		stage_4 = "{}_{}".format(conv_stage_name[3],len(stage4_layers)-1)
        net = UnifiedMultiScaleLayers(net, layers=[stage_4,stage_5], tags=["Ref","Up"], unifiedlayer="convf", upsampleMethod="Reorg")
        # Stages
        baselayer = "convf"
	stage_lr = 1
        # STG#1
        net = RemPoseStage_Train(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
	                       mask_vec="vec_mask", mask_heat="heat_mask", \
	                       label_vec="vec_label", label_heat="heat_label", \
	                       short_cut=True, base_layer=baselayer, lr=stage_lr, decay=1)
        # STG#2
        net = RemPoseStage_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
	                       mask_vec="vec_mask", mask_heat="heat_mask", \
	                       label_vec="vec_label", label_heat="heat_label", \
	                       short_cut=True, base_layer=baselayer, lr=stage_lr, decay=1)
        # STG#3
        net = RemPoseStage_Train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
	                       mask_vec="vec_mask", mask_heat="heat_mask", \
	                       label_vec="vec_label", label_heat="heat_label", \
	                       short_cut=False, base_layer=baselayer, lr=stage_lr, decay=1)
        return net

def RemPoseNet_Test(net, from_layer="data", frame_layer="orig_data", **pose_kwargs):
        # BaseNet
	net = RemBaseNet(net, from_layer=from_layer, use_bn=base_use_bn, use_conv6=False, lr=1, decay=1)
        # Stage-5
        stage_5 = "{}_{}".format(conv_stage_name[4],len(stage5_layers))
	if use_stride_conv[4]:
		stage_5 = "{}_{}".format(conv_stage_name[4],len(stage5_layers)-1)
	# Stage-4
	stage_4 = "{}_{}".format(conv_stage_name[3],len(stage4_layers))
	if use_stride_conv[3]:
		stage_4 = "{}_{}".format(conv_stage_name[3],len(stage4_layers)-1)
        net = UnifiedMultiScaleLayers(net, layers=[stage_4,stage_5], tags=["Ref","Up"], unifiedlayer="convf", upsampleMethod="Reorg")

        # STG#1
        net = RemPoseStage_Test(net, from_layer=baselayer, out_layer="concat_stage1", stage=1, \
	                       short_cut=True, base_layer=baselayer, lr=1, decay=1)
        # STG#2
        net = RemPoseStage_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=2, \
	                       short_cut=True, base_layer=baselayer, lr=1, decay=1)
        # STG#3
        net = RemPoseStage_Train(net, from_layer="concat_stage2", out_layer="concat_stage3", stage=3, \
	                       short_cut=False, base_layer=baselayer, lr=1, decay=1)

        conv_vec = "stage{}_conv{}_vec".format(3,use_layers)
        conv_heat = "stage{}_conv{}_heat".format(3,use_layers)
        feaLayers = []
        feaLayers.append(net[conv_heat])
        feaLayers.append(net[conv_vec])
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_kwargs.get("resize_factor", 2),
            'scale_gap': pose_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # VisualizePose
        visual_kwargs = {
            'is_type_coco': pose_kwargs.get("conn_is_type_coco", True),
            'type': pose_kwargs.get("visual_type", P.Visualizepose.POSE),
            'visualize': pose_kwargs.get("visual_visualize", True),
            'draw_skeleton': pose_kwargs.get("visual_draw_skeleton", True),
            'print_score': pose_kwargs.get("visual_print_score", False),
            'part_id': pose_kwargs.get("visual_part_id", 0),
            'from_part': pose_kwargs.get("visual_from_part", 0),
            'vec_id': pose_kwargs.get("visual_vec_id", 0),
            'from_vec': pose_kwargs.get("visual_from_vec", 0),
            'pose_threshold': pose_kwargs.get("visual_pose_threshold", 0.05),
            'write_frames': pose_kwargs.get("visual_write_frames", False),
            'output_directory': pose_kwargs.get("visual_output_directory", ""),
        }
        net.visual = L.Visualizepose(net[frame_layer], net.resized_map, net.limbs, visualize_pose_param=visual_kwargs)
        return net
