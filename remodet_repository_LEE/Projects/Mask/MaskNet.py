# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
import caffe

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.path.append('../')
from PyLib.NetLib.YoloNet import *
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.SsdDetector_r2 import *
from mask_param import *

def SSDHeader(net,data_layer="data",from_layers=[],input_height=300,input_width=300,loc_postfix='',**ssdparam):
	# Create SSD Header
	mbox_layers = SsdDetectorHeaders(
				  net, \
                  boxsizes=ssdparam.get("boxsizes",[]), \
                  net_width=input_width, net_height=input_height, \
                  data_layer="data", \
				  num_classes=ssdparam.get("num_classes",2), \
                  from_layers=from_layers, \
                  use_batchnorm=ssdparam.get("use_bn",True), \
                  prior_variance=ssdparam.get("prior_variance",[0.1,0.1,0.2,0.2]), \
                  normalizations=ssdparam.get("normalizations",[]), \
                  aspect_ratios=ssdparam.get("aspect_ratios",[]), \
                  flip=ssdparam.get("flip",True), \
				  clip=ssdparam.get("clip",True), \
                  inter_layer_channels=ssdparam.get("inter_layer_channels",[]), \
                  kernel_size=ssdparam.get("kernel_size",3), \
				  pad=ssdparam.get("pad",1),loc_postfix=loc_postfix)
	return mbox_layers

# def KpsHeader1(net,from_layer="roi_maps",out_layer="kps_maps",use_layers=5,num_channels=128, \
# 			  kernel_size=3,pad=1,use_deconv_layers=1,lr=1,decay=1):
# 	# use conv & deconv operations
# 	kwargs = {
#        'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
#        'weight_filler': dict(type='xavier'),
#        'bias_filler': dict(type='constant', value=0)
# 	}
# 	# conv
# 	start_layer = from_layer
# 	for layer in range(1, use_layers):
# 		conv_kps = "kps_conv{}".format(layer)
# 		net[conv_kps] = L.Convolution(net[start_layer], num_output=num_channels, \
# 			pad=pad, kernel_size=kernel_size, **kwargs)
# 		relu_kps = "kps_relu{}".format(layer)
# 		net[relu_kps] = L.ReLU(net[conv_kps], in_place=True)
# 		start_layer = relu_kps

# 	if use_deconv_layers > 0:
# 		start_layer = relu_kps
# 		# deconv
# 		deconv_kwargs = {
# 		    'param': [dict(lr_mult=lr, decay_mult=decay)],
# 		    'convolution_param': {
# 		        'num_output': num_channels,
# 		        'kernel_size': 2,
# 		        'pad': 0,
# 		        'stride': 2,
# 		        'weight_filler': dict(type='xavier'),
# 		        'bias_term': True,
# 		        'group': 1,
# 		        'bias_filler': dict(type='constant', value=0.0),
# 		        }
# 	    }
# 		for layer in range(1, use_deconv_layers+1):
# 			deconv_kps = "kps_deconv{}".format(layer)
# 			net[deconv_kps] = L.Deconvolution(net[start_layer], **deconv_kwargs)
# 			derelu_kps = "kps_derelu{}".format(layer)
# 			net[derelu_kps] = L.ReLU(net[deconv_kps], in_place=True)
# 			start_layer = derelu_kps
# 	# Last Layer
# 	conv_kps = "kps_conv{}".format(use_layers+use_deconv_layers)
# 	net[conv_kps] = L.Convolution(net[start_layer], num_output=18, \
# 		pad=pad, kernel_size=kernel_size, **kwargs)	
# 	net[out_layer] = net[conv_kps]
# 	return net
def KpsHeader(net,from_layer="roi_maps",out_layer="kps_maps",use_layers=[],num_channels=[], \
			  all_kernel_size=[],use_bn=True,pad=3,lr=1,decay=1,use_deconv_layers=0,pre_name="kps_conv",post_name=""):
	kwargs = {
       'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
       'weight_filler': dict(type='xavier'),
       'bias_filler': dict(type='constant', value=0)
	}
	# conv
	assert len(use_layers)==len(num_channels)
	assert len(num_channels)==len(all_kernel_size)
	for i in range(len(use_layers)):
		next_layer = '{}_{}{}'.format(pre_name,i+1,post_name)
		use_conv=use_layers[i]
		kernel_size = all_kernel_size[i]
		if use_conv==1:
			if kernel_size==3:
				pad = 1
				ConvBNUnitLayer(net,from_layer,next_layer,use_bn=use_bn,use_relu=True,num_output=num_channels[i],
					kernel_size=kernel_size,pad=pad,stride=1,use_scale=True,leaky=True,lr_mult=lr, decay_mult=decay,n_group=1)
			else:
				pad = 0
				ConvBNUnitLayer(net,from_layer,next_layer,use_bn=use_bn,use_relu=True,num_output=num_channels[i],
					kernel_size=kernel_size,pad=pad,stride=1,use_scale=True,leaky=True,lr_mult=lr, decay_mult=decay,n_group=1)
			from_layer = next_layer
		else:			
			if use_conv==0:
				deconv_kwargs = {
				'param': [dict(lr_mult=lr, decay_mult=decay)],
				'convolution_param': {
					'num_output': num_channels[i],
					'kernel_size': 2,
					'pad': 0,
					'stride': 2,
					'weight_filler': dict(type='xavier'),
					'bias_term': True,
					'group': 1,
					'bias_filler': dict(type='constant', value=0.0),
				}
			}
				
 			else:
 				deconv_kwargs = {
					'param': [dict(lr_mult=0, decay_mult=0)],
					'convolution_param': {
						'num_output': num_channels[i],
						'kernel_size': 2,
						'pad': 0,
						'stride': 2,
						'weight_filler': dict(type="bilinear"),
						'bias_term': False,
 						'group': 1,
 						 }
 					 }
			deconv_kps = "kps_deconv{}".format(i+1)
			net[deconv_kps]=L.Deconvolution(net[from_layer],**deconv_kwargs)
			derelu_kps = "kps_derelu{}".format(i+1)
			net[derelu_kps] = L.ReLU(net[deconv_kps], in_place=True)
			from_layer = deconv_kps	
	# Last Layer
	conv_kps = "kps_conv{}".format(len(use_layers)+2)
	net[conv_kps] = L.Convolution(net[from_layer], num_output=18, \
		pad=1, kernel_size=kernel_size, **kwargs)	
	net[out_layer] = net[conv_kps]
	return net
def MaskHeader(net,from_layer="roi_maps",out_layer="mask_maps",use_layers=5,num_channels=128, \
			  kernel_size=3,pad=1,use_deconv_layers=0,lr=1,decay=1):
	# use conv & deconv operations
	kwargs = {
       'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
       'weight_filler': dict(type='xavier'),
       'bias_filler': dict(type='constant', value=0)
	}
	# conv
	start_layer = from_layer
	for layer in range(1, use_layers):
		conv_mask = "mask_conv{}".format(layer)
		net[conv_mask] = L.Convolution(net[start_layer], num_output=num_channels, \
			pad=pad, kernel_size=kernel_size, **kwargs)
		relu_mask = "mask_relu{}".format(layer)
		net[relu_mask] = L.ReLU(net[conv_mask], in_place=True)
		start_layer = relu_mask
	
	if use_deconv_layers > 0:
		start_layer = relu_mask
		# deconv
		deconv_kwargs = {
		    'param': [dict(lr_mult=lr, decay_mult=decay)],
		    'convolution_param': {
		        'num_output': num_channels,
		        'kernel_size': 2,
		        'pad': 0,
		        'stride': 2,
		        'weight_filler': dict(type='xavier'),
		        'bias_term': True,
		        'group': 1,
		        'bias_filler': dict(type='constant', value=0.0),
		        }
	    }
		for layer in range(1, use_deconv_layers+1):
			deconv_mask = "mask_deconv{}".format(layer)
			net[deconv_mask] = L.Deconvolution(net[start_layer], **deconv_kwargs)
			derelu_mask = "mask_derelu{}".format(layer)
			net[derelu_mask] = L.ReLU(net[deconv_mask], in_place=True)
			start_layer = derelu_mask
	# Last Layer
	conv_mask = "mask_conv{}".format(use_layers+use_deconv_layers)
	net[conv_mask] = L.Convolution(net[start_layer], num_output=1, \
		pad=pad, kernel_size=kernel_size, **kwargs)
	net[out_layer] = net[conv_mask]
	return net
def Face_train(net,from_layer="data",label="label",lr=1,decay=1):
	# net = FaceNet(net, from_layer="data", use_bn=True)
    # net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=0.1)
	net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=0, decay=0)

	ConvBNUnitLayer(net,"conv2", "conv2_pool", use_bn=True, use_relu=True, num_output=64,
            kernel_size=1, pad=0, stride=2) 
	net = UnifiedMultiScaleLayers(net,layers=["conv2_pool","conv3_3","conv4_3"],tags=["Down","Down","Ref"],unifiedlayer="featuremap11",dnsampleMethod=[["Conv"],["MaxPool"]],dnsampleChannels=64)
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap22",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap33",dnsampleMethod=[["MaxPool"]],pad=True)
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap11","featuremap22","featuremap33"],input_height=Input_Height,input_width=Input_Width,**ssdparam)

	# mbox_layers = SSDHeader(net,data_layer="data",from_layers=["inception3","conv3_2","conv4_2"],input_height=Input_Height,input_width=Input_Width,**ssdparam)
	mbox_layers.append(net.label)
	net.bbox_loss = L.BBoxLoss(*mbox_layers,name="BBoxLoss",bbox_loss_param=bbox_loss_param)
	# net.bbox_loss = L.DenseBBoxLoss(*mbox_layers,name="DenseDetLoss",dense_bbox_loss_param=dense_bbox_loss_param)
	return net

def Face_eval(net, from_layer="data", label="label", lr=1, decay=1,visualize=False):

	# net =YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)	
	# net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=0.1)
	# net = FaceNet(net, from_layer="data", use_bn=True)
	net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=0, decay=0)

	
	ConvBNUnitLayer(net,"conv2", "conv2_pool", use_bn=True, use_relu=True, num_output=64,
            kernel_size=1, pad=0, stride=2) 
	net = UnifiedMultiScaleLayers(net,layers=["conv2_pool","conv3_3","conv4_3"],tags=["Down","Down","Ref"],unifiedlayer="featuremap11",dnsampleMethod=[["Conv"],["MaxPool"]],dnsampleChannels=64)
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap22",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap33",dnsampleMethod=[["MaxPool"]],pad=True)
	

	# mbox_layers = SSDHeader(net,data_layer="data",from_layers=["inception3","conv3_2","conv4_2"],input_height=Input_Height,input_width=Input_Width,**ssdparam)
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap11","featuremap22","featuremap33"],input_height=Input_Height,input_width=Input_Width,**ssdparam)

	
	reshape_name = "mbox_conf_reshape"
	net[reshape_name] = L.Reshape(mbox_layers[1], \
		shape=dict(dim=[0, -1, ssdparam.get("num_classes",2)]))
	softmax_name = "mbox_conf_softmax"
	net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
	flatten_name = "mbox_conf_flatten"
	net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
 	mbox_layers[1] = net[flatten_name]
	# elif bbox_loss_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
 #    	sigmoid_name = "mbox_conf_sigmoid"
 #    	net[sigmoid_name] = L.Sigmoid(mbox_layers[1])
 #    	mbox_layers[1] = net[sigmoid_name]
	# if visualize:
	# 	mbox_layers.append(net["data"])
	# net.detection_out=L.DenseDetOut(*mbox_layers,
	# 	detection_output_param=det_out_param,
	# 	include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	# net.detection_eval_accu = L.DetEval(net.detection_out,net.label,detection_evaluate_param=det_eval_param,
	# 	include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.detection_out=L.DetOut(*mbox_layers,
		detection_output_param=det_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.detection_eval_accu = L.DetEval(net.detection_out,net.label,detection_evaluate_param=det_eval_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))


	return net
def MTD_Train(net,from_layer="data",label="label",lr=1,decay=1):
	# net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=0, decay=0)
	net,mbox_layers,parts_layers=MTD_BODY(net)
	net.bbox,net.parts=L.SplitLabel(net[label],name="SplitLabel",ntop=2,split_label_param=dict(add_parts=True))
	
	# ConvBNUnitLayer(net,"conv2", "conv2_pool", use_bn=True, use_relu=True, num_output=64,
 #            kernel_size=1, pad=0, stride=2) 
	# net = UnifiedMultiScaleLayers(net,layers=["conv2_pool","conv3_3","conv4_3"],tags=["Down","Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Conv"],["MaxPool"]],dnsampleChannels=64)
	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap3",dnsampleMethod=[["MaxPool"]],pad=True)
	# mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2","featuremap3"],input_height=Input_Height,input_width=Input_Width,loc_postfix='det',**ssdparam)
	
	mbox_layers.append(net.bbox)
	net.bbox_loss = L.DenseBBoxLoss(*mbox_layers,name="DenseDetLoss",dense_bbox_loss_param=dense_bbox_loss_param)
	# net = UnifiedMultiScaleLayers(net,layers=["conv2","conv3_3"],tags=["Down","Ref"],unifiedlayer="conf23",dnsampleMethod=[["Conv"]],dnsampleChannels=64)	
	# net = UnifiedMultiScaleLayers(net,layers=["conf23","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv3_3","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conf34","conv5_5"],tags=["Down","Ref"],unifiedlayer="conf45",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conf45","conv6_7"],tags=["Down","Ref"],unifiedlayer="conf56",dnsampleMethod=[["MaxPool"]],pad=True)
	# parts_layers = SSDHeader(net,data_layer="data",from_layers=["conf34","conf45","conv6_7"],input_height=Input_Height,input_width=Input_Width,loc_postfix='parts',**partsparam)
	parts_layers.append(net.parts)
	net.parts_loss = L.DenseBBoxLoss(*parts_layers,name="DensePartsLoss",dense_bbox_loss_param=dense_parts_loss_param)	
	
	return net
def MaskNet_Train(net, from_layer="data", label="label", lr=1, decay=1):
	# ==========================================================================
	# DarkNet19
	# net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=lr, decay=decay)
	net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.01, decay=1)
	out_layer = "conv5_5"
	net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=[128,128,128,128,128,128], \
        conv6_kernal_size=[3,3,3,3,3,3], pre_name="conv6",start_pool=True,lr_mult=0.1, decay_mult=1,n_group=1)
	# ==========================================================================
	# Label Split
	net.bbox, net.kps, net.mask = L.SplitLabel(net[label],name="SplitLabel",ntop=3)
	# ==========================================================================
	# Concat [conv5_5, conv6_7]
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_6"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Reorg"]])
	# Concat [conv4_3, conv5_5, conv6_7]
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"], tags=["Down","Ref"], unifiedlayer="convf_mask", \
	                            dnsampleMethod=[["MaxPool"]])
	# ==========================================================================
	# Create SSD Header
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2"],input_height=Input_Height,input_width=Input_Width,**ssdparam)
	# BBox Loss
	mbox_layers.append(net.bbox)
	net.bbox_loss = L.BBoxLoss(*mbox_layers,name="BBoxLoss",bbox_loss_param=bbox_loss_param)
	# ==========================================================================
	net.roi = L.BoxMatching(net["mbox_priorbox"],net.bbox,box_matching_param=box_matching_param)
	# ROI Align
	net.roi_maps = L.RoiAlign(net.convf_mask,net.roi,roi_align_param=roi_align_param)
	# ==========================================================================
	# Kps Layers
	net = KpsHeader(net,from_layer="roi_maps",out_layer="kps_maps",use_layers=kps_use_conv_layers,num_channels=channels_of_kps, \
				    all_kernel_size=kernel_size_of_kps,pad=pad_of_kps,use_deconv_layers=kps_use_deconv_layers,lr=10,decay=decay)
	net.kps_flatten = L.Flatten(net.kps_maps,flatten_param=dict(axis=2,end_axis=-1))

	# ==========================================================================
	# Mask Layers
	net = MaskHeader(net,from_layer="roi_maps",out_layer="mask_maps",use_layers=mask_use_conv_layers,num_channels=channels_of_mask, \
				    kernel_size=kernel_size_of_mask,pad=pad_of_mask,use_deconv_layers=mask_use_deconv_layers,lr=lr,decay=decay)
    # ==========================================================================
    # Labels for ROIs of Kps and Mask
	# kps-label gen
	net.kps_label_map, net.kps_label_flags = L.KpsGen(net.roi,net.kps,name="KpsGen",ntop=2,kps_gen_param=dict(resized_height=Rh_Kps,resized_width=Rw_Kps,use_softmax=True))

	# mask-label gen
	net.mask_label_map, net.mask_label_flags = L.MaskGen(net.roi,net.mask,name="MaskGen",ntop=2, \
		mask_gen_param=dict(height=Input_Height,width=Input_Width,resized_height=Rh_Mask,resized_width=Rw_Mask))
	# ==========================================================================
	# Kps-Loss & Mask-Loss
	net.kps_loss = L.MaskSoftmaxWithLoss(net.kps_flatten,net.kps_label_map,net.kps_label_flags,mask_loss_param=dict(scale=loss_scale_kps),loss_param=dict(normalization=0),softmax_param=dict(axis=2))
	net.mask_loss = L.MaskSigmoidCrossEntropyLoss(net.mask_maps,net.mask_label_map,net.mask_label_flags,mask_loss_param=dict(scale=loss_scale_mask))
	
	return net
def MaskNet_Val_MTD(net, from_layer="data", label="label", lr=1, decay=1,visualize=False):
	# net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=0, decay=0)
	net,mbox_layers,parts_layers=MTD_BODY(net)
	net.bbox,net.parts=L.SplitLabel(net[label],name="SplitLabel",ntop=2,split_label_param=dict(add_parts=True))
	# ConvBNUnitLayer(net,"conv2", "conv2_pool", use_bn=True, use_relu=True, num_output=64,
 #            kernel_size=1, pad=0, stride=2) 

	# net = UnifiedMultiScaleLayers(net,layers=["conv2_pool","conv3_3","conv4_3"],tags=["Down","Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Conv"],["MaxPool"]],dnsampleChannels=64)
	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap3",dnsampleMethod=[["MaxPool"]],pad=True)
	# mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2","featuremap3"],input_height=Input_Height,input_width=Input_Width,loc_postfix='det',**ssdparam)
	reshape_name = "mbox_conf_reshape"
	net[reshape_name] = L.Reshape(mbox_layers[1], \
		shape=dict(dim=[0, -1, ssdparam.get("num_classes",2)]))
	softmax_name = "mbox_conf_softmax"
	net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
	flatten_name = "mbox_conf_flatten"
	net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
	mbox_layers[1] = net[flatten_name]
	net.detection_out=L.DenseDetOut(*mbox_layers,
		detection_output_param=det_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.detection_eval_accu = L.DetEval(net.detection_out,net.bbox,detection_evaluate_param=det_eval_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	# net = UnifiedMultiScaleLayers(net,layers=["conv2","conv3_3"],tags=["Down","Ref"],unifiedlayer="conf23",dnsampleMethod=[["Conv"]],dnsampleChannels=64)	
	# net = UnifiedMultiScaleLayers(net,layers=["conf23","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv3_3","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conf34","conv5_5"],tags=["Down","Ref"],unifiedlayer="conf45",dnsampleMethod=[["MaxPool"]])
	# # net = UnifiedMultiScaleLayers(net,layers=["conf45","conv6_7"],tags=["Down","Ref"],unifiedlayer="conf56",dnsampleMethod=[["MaxPool"]],pad=True)
	# parts_layers = SSDHeader(net,data_layer="data",from_layers=["conf34","conf45","conv6_7"],input_height=Input_Height,input_width=Input_Width,loc_postfix='parts',**partsparam)

	# parts_layers = SSDHeader(net,data_layer="data",from_layers=["conf23","conf34","conf45","conf56"],input_height=Input_Height,input_width=Input_Width,loc_postfix='parts',**partsparam)
	sigmoid_name = "parts_conf_sigmoid"
	net[sigmoid_name] = L.Sigmoid(parts_layers[1])
	parts_layers[1] = net[sigmoid_name]
	net.parts_out=L.DenseDetOut(*parts_layers,
		detection_output_param=parts_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.parts_eval_accu = L.DetEval(net.parts_out,net.parts,detection_evaluate_param=parts_eval_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	# net.out=L.Concat(net.detection_eval_accu,net.parts_eval_accu,axis=2)
	return net
def MaskNet_Val_Det(net, from_layer="data", label="label", lr=1, decay=1,visualize=False):

	# net =YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)	
	net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=0.1)
	out_layer = "conv5_5"
	net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=[128,128,128,128,128,128], \
        conv6_kernal_size=[3,3,3,3,3,3], pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1,n_group=1)
	net.bbox,net.kps,net.mask = L.SplitLabel(net[label],name='SplitLabel',ntop=3)
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_6"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Reorg"]])
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2"],input_height=Input_Height,input_width=Input_Width,**ssdparam)

	if bbox_loss_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.SOFTMAX:
		reshape_name = "mbox_conf_reshape"
		net[reshape_name] = L.Reshape(mbox_layers[1], \
			shape=dict(dim=[0, -1, ssdparam.get("num_classes",2)]))
		softmax_name = "mbox_conf_softmax"
		net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
		flatten_name = "mbox_conf_flatten"
		net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
     	mbox_layers[1] = net[flatten_name]
	# elif bbox_loss_param.get("multiloss_conf_loss_type",P.MultiBoxLoss.SOFTMAX) == P.MultiBoxLoss.LOGISTIC:
 #    	sigmoid_name = "mbox_conf_sigmoid"
 #    	net[sigmoid_name] = L.Sigmoid(mbox_layers[1])
 #    	mbox_layers[1] = net[sigmoid_name]
	if visualize:
		mbox_layers.append(net["data"])
	net.detection_out=L.DetOut(*mbox_layers,
		detection_output_param=det_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.detection_eval_accu = L.DetEval(net.detection_out,net.bbox,detection_evaluate_param=det_eval_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))

	return net

def MaskNet_Val_Pose(net, from_layer="data", label="label", lr=1, decay=1):

	# net = YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)
	net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=1)
	net.bbox,net.kps,net.mask = L.SplitLabel(net[label],name='SplitLabel',ntop=3)
	
	net.roi,net.kps_active_flags = L.TrueRoi(net[label],true_roi_param=dict(type='pose'),ntop=2)
	out_layer = "conv5_5"
	# net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=[128,128,128,128,128,128], \
 #        conv6_kernal_size=[3,3,3,3,3,3], pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1,n_group=1)
	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5","conv6_7"], tags=["Down","Ref","Up"], unifiedlayer="convf_mask", \
	#                             dnsampleMethod=[["MaxPool"]],upsampleMethod="Reorg")
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"], tags=["Down","Ref"], unifiedlayer="convf_mask", \
	                            dnsampleMethod=[["MaxPool"]])
	net.roi_maps = L.RoiAlign(net.convf_mask,net.roi,roi_align_param=roi_align_param)
	net = KpsHeader(net,from_layer="roi_maps",out_layer="kps_maps",use_layers=kps_use_conv_layers,num_channels=channels_of_kps, \
				    all_kernel_size=kernel_size_of_kps,pad=pad_of_kps,use_deconv_layers=kps_use_deconv_layers,lr=lr,decay=decay)
	net.kps_flatten = L.Flatten(net.kps_maps,flatten_param=dict(axis=2,end_axis=-1))
	net.kps_softmax = L.Softmax(net.kps_flatten,softmax_param=dict(axis=2))
	
	net.kps_peaks = L.PeaksFind(net.kps_softmax,peaks_find_param=dict(height=Rh_Kps,width=Rw_Kps))	

	net.kps_label_map, net.kps_label_flags = L.KpsGen(net.roi,net.kps,name="KpsGen",ntop=2,kps_gen_param=dict(resized_height=Rh_Kps,resized_width=Rw_Kps,use_softmax=True))
	
	net.kps_label=L.KpsLabel(net.kps_label_map,net.kps_label_flags,net.kps_active_flags,kps_gen_param=dict(resized_height=Rh_Kps,resized_width=Rw_Kps,use_softmax=True))

	net.pose_eval = L.KpsEval(net.kps_peaks,net.kps_label,net.roi,net.kps_active_flags,kps_eval_param=dict(conf_thre=0.3,distance_thre=0.01))

	return net

def MaskNet_Val_Mask(net, from_layer="data", label="label", lr=1, decay=1):
	# net = YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)
	net.bbox,net.kps,net.mask = L.SplitLabel(net[label],name='SplitLabel',ntop=3)
	net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=0.1)
	net.roi,net.kps_active_flags = L.TrueRoi(net[label],true_roi_param=dict(type='mask'),ntop=2)
	out_layer = "conv5_5"
	# net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=[128,128,128,128,128,128], \
 #        conv6_kernal_size=[3,3,3,3,3,3], pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1,n_group=1)
	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5","conv6_7"], tags=["Down","Ref","Up"], unifiedlayer="convf_mask", \
	#                             dnsampleMethod=[["MaxPool"]],upsampleMethod="Reorg")
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"], tags=["Down","Ref"], unifiedlayer="convf_mask", \
	                            dnsampleMethod=[["MaxPool"]])
	net.roi_maps=L.RoiAlign(net.convf_mask,net.roi,roi_align_param=roi_align_param)
	net = MaskHeader(net,from_layer="roi_maps",out_layer="mask_maps",use_layers=mask_use_conv_layers,num_channels=channels_of_mask, \
				    kernel_size=kernel_size_of_mask,pad=pad_of_mask,use_deconv_layers=mask_use_deconv_layers,lr=lr,decay=decay)
	net.mask_sigmoid = L.Sigmoid(net.mask_maps)
	net.pred = L.Threshold(net.mask_sigmoid,threshold_param=dict(threshold=0.5))
	net.mask_label_map, net.mask_label_flags = L.MaskGen(net.roi,net.mask,name="MaskGen",ntop=2, \
		mask_gen_param=dict(height=Input_Height,width=Input_Width,resized_height=Rh_Mask,resized_width=Rw_Mask))
	net.mask_eval = L.MaskEval(net.pred,net.mask_label_map,net.roi,net.kps_active_flags)
	return net

def MaskNet_Test(net, from_layer="data", image="image",lr=1, decay=1):
	# net = YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)
	net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,strid_conv=[1,1,1,0,0],final_pool=False,lr=0.1, decay=0.1)
	out_layer = "conv5_5"
	net = addconv6(net, from_layer=out_layer, use_bn=True, conv6_output=[128,128,128,128,128,128], \
        conv6_kernal_size=[3,3,3,3,3,3], pre_name="conv6",start_pool=True,lr_mult=1, decay_mult=1,n_group=1)
	# Concat [conv5_5, conv6_7]
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_6"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Reorg"]])
	# Concat [conv4_3, conv5_5, conv6_7]
	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5","conv6_7"], tags=["Down","Ref","Up"], unifiedlayer="convf_mask", \
	#                             dnsampleMethod=[["MaxPool"]],upsampleMethod="Reorg")
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"], tags=["Down","Ref"], unifiedlayer="convf_mask", \
	                            dnsampleMethod=[["MaxPool"]])
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2"],input_height=Input_Height,input_width=Input_Width,**ssdparam)
	
	net.detection_out=L.DetOut(*mbox_layers,detection_output_param=det_out_param,include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	
	net.roi_maps=L.RoiAlign(net.convf_mask,net.detection_out,roi_align_param=roi_align_param)
	net = KpsHeader(net,from_layer="roi_maps",out_layer="kps_maps",use_layers=kps_use_conv_layers,num_channels=channels_of_kps, \
				    all_kernel_size=kernel_size_of_kps,pad=pad_of_kps,use_deconv_layers=kps_use_deconv_layers,lr=lr,decay=decay)
	net.kps_flatten = L.Flatten(net.kps_maps,flatten_param=dict(axis=2,end_axis=-1))
	net.kps_softmax = L.Softmax(net.kps_flatten,softmax_param=dict(axis=2))
	net.kps_peaks = L.PeaksFind(net.kps_softmax,peaks_find_param=dict(height=Rh_Kps,width=Rw_Kps))

	net = MaskHeader(net,from_layer="roi_maps",out_layer="mask_maps",use_layers=mask_use_conv_layers,num_channels=channels_of_mask, \
				    kernel_size=kernel_size_of_mask,pad=pad_of_mask,use_deconv_layers=mask_use_deconv_layers,lr=lr,decay=decay)
	net.mask_sigmoid = L.Sigmoid(net.mask_maps)
	net.mask_thre = L.Threshold(net.mask_sigmoid,threshold_param=dict(threshold=0.5))
	net.vis=L.VisualMask(net.orig_data,net.detection_out,net.mask_thre,net.kps_peaks,**visual_mask_param)

	return net
def MTD_TEST(net,from_layer="data",image="image",lr=1,decay=1):
	# net =YoloNetPart(net,from_layer=from_layer,use_bn=True,use_layers=6,use_sub_layers=7,lr=lr,decay=decay)
	net,mbox_layers,parts_layers=MTD_BODY(net)


	# net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Reorg"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]],pad=True)
	# mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2"],input_height=Input_Height,input_width=Input_Width,loc_postfix='det',**ssdparam)
	reshape_name = "mbox_conf_reshape"
	net[reshape_name] = L.Reshape(mbox_layers[1], \
		shape=dict(dim=[0, -1, ssdparam.get("num_classes",2)]))
	softmax_name = "mbox_conf_softmax"
	net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
	flatten_name = "mbox_conf_flatten"
	net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
	mbox_layers[1] = net[flatten_name]
	# mbox_layers.append(net.orig_data)
	net.detection_out=L.DenseDetOut(*mbox_layers,
		detection_output_param=det_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))

	# net = UnifiedMultiScaleLayers(net,layers=["conv3_3","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# parts_layers = SSDHeader(net,data_layer="data",from_layers=["conf34","conv5_5","conv6_7"],input_height=Input_Height,input_width=Input_Width,loc_postfix='parts',**partsparam)
	sigmoid_name = "parts_conf_sigmoid"
	net[sigmoid_name] = L.Sigmoid(parts_layers[1])
	parts_layers[1] = net[sigmoid_name]

	net.parts_out=L.DenseDetOut(*parts_layers,
		detection_output_param=parts_out_param,
		include=dict(phase=caffe_pb2.Phase.Value('TEST')))
	net.roi=L.Concat(net.detection_out,net.parts_out,axis=2)
	net.vis=L.VisualMtd(net.roi,net.orig_data,detection_output_param=vis_out_param)
	return net
def MTD_BODY(net,from_layer="data",label="label",lr=1,decay=1):
	net = YoloNetPart(net, from_layer=from_layer, use_bn=True, use_layers=6, use_sub_layers=7, lr=lr, decay=decay)

	
	# ConvBNUnitLayer(net,"conv2", "conv2_pool", use_bn=True, use_relu=True, num_output=64,
 #            kernel_size=1, pad=0, stride=2) 
	# net = UnifiedMultiScaleLayers(net,layers=["conv2_pool","conv3_3","conv4_3"],tags=["Down","Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["Conv"],["MaxPool"]],dnsampleChannels=64)
	net = UnifiedMultiScaleLayers(net,layers=["conv4_3","conv5_5"],tags=["Down","Ref"],unifiedlayer="featuremap1",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conv5_5","conv6_7"],tags=["Down","Ref"],unifiedlayer="featuremap2",dnsampleMethod=[["MaxPool"]],pad=True)
	mbox_layers = SSDHeader(net,data_layer="data",from_layers=["featuremap1","featuremap2"],input_height=Input_Height,input_width=Input_Width,loc_postfix='det',**ssdparam)

	
	net = UnifiedMultiScaleLayers(net,layers=["conv2","conv3_3"],tags=["Down","Ref"],unifiedlayer="conf23",dnsampleMethod=[["Conv"]],dnsampleChannels=64)	
	net = UnifiedMultiScaleLayers(net,layers=["conf23","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	# net = UnifiedMultiScaleLayers(net,layers=["conv3_3","conv4_3"],tags=["Down","Ref"],unifiedlayer="conf34",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conf34","conv5_5"],tags=["Down","Ref"],unifiedlayer="conf45",dnsampleMethod=[["MaxPool"]])
	net = UnifiedMultiScaleLayers(net,layers=["conf45","conv6_7"],tags=["Down","Ref"],unifiedlayer="conf56",dnsampleMethod=[["MaxPool"]],pad=True)
	parts_layers = SSDHeader(net,data_layer="data",from_layers=["conf34","conf45","conf56"],input_height=Input_Height,input_width=Input_Width,loc_postfix='parts',**partsparam)
	
	
	return net,mbox_layers,parts_layers