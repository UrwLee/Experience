# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *

from DeconvLayer import *

def YoloNet(net, from_layer="data", lr=1, decay=1):

	out_layer = 'conv1'

	from_layer = out_layer

	out_layer = 'pool1'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=64, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool2'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv3_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool3'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv4_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv4_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv4_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool4'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv5_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_4'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_5'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool5'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv6_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv6_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv6_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv6_4'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv6_5'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv6_6'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	# out_layer = 'conv6_7'
	# ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
	# 	num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	# from_layer = out_layer

	return net

def YoloNetPart(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, lr=1, decay=1,leaky=True,
				ChangeNameAndChannel={},special_layers=""):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 32
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	if final_pool or layers < use_layers:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 64
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	if final_pool or layers < use_layers:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 128
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 64
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 128
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	if final_pool or layers < use_layers:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 256
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 128
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 256
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	if final_pool or layers < use_layers:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 512
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 256
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 512
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 256
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		if out_layer == special_layers:
			flag_bninplace = False
		else:
			flag_bninplace = True
		if out_layer in ChangeNameAndChannel.keys():
			num_output = ChangeNameAndChannel[out_layer]
			out_layer += "_new"
		else:
			num_output = 512
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=num_output, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
		if flag_bninplace:
			from_layer = out_layer
		else:
			from_layer = out_layer + "_bn"

	if final_pool or layers < use_layers:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 6
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 6
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 7
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_7'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	return net

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
############################################################
def  YoloNetPartCompress(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, leaky=True,
						 strid_conv= [],lr=1, decay=1,kernel_size_first = 3,stride_first=2,channel_divides = (1,1,1,1,1),num_channel_conv5_5=512):
	assert use_layers >= 1
	assert use_sub_layers >= 1
	layers = 1
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size_first, pad=(kernel_size_first-1)/2, stride=stride_first, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64/channel_divides[layers-1], kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128/channel_divides[layers-1], kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64/channel_divides[layers-1], kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128/channel_divides[layers-1], kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256/channel_divides[layers-1], kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128/channel_divides[layers-1], kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256/channel_divides[layers-1], kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer


	layers = 5
	num_channel = 512 / channel_divides[layers-1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=num_channel, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=num_channel/2, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=num_channel, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=num_channel/2, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=num_channel_conv5_5, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer
	return net

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with depthwise convolution
############################################################
def YoloNetPartCompressDepthwise(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, kernel_sizes = [3,3,3,3,3],
								 final_pool=False, strid_conv= [],group_divide =1, lr=1, decay=1, addstrs = ''):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers-1]
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2_dw' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
						stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32/group_divide)
		from_layer = out_layer
		out_layer = 'conv2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1, pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 64/group_divide)
		from_layer = out_layer
		out_layer = 'conv3_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 128/group_divide)
		from_layer = out_layer
		out_layer = 'conv3_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 128/group_divide)
		from_layer = out_layer
		out_layer = 'conv4_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256/group_divide)
		from_layer = out_layer
		out_layer = 'conv4_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256/group_divide)
		from_layer = out_layer
		out_layer = 'conv5_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512/group_divide)
		from_layer = out_layer
		out_layer = 'conv5_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512/group_divide)
		from_layer = out_layer
		out_layer = 'conv5_6' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_7' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512/group_divide)
		from_layer = out_layer
		out_layer = 'conv5_8' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	return net

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with depthwise convolution
############################################################
def YoloNetPartCompressDepthwiseD(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, kernel_sizes = [3,3,3,3,3],
								 final_pool=False, strid_conv= [],group_divide =1, lr=1, decay=1, addstrs = ''):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers-1]
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2_dw' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
						stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1, pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv3_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 4)
		from_layer = out_layer
		out_layer = 'conv3_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 64)
		from_layer = out_layer
		out_layer = 'conv4_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 4)
		from_layer = out_layer
		out_layer = 'conv4_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 128)
		from_layer = out_layer
		out_layer = 'conv5_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 4)
		from_layer = out_layer
		out_layer = 'conv5_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 4)
		from_layer = out_layer
		out_layer = 'conv5_6' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_7' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 4)
		from_layer = out_layer
		out_layer = 'conv5_8' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	return net

def YoloNetPartCompressDepthwiseE(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, kernel_sizes = [3,3,3,3,3],
								 final_pool=False, strid_conv= [],group_divide =1, lr=1, decay=1, addstrs = ''):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers-1]
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2_dw' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
						stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1, pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv3_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv3_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 64)
		from_layer = out_layer
		out_layer = 'conv4_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv4_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 128)
		from_layer = out_layer
		out_layer = 'conv5_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv5_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv5_6' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_7' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv5_8' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	return net

def YoloNetPartCompressDepthwiseF(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, kernel_sizes = [3,3,3,3,3],
								 final_pool=False, strid_conv= [],group_divide =1, lr=1, decay=1, addstrs = ''):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers-1]
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2_dw' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
						stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 8)
		from_layer = out_layer
		out_layer = 'conv2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1, pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv3_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv3_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32)
		from_layer = out_layer
		out_layer = 'conv4_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv4_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 64)
		from_layer = out_layer
		out_layer = 'conv5_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv5_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv5_6' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_7' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 16)
		from_layer = out_layer
		out_layer = 'conv5_8' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	return net
############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with depthwise convolution
############################################################
def YoloNetPartCompressDepthwisePartial(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7,
								 kernel_sizes=[3, 3, 3, 3, 3], final_pool=False, strid_conv=[], group_divide=1, lr=1, decay=1, addstrs=''):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers - 1]
	if strid_conv[layers - 1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=stride_general,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers - 1] == 0:
		out_layer = 'pool1' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
								   kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if strid_conv[layers - 1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2_dw' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=32,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2,
						stride=stride_general, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay,
						n_group=32 / group_divide)
		from_layer = out_layer
		out_layer = 'conv2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1,
						pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers - 1] == 0:
		out_layer = 'pool2' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
								   kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=64 / group_divide)
		from_layer = out_layer
		out_layer = 'conv3_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1,
						pad=0, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers - 1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=stride_general,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=128 / group_divide)
		from_layer = out_layer
		out_layer = 'conv3_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1,
						pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers - 1] == 0:
		out_layer = 'pool3' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
								   kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=128 / group_divide)
		from_layer = out_layer
		out_layer = 'conv4_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1,
						pad=0, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if strid_conv[layers - 1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=stride_general,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=256 / group_divide)
		from_layer = out_layer
		out_layer = 'conv4_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1,
						pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers - 1] == 0:
		out_layer = 'pool4' + addstrs
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
								   kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=256 / group_divide)
		from_layer = out_layer
		out_layer = 'conv5_2' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1,
						pad=0, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=512 / group_divide)
		from_layer = out_layer
		out_layer = 'conv5_4' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1,
						pad=0, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=512 / group_divide)
		from_layer = out_layer
		out_layer = 'conv5_6' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1,
						pad=0,
						stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if strid_conv[layers - 1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_7' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256,
						kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=1)
		from_layer = out_layer
		out_layer = 'conv5_8' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=3,
						pad=1,stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)

# if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
# 	out_layer = 'pool5'
# 	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
# 		kernel_size=2, stride=2, pad=0)
# 	from_layer = out_layer

	return net

############################################################
# Realize DepthWise as the paper
############################################################
def YoloNetPartCompressDepthwiseAsPaper(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, kernel_sizes = [3,3,3,3,3],group_divide =1,
										expandlastconv=False,lr=1, decay=1, addstrs = ''):
	assert use_layers >= 1
	assert use_sub_layers >= 1
	###### layer ONE
	layers = 1
	sub_layers = 1
	kernel_size = kernel_sizes[layers - 1]
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1' + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
						stride=2, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	###### layer TWO
	layers = 2
	kernel_size = kernel_sizes[layers - 1]
	sub_layers = 1
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=32, kernel_size=kernel_size, pad=(kernel_size-1)/2,
					stride=1, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 32/group_divide)
	from_layer = out_layer

	sub_layers = 2
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	sub_layers = 3
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=kernel_size,
					pad=(kernel_size - 1) / 2,stride=2, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=64 / group_divide)
	from_layer = out_layer

	###### layer THREE
	layers = 3
	kernel_size = kernel_sizes[layers - 1]
	nchannel = 128
	sub_layers = 1
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	sub_layers = 2
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=kernel_size,
					pad=(kernel_size - 1) / 2, stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay,
					n_group=nchannel / group_divide)
	from_layer = out_layer

	sub_layers = 3
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	sub_layers = 4
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel,kernel_size=kernel_size,
					pad=(kernel_size - 1) / 2, stride=2, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay,
					n_group=nchannel / group_divide)
	from_layer = out_layer

	###### layer FOUR
	layers = 4
	kernel_size = kernel_sizes[layers - 1]
	nchannel = 256
	sub_layers = 1
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	sub_layers = 2
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel,
					kernel_size=kernel_size,
					pad=(kernel_size - 1) / 2, stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay,
					n_group=nchannel / group_divide)
	from_layer = out_layer

	sub_layers = 3
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	sub_layers = 4
	out_layer = 'conv{}_{}_dw'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel,kernel_size=kernel_size,
					pad=(kernel_size - 1) / 2, stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay,
					n_group=nchannel / group_divide)
	from_layer = out_layer
	out_layer = 'pool4' + addstrs
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	#### layers FIVE
	layers = 5
	kernel_size = kernel_sizes[layers - 1]
	nchannel = 512
	sub_layers = 0
	out_layer = 'conv{}_{}'.format(layers, sub_layers) + addstrs
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1, pad=0,
					stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	from_layer = out_layer

	for sub_layers in xrange(1, use_sub_layers + 1):

		out_layer = 'conv{}_{}_dw'.format(layers, sub_layers*2 - 1) + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel,
						kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay, n_group=nchannel / group_divide)
		from_layer = out_layer
		if use_sub_layers == sub_layers and expandlastconv:
			nchannel *= 2
		out_layer = 'conv{}_{}'.format(layers, sub_layers*2) + addstrs
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=nchannel, kernel_size=1,
						pad=0,stride=1, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
	return net
############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with dept hwise convolution
############################################################
def YoloNetPartCompressDepthwiseA(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=3, pad=1,
						stride=stride_general, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512)
		from_layer = out_layer
		out_layer = 'conv5_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)

	# if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
	# 	out_layer = 'pool5'
	# 	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
	# 		kernel_size=2, stride=2, pad=0)
	# 	from_layer = out_layer

	return net

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with dept hwise convolution
############################################################
def YoloNetPartCompressDepthwiseB(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=3, pad=1,
						stride=stride_general, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=64, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=64)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=128)
		from_layer = out_layer
		out_layer = 'conv3_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, kernel_size=1, pad=0,	stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=128)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=256)
		from_layer = out_layer
		out_layer = 'conv4_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,	stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512)
		from_layer = out_layer
		out_layer = 'conv5_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)

	# if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
	# 	out_layer = 'pool5'
	# 	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
	# 		kernel_size=2, stride=2, pad=0)
	# 	from_layer = out_layer

	return net

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# replace each standard convolution with dept hwise convolution
############################################################
def YoloNetPartCompressDepthwiseC(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, kernel_size=3, pad=1,
						stride=stride_general, use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer


	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=128)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group=256)
		from_layer = out_layer
		out_layer = 'conv4_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, kernel_size=1, pad=0,	stride=1,
						use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=256, kernel_size=3, pad=1, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 256)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=1, pad=0, stride=1,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=512, kernel_size=3, pad=1, stride=stride_general,
						use_scale=True, leaky=True,lr_mult=lr,decay_mult=decay,n_group = 512)
		from_layer = out_layer
		out_layer = 'conv5_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, kernel_size=1, pad=0,
						stride=1,use_scale=True, leaky=True, lr_mult=lr, decay_mult=decay)

	# if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
	# 	out_layer = 'pool5'
	# 	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
	# 		kernel_size=2, stride=2, pad=0)
	# 	from_layer = out_layer

	return net
############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# change the layers of Conv5 to three layers
############################################################
def YoloNetPartCompressConv5ThreeLayers(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_{}'.format(sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=4, decay_mult=4)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_{}'.format(sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=4, decay_mult=4)
		from_layer = out_layer


	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_{}'.format(sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=512, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=4, decay_mult=4)
		from_layer = out_layer

	return net
############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
# Reduce the conv5 layers and add another 3x3 layer
############################################################
def YoloNetPartCompressReduceConv5(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer
	sub_layers = 6
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	return net
############################################################
# needs to be written
############################################################
def YoloNetPartCompressResid(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if final_pool or layers < use_layers:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if final_pool or layers < use_layers:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	net['conv2_reorg'] = L.Reorg(net[from_layer], reorg_param=dict(up_down=P.Reorg.DOWN))
	ConvBNUnitLayer(net, 'conv2_reorg', 'conv2_reorg/adapt', use_bn=use_bn, use_relu=False,
					num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	layers = 3
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if final_pool or layers < use_layers:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False, \
						num_output=128, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	out_layer = 'residual1'
	net[out_layer] = L.Eltwise(net['conv2_reorg/adapt'], net[from_layer], eltwise_param=dict(operation=P.Eltwise.SUM))
	from_layer = out_layer
	out_layer = 'residual1_relu'
	net[out_layer] = L.ReLU(net[from_layer], in_place=True)
	from_layer = out_layer
	net['conv3_reorg'] = L.Reorg(net[from_layer], reorg_param=dict(up_down=P.Reorg.DOWN))
	ConvBNUnitLayer(net, 'conv3_reorg', 'conv3_reorg/adapt', use_bn=use_bn, use_relu=False,
					num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if final_pool or layers < use_layers:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False, \
						num_output=256, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	out_layer = 'residual2'
	net[out_layer] = L.Eltwise(net['conv3_reorg/adapt'], net[from_layer], eltwise_param=dict(operation=P.Eltwise.SUM))
	from_layer = out_layer
	out_layer = 'residual2_relu'
	net[out_layer] = L.ReLU(net[from_layer], in_place=True)
	from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=384, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=192, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False, \
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	out_layer = 'residual3'
	net[out_layer] = L.Eltwise(net['residual2_relu'], net[from_layer], eltwise_param=dict(operation=P.Eltwise.SUM))
	from_layer = out_layer
	out_layer = 'residual3_relu'
	net[out_layer] = L.ReLU(net[from_layer], in_place=True)
	from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 5
	if final_pool or layers < use_layers:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=384, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer


	return net

def YoloNetPartDeconvResid(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=32, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=64, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers,sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1

	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=True)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=512, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=True)
		from_layer = out_layer
	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		DeconvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=512, kernel_size=4, pad=1, stride=2, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
						num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)

	net['concat1/32'] = L.Eltwise(net['conv4_3'], net[out_layer], eltwise_param=dict(operation=P.Eltwise.SUM))
	net['concat1/32_relu'] = L.ReLU(net['concat1/32'], in_place=True)
	from_layer = 'concat1/32'
	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv{}_{}'.format(layers, sub_layers)
		DeconvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=False,
						num_output=256, kernel_size=4, pad=1, stride=2, use_scale=True, leaky=True)
	net['concat1/16'] = L.Eltwise(net['conv4_1'], net[out_layer], eltwise_param=dict(operation=P.Eltwise.SUM))
	net['concat1/16_relu'] = L.ReLU(net['concat1/16'], in_place=True)


	return net
def TinyYoloNet(net, from_layer="data", use_bn=False, lr=1, decay=1):
	out_layer = 'conv1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool1'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool2'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv3_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_4'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_5'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=16, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv3_6'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool3'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv4_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv4_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv4_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=32, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv4_4'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'pool4'
	net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
		kernel_size=2, stride=2, pad=0)
	from_layer = out_layer

	out_layer = 'conv5_1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_2'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_3'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_4'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	out_layer = 'conv5_5'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
		num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
	from_layer = out_layer

	return net
