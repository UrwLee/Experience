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
from InceptionLayer import *
from ConvBNLayer_decomp import *

def YoloNet(net, from_layer="data", lr=1, decay=1):

	out_layer = 'conv1'
	ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
		num_output=32, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
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

def YoloNetPart(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
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
			num_output=64, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool2'

		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=3, stride=2, pad=1	)
		from_layer = out_layer

	layers = 6
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 6
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 7
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_7'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
			num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

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

def YoloNetPart_Decomp(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,
					   final_pool=False, lr=1, decay=1, **decomp_kwargs):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv1'
		decomp_num = decomp_kwargs.get("conv1", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=32, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=32, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool1'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 2
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv2'
		decomp_num = decomp_kwargs.get("conv2", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=64, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool2'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 3
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_1'
		decomp_num = decomp_kwargs.get("conv3_1", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		decomp_num = decomp_kwargs.get("conv3_2", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=64, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		decomp_num = decomp_kwargs.get("conv3_3", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool3'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 4
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_1'
		decomp_num = decomp_kwargs.get("conv4_1", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		decomp_num = decomp_kwargs.get("conv4_2", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=128, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		decomp_num = decomp_kwargs.get("conv4_3", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		decomp_num = decomp_kwargs.get("conv5_1", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		decomp_num = decomp_kwargs.get("conv5_2", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		decomp_num = decomp_kwargs.get("conv5_3", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		decomp_num = decomp_kwargs.get("conv5_4", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=256, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		decomp_num = decomp_kwargs.get("conv5_5", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	layers = 6
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_1'
		decomp_num = decomp_kwargs.get("con6_1", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=1024, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_2'
		decomp_num = decomp_kwargs.get("con6_2", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_3'
		decomp_num = decomp_kwargs.get("con6_3", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=1024, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_4'
		decomp_num = decomp_kwargs.get("con6_4", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=512, \
			    kernel_size=1, pad=0, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=512, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_5'
		decomp_num = decomp_kwargs.get("con6_5", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=1024, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 6
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_6'
		decomp_num = decomp_kwargs.get("con6_6", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=1024, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 7
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv6_7'
		decomp_num = decomp_kwargs.get("con6_7", [])
		if decomp_num:
			assert len(decomp_num) == 2, "length of R must be 2."
			out_layer = ConvBNUnitLayer_decomp(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, num_output=1024, \
			    kernel_size=3, pad=1, stride=1, lr=lr, decay=decay, R1_channels=decomp_num[0], \
				R2_channels=decomp_num[1], use_scale=True, leaky=True, init_xavier=True)
		else:
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=1024, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	return net
def FaceNet(net, from_layer="data", use_bn=True, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
	assert use_layers >= 1
	assert use_sub_layers >= 1

	layers = 1
	sub_layers = 1

	out_layer = "conv1_L"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=24,kernel_size=7,pad=3,stride=4,use_scale=True,leaky=True)
	from_layer = out_layer
	
	out_layer = "pool1"
	net[out_layer] = L.Pooling(net[from_layer],pool = P.Pooling.MAX,kernel_size=3,stride=2,pad=1)
	from_layer = out_layer

	layers = 2
	sub_layers = 1

	out_layer="conv2"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=64,kernel_size=5,pad=2,stride=2,use_scale=True,leaky=True)
	from_layer = out_layer

	out_layer="pool2"
	net[out_layer] = L.Pooling(net[from_layer],pool = P.Pooling.MAX,kernel_size=3,stride=2,pad=0)
	from_layer = out_layer

	out_layer="inception1"
	# ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
	# 	num_output=64,kernel_size=3,pad=1,stride=1,use_scale=True,leaky=True)
	InceptionLayer(net, from_layer, out_layer, use_prior_bn=False, cross_stage=True, channels_1=32, \
                      channels_3=[24,32], channels_5=[24,32,32],channels_pool=32, use_out_conv=False, \
                      channels_output=128, lr=1, decay=1, out_bn=False, use_shortcut=False)
	from_layer = out_layer

	out_layer="inception2"
	# ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
	# 	num_output=64,kernel_size=3,pad=1,stride=1,use_scale=True,leaky=True)
	InceptionLayer(net, from_layer, out_layer, use_prior_bn=False, cross_stage=True, channels_1=32, \
                      channels_3=[24,32], channels_5=[24,32,32],channels_pool=32, use_out_conv=False, \
                      channels_output=128, lr=1, decay=1, out_bn=False, use_shortcut=False)
	from_layer = out_layer

	out_layer="inception3"
	# ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
	# 	num_output=64,kernel_size=3,pad=1,stride=1,use_scale=True,leaky=True)
	InceptionLayer(net, from_layer, out_layer, use_prior_bn=False, cross_stage=True, channels_1=32, \
                      channels_3=[24,32], channels_5=[24,32,32],channels_pool=32, use_out_conv=False, \
                      channels_output=128, lr=1, decay=1, out_bn=False, use_shortcut=False)
	from_layer = out_layer

	out_layer="conv3_1"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=128,kernel_size=1,pad=0,stride=1,use_scale=True,leaky=True)
	from_layer = out_layer

	out_layer="conv3_2"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=256,kernel_size=3,pad=1,stride=2,use_scale=True,leaky=True)
	from_layer = out_layer

	out_layer="conv4_1"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=128,kernel_size=1,pad=0,stride=1,use_scale=True,leaky=True)
	from_layer = out_layer

	out_layer="conv4_2"
	ConvBNUnitLayer(net,from_layer,out_layer,use_bn=use_bn,use_relu=True,lr_mult=lr,decay_mult=decay,
		num_output=256,kernel_size=3,pad=1,stride=2,use_scale=True,leaky=True)
	from_layer = out_layer

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
def addconv6(net, from_layer="data", use_bn=False, conv6_output=[],conv6_kernal_size=[],start_pool=False,lr_mult=1, decay_mult=1,pre_name="conv6",post_name="",n_group=1):


    if start_pool:
        out_layer = 'pool5'
        net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
            kernel_size=2, stride=2, pad=0)
        from_layer = out_layer

    assert len(conv6_output) == len(conv6_kernal_size)

    for i in range(len(conv6_output)):
        out_layer = '{}_{}{}'.format(pre_name,i+1,post_name)
        num_output = conv6_output[i]
        kernel_size = conv6_kernal_size[i]
        if kernel_size == 3:
            pad = 1
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
                num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=True,lr_mult=lr_mult, decay_mult=decay_mult,n_group=n_group)
        else:
            pad = 0
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
                num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=True,lr_mult=lr_mult, decay_mult=decay_mult,n_group=1)
        from_layer = out_layer

    return net
def YoloNetPartCompress(net, from_layer="data", use_bn=False, use_layers=6, use_sub_layers=7, final_pool=False, strid_conv= [],lr=1, decay=1):
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
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
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
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
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
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
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
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
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
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 3
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=256, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True)
		from_layer = out_layer

	sub_layers = 5
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, lr_mult=lr, decay_mult=decay,\
						num_output=512, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True)
		from_layer = out_layer

	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool5'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=2, stride=2, pad=0)
		from_layer = out_layer

	return net