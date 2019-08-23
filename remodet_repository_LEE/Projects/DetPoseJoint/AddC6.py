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
# ##############################################################################
# ##############################Add Conv6 for BaseNet###########################
# ##############################################################################
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
				num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=leaky,lr_mult=lr_mult, decay_mult=decay_mult,n_group=n_group,constant_value=0.2)
		else:
			pad = 0
			ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
				num_output=num_output, kernel_size=kernel_size, pad=pad, stride=1, use_scale=True, leaky=leaky,lr_mult=lr_mult, decay_mult=decay_mult,n_group=1,constant_value=0.2)
		from_layer = out_layer
	return net
