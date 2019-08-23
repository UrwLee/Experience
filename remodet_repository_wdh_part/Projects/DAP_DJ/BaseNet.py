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
# ##############################BaseNet for Detection###########################
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
# ##############################################################################
# ------------------------------------------------------------------------------
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

# ResNet-Unit
def ResNetTwoLayers_UnitB(net, base_layer, name_prefix, stride, num_channel,num_channel_change = 0,
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
        add_layer1 = add_layer
        if stride == 2:
            from_layer = base_layer
            add_layer = name_prefix + '_bridge'
            ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                            num_output=num_channel, kernel_size=3, pad=1, stride=2, use_scale=True,lr_mult=lr,
                decay_mult=decay)
            add_layer2 = add_layer
        else:
            add_layer2 = base_layer
        add_layer = name_prefix + '_Add'
        net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
# ##############################################################################
# ------------------------------------------------------------------------------
# ResNet
def ResidualVariant_Base_B(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,lr=1,decay=1,add_strs=""):
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
            if not output_channel_layer == 0 and sublayer == use_sub_layers[layer] - 1:
                num_channel_change = output_channel_layer
            else:
                num_channel_change = 0
            ResNetTwoLayers_UnitB(net, base_layer, name_prefix, stride, num_channel_layer, num_channel_change=num_channel_change,
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False,lr=lr, decay=decay)
            out_layer = name_prefix + '_relu'
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
						num_output=32, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
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
						num_output=64, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
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
						num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=64, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv3_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
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
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 2
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_2'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 3
	if strid_conv[layers-1] == 1:
		stride_general = 2
	else:
		stride_general = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv4_3'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True, \
						num_output=256, kernel_size=3, pad=1, stride=stride_general, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	if final_pool or layers < use_layers and strid_conv[layers-1] == 0:
		out_layer = 'pool4'
		net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
			kernel_size=3, stride=2, pad=0)
		from_layer = out_layer

	layers = 5
	sub_layers = 1
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_1'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
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
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 4
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_4'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=128, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	sub_layers = 5
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_5'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer
	sub_layers = 6
	if use_layers >= layers and use_sub_layers >= sub_layers:
		out_layer = 'conv5_6'
		ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,
						num_output=512, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=True,lr_mult=lr, decay_mult=decay)
		from_layer = out_layer

	return net