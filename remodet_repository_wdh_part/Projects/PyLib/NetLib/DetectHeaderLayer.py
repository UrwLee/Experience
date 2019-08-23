# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

from ConvBNLayer import *

# create loc_conv / conf_conv / prior_box layers
# num_classes -> numclasses
# feature_layer -> detector from which feature-layer
# use_objectness -> weather to create a conv-layer for evaluating an object
# normalizations: if a norm-layer is used before loc/conf evaluation
# use_batchnorm: conv + BN?
# prior_variance: box coding
# min_sizes/max_sizes/aspect_ratios: boxes proposed by RPN
# pro_widths/pro_heights: boxes proposed by given w/h
# share_location: boxes for all classes
# flip: boxes proposed by RPN to use flip ar -> 1./ar
# clip: clip all prior-boxes to [0,1]
# inter_layer_channels: a 3/1/1 conv-layer is used before outputs, channels
# kernel_size/pad: conv-layer ksize/pad
# conf_postfix/loc_postfix: naming of conf and loc layer
# return:
# >>>[loc_layer, conf_layer, prior_layer, <objectness_layer>]
# loc_layer: [n,h,w,(num_per_locations * 4)]
# conf_layer: [n,h,w,(num_per_locations * num_classes)]
# prior_layer: [n,2,locations*4]
# objectness_layer: [n,h,w,(num_per_locations * 2)]


def UnitLayerDetectorHeader(net, data_layer="data", num_classes=2, feature_layer="conv5", \
        use_objectness=False, normalization=-1, use_batchnorm=True, prior_variance = [0.1], \
        min_sizes=[], max_sizes=[], aspect_ratios=[], pro_widths=[], pro_heights=[], \
        share_location=True, flip=True, clip=False, inter_layer_channels=0, kernel_size=1, \
        pad=0, conf_postfix='', loc_postfix='', flat=False, use_focus_loss=False,stage=1):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"

    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers."
    assert feature_layer in net_layers, "feature_layer is not in net's layers."

    if min_sizes:
        assert not pro_widths, "pro_widths should not be provided when using min_sizes."
        assert not pro_heights, "pro_heights should not be provided when using min_sizes."
        if max_sizes:
            assert len(max_sizes) == len(min_sizes), "min_sizes and max_sizes must have the same legnth."
    else:
        assert pro_widths, "Must provide proposed width/height."
        assert pro_heights, "Must provide proposed width/height."
        assert len(pro_widths) == len(pro_heights), "pro_widths/heights must have the same length."
        assert not min_sizes, "min_sizes should be not provided when using pro_widths/heights."
        assert not max_sizes, "max_sizes should be not provided when using pro_widths/heights."

    from_layer = feature_layer
    prefix_name = '{}_{}'.format(from_layer,stage)
    # Norm-Layer
    if normalization != -1:
        norm_name = "{}_{}_norm".format(prefix_name,stage)
        net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalization), \
            across_spatial=False, channel_shared=False)
        from_layer = norm_name

    # Add intermediate Conv layers.
    # if inter_layer_channels > 0:
    #     inter_name = "{}_inter".format(from_layer)
    #     ConvBNUnitLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, \
    #         num_output=inter_layer_channels, kernel_size=kernel_size, pad=pad, stride=1,use_scale=True, leaky=True)
    #     from_layer = inter_name
    if len(inter_layer_channels) > 0:
        start_inter_id = 1
        for inter_channel_kernel in inter_layer_channels:
            inter_channel = inter_channel_kernel[0]
            inter_kernel = inter_channel_kernel[1]
            inter_name = "{}_inter_{}".format(prefix_name,start_inter_id)
            if inter_kernel == 1:
                inter_pad = 0
            elif inter_kernel == 3:
                inter_pad = 1
            ConvBNUnitLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, \
                num_output=inter_channel, kernel_size=inter_kernel, pad=inter_pad, stride=1,use_scale=True, leaky=False)
            from_layer = inter_name
            start_inter_id = start_inter_id + 1
    # Estimate number of priors per location given provided parameters.
    if min_sizes:
        if aspect_ratios:
            num_priors_per_location = len(aspect_ratios) + 1
            if flip:
                num_priors_per_location += len(aspect_ratios)
            if max_sizes:
                num_priors_per_location += 1
            num_priors_per_location *= len(min_sizes)
        else:
            if max_sizes:
                num_priors_per_location = 2 * len(min_sizes)
            else:
                num_priors_per_location = len(min_sizes)
    else:
        num_priors_per_location = len(pro_widths)

    # Create location prediction layer.
    name = "{}_mbox_loc{}".format(prefix_name, loc_postfix)
    num_loc_output = num_priors_per_location * 4 * (num_classes-1)
    if not share_location:
        num_loc_output *= num_classes
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_loc_output, kernel_size=3, pad=1, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layer = net[flatten_name]
    else:
        loc_layer = net[permute_name]

    # Create confidence prediction layer.
    name = "{}_mbox_conf{}".format(prefix_name, conf_postfix)
    num_conf_output = num_priors_per_location * num_classes
    if use_focus_loss:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,init_xavier=False,bias_type='focal',sparse=num_classes)
    else:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layer = net[flatten_name]
    else:
        conf_layer = net[permute_name]

    # Create prior generation layer.
    name = "{}_mbox_priorbox".format(prefix_name)
    if min_sizes:
        if aspect_ratios:
            if max_sizes:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes, max_size=max_sizes, \
                    aspect_ratio=aspect_ratios, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes, \
                    aspect_ratio=aspect_ratios, flip=flip, clip=clip, variance=prior_variance)
        else:
            if max_sizes:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes, max_size=max_sizes, \
                    flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes, \
                    flip=flip, clip=clip, variance=prior_variance)
        priorbox_layer = net[name]
    else:
        net[name] = L.PriorBox(net[from_layer], net[data_layer], pro_width=pro_widths, pro_height=pro_heights, \
            flip=flip, clip=clip, variance=prior_variance)
        priorbox_layer = net[name]

    # Create objectness prediction layer.
    if use_objectness:
        name = "{}_mbox_objectness".format(prefix_name)
        num_obj_output = num_priors_per_location * 2
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        if flat:
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layer = net[flatten_name]
        else:
            objectness_layer = net[permute_name]

    if use_objectness:
        return loc_layer,conf_layer,priorbox_layer,objectness_layer
    else:
        return loc_layer,conf_layer,priorbox_layer

def MultiLayersDetectorHeader(net, data_layer="data", num_classes=2, from_layers=[], \
        use_objectness=False, normalizations=[], use_batchnorm=True, prior_variance = [0.1], \
        min_sizes=[], max_sizes=[], aspect_ratios=[], pro_widths=[], pro_heights=[], \
        share_location=True, flip=True, clip=False, inter_layer_channels=[], \
        kernel_size=1, pad=0, conf_postfix='', loc_postfix='', use_focus_loss=False, stage=1):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    if min_sizes:
        assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
        if max_sizes:
            assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    else:
        assert len(from_layers) == len(pro_widths), "from_layers and pro_widths should have same length"
        assert len(from_layers) == len(pro_heights), "from_layers and pro_heights should have same length"
    if inter_layer_channels:
        assert len(from_layers) == len(inter_layer_channels), "from_layers and inter_layer_channels should have the same length"

    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []

    for i in range(0, num):
        # get feature layer
        from_layer = from_layers[i]
        # get sizes of prior-box layer
        minsizes = []
        maxsizes = []
        aspectratios = []
        prowidths = []
        proheights = []
        if min_sizes:
            minsizes = min_sizes[i] if type(min_sizes[i]) is list else [min_sizes[i]]
            if max_sizes:
                maxsizes = max_sizes[i] if type(max_sizes[i]) is list else [max_sizes[i]]
            if aspect_ratios:
                aspectratios = aspect_ratios[i] if type(aspect_ratios[i]) is list else [aspect_ratios[i]]
        else:
            prowidths = pro_widths[i] if type(pro_widths[i]) is list else [pro_widths[i]]
            proheights = pro_heights[i] if type(pro_heights[i]) is list else [pro_heights[i]]
        # get norm value
        normalization = -1
        if normalizations:
            normalization = normalizations[i]
        # get inter_layer_depth
        inter_layer_depth = 0
        if inter_layer_channels:
            inter_layer_depth = inter_layer_channels[i]
        if use_objectness:
            loc_layer,conf_layer,priorbox_layer,objectness_layer = \
                UnitLayerDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                    feature_layer=from_layer, use_objectness=True, \
                    normalization=normalization, use_batchnorm=use_batchnorm, \
                    prior_variance = prior_variance, min_sizes=minsizes, max_sizes=maxsizes, \
                    aspect_ratios=aspectratios, pro_widths=prowidths, pro_heights=proheights, \
                    share_location=share_location, flip=flip, clip=clip, \
                    inter_layer_channels=inter_layer_depth, \
                    kernel_size=1, pad=0, conf_postfix='', loc_postfix='', flat=True, stage=stage)
            loc_layers.append(loc_layer)
            conf_layers.append(conf_layer)
            priorbox_layers.append(priorbox_layer)
            objectness_layers.append(objectness_layer)
        else:
            loc_layer,conf_layer,priorbox_layer = \
                UnitLayerDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                    feature_layer=from_layer, \
                    normalization=normalization, use_batchnorm=use_batchnorm, \
                    prior_variance = prior_variance, min_sizes=minsizes, max_sizes=maxsizes, \
                    aspect_ratios=aspectratios, pro_widths=prowidths, pro_heights=proheights, \
                    share_location=share_location, flip=flip, clip=clip, \
                    inter_layer_channels=inter_layer_depth, \
                    kernel_size=kernel_size, pad=pad, conf_postfix='', \
                    loc_postfix=loc_postfix, flat=True, use_focus_loss=use_focus_loss,stage=stage)
            loc_layers.append(loc_layer)
            conf_layers.append(conf_layer)
            priorbox_layers.append(priorbox_layer)

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_{}_loc".format(stage)
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_conf".format(stage)
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_priorbox".format(stage)
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_{}_objectness".format(stage)
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

# MC-detector
def McDetectorHeader(net, num_classes=1, feature_layer="conv5", \
        normalization=-1, use_batchnorm=False,
        boxsizes=[], aspect_ratios=[], pwidths=[], pheights=[], \
        inter_layer_channels=0, kernel_size=1, pad=0):

    assert num_classes > 0, "num_classes must be positive number"

    net_layers = net.keys()
    assert feature_layer in net_layers, "feature_layer is not in net's layers."

    if boxsizes:
        assert not pwidths, "pwidths should not be provided when using boxsizes."
        assert not pheights, "pheights should not be provided when using boxsizes."
        assert aspect_ratios, "aspect_ratios should be provided when using boxsizes."
    else:
        assert pwidths, "Must provide proposed width/height."
        assert pheights, "Must provide proposed width/height."
        assert len(pwidths) == len(pheights), "provided widths/heights must have the same length."
        assert not boxsizes, "boxsizes should be not provided when using pro_widths/heights."
        assert not aspect_ratios, "aspect_ratios should be not provided when using pro_widths/heights."

    from_layer = feature_layer
    loc_conf_layers = []

    # Norm-Layer
    if normalization > 0:
        norm_name = "{}_norm".format(from_layer)
        net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalization), \
            across_spatial=False, channel_shared=False)
        from_layer = norm_name

    # Add intermediate Conv layers.
    if inter_layer_channels > 0:
        inter_name = "{}_inter".format(from_layer)
        ConvBNUnitLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, \
            num_output=inter_layer_channels, kernel_size=3, pad=1, stride=1)
        from_layer = inter_name

    # Estimate number of priors per location given provided parameters.
    if boxsizes:
        num_priors_per_location = len(aspect_ratios) * len(boxsizes) + 1
    else:
        num_priors_per_location = len(pwidths) + 1

    # Create location prediction layer.
    name = "{}_loc".format(from_layer)
    num_loc_output = num_priors_per_location * 4
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    loc_conf_layers.append(net[permute_name])

    # Create confidence prediction layer.
    name = "{}_conf".format(from_layer)
    num_conf_output = num_priors_per_location * (num_classes + 1)
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    loc_conf_layers.append(net[permute_name])

    return loc_conf_layers

def UnitLayerDenseDetectorHeader(net, data_layer="data", num_classes=2, feature_layer="conv5", \
        normalization=-1, use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], share_location=True, flip=True, clip=True, \
        inter_layer_channels=0, kernel_size=1, pad=0, conf_postfix='', loc_postfix='',\
        flat=False, use_focus_loss=False,stage=1):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers."
    assert feature_layer in net_layers, "feature_layer is not in net's layers."
    assert pro_widths, "Must provide proposed width/height."
    assert pro_heights, "Must provide proposed width/height."
    assert len(pro_widths) == len(pro_heights), "pro_widths/heights must have the same length."
    from_layer = feature_layer
    prefix_name = '{}_{}'.format(from_layer,stage)
    # Norm-Layer
    if normalization != -1:
        norm_name = "{}_norm".format(prefix_name)
        net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalization), \
            across_spatial=False, channel_shared=False)
        from_layer = norm_name
    # InterLayers
    if len(inter_layer_channels) > 0:
        start_inter_id = 1
        for inter_channel_kernel in inter_layer_channels:
            inter_channel = inter_channel_kernel[0]
            inter_kernel = inter_channel_kernel[1]
            inter_name = "{}_inter_{}".format(prefix_name,start_inter_id)
            if inter_kernel == 1:
                inter_pad = 0
            elif inter_kernel == 3:
                inter_pad = 1
            ConvBNUnitLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, \
                num_output=inter_channel, kernel_size=inter_kernel, pad=inter_pad, stride=1,use_scale=True, leaky=False)
            from_layer = inter_name
            start_inter_id = start_inter_id + 1
    # PriorBoxes
    num_priors_per_location = len(pro_widths)
    # LOC
    name = "{}_mbox_loc{}".format(prefix_name, loc_postfix)
    num_loc_output = num_priors_per_location * 4 * (num_classes-1)
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_loc_output, kernel_size=3, pad=1, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layer = net[flatten_name]
    else:
        loc_layer = net[permute_name]
    # CONF
    name = "{}_mbox_conf{}".format(prefix_name, conf_postfix)
    num_conf_output = num_priors_per_location * num_classes
    if use_focus_loss:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,init_xavier=False,bias_type='focal',sparse=num_classes)
    else:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1)
    permute_name = "{}_perm".format(name)
    net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layer = net[flatten_name]
    else:
        conf_layer = net[permute_name]
    # PRIOR
    name = "{}_mbox_priorbox".format(prefix_name)
    net[name] = L.PriorBox(net[from_layer], net[data_layer], pro_width=pro_widths, pro_height=pro_heights, \
        flip=flip, clip=clip, variance=prior_variance)
    priorbox_layer = net[name]
    return loc_layer,conf_layer,priorbox_layer

def MultiLayersDenseDetectorHeader(net, data_layer="data", num_classes=2, from_layers=[], \
        normalizations=[], use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], share_location=True, flip=True, clip=True, \
        inter_layer_channels=[], kernel_size=1, pad=0, conf_postfix='', loc_postfix='', \
        use_focus_loss=False,stage=1):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(pro_widths), "from_layers and pro_widths should have same length"
    assert len(from_layers) == len(pro_heights), "from_layers and pro_heights should have same length"
    if inter_layer_channels:
        assert len(from_layers) == len(inter_layer_channels), "from_layers and inter_layer_channels should have the same length"

    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []

    for i in range(0, num):
        # get feature layer
        from_layer = from_layers[i]
        # get sizes of prior-box layer
        prowidths = []
        proheights = []
        prowidths = pro_widths[i] if type(pro_widths[i]) is list else [pro_widths[i]]
        proheights = pro_heights[i] if type(pro_heights[i]) is list else [pro_heights[i]]
        # get norm value
        normalization = -1
        if normalizations:
            normalization = normalizations[i]
        # get inter_layer_depth
        inter_layer_depth = 0
        if inter_layer_channels:
            inter_layer_depth = inter_layer_channels[i]
        # consist mbox layer
        loc_layer,conf_layer,priorbox_layer = \
            UnitLayerDenseDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                feature_layer=from_layer, \
                normalization=normalization, use_batchnorm=use_batchnorm, \
                prior_variance = prior_variance, pro_widths=prowidths, pro_heights=proheights, \
                share_location=share_location, flip=flip, clip=clip, \
                inter_layer_channels=inter_layer_depth, \
                kernel_size=kernel_size, pad=pad, conf_postfix='', \
                loc_postfix=loc_postfix, flat=True, use_focus_loss=use_focus_loss, \
                stage=stage)
        loc_layers.append(loc_layer)
        conf_layers.append(conf_layer)
        priorbox_layers.append(priorbox_layer)

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_{}_loc".format(stage)
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_conf".format(stage)
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_priorbox".format(stage)
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    return mbox_layers
