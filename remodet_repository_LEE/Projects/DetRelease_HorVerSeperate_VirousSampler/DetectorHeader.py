# -*- coding: utf-8 -*-
import os
import caffe
import math
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
sys.path.append('../')
from PyLib.NetLib.ConvBNLayer import *
####################################################################################################
#####################################Create Unit Header#############################################
def UnitLayerDetectorHeader(net, data_layer="data", num_classes=2, feature_layer="conv5", \
        normalization=-1, use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], flip=True, clip=True, inter_layer_channels=[], \
        flat=False, use_focus_loss=False, stage=1,lr_mult=1.0,decay_mult=1.0,flag_withparamname=False,add_str = ""):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers."
    print feature_layer
    assert feature_layer + add_str in net_layers, "feature_layer is not in net's layers.(%s)"%feature_layer
    assert pro_widths, "Must provide proposed width/height. "
    assert pro_heights, "Must provide proposed width/height."
    assert len(pro_widths) == len(pro_heights), "pro_widths/heights must have the same length."
    from_layer = feature_layer
    prefix_name = '{}_{}'.format(from_layer,stage)
    from_layer += add_str
    # Norm-Layer
    if normalization != -1:
        norm_name = "{}_{}_norm".format(prefix_name,stage)
        net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalization), \
            across_spatial=False, channel_shared=False)
        from_layer = norm_name
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
                num_output=inter_channel, kernel_size=inter_kernel, pad=inter_pad, stride=1,use_scale=True, leaky=False,
                            lr_mult=lr_mult, decay_mult=decay_mult,flag_withparamname=flag_withparamname,pose_string=add_str)
            from_layer = inter_name + add_str
            start_inter_id = start_inter_id + 1
    # Estimate number of priors per location given provided parameters.
    num_priors_per_location = len(pro_widths)
    # Create location prediction layer.
    name = "{}_mbox_loc".format(prefix_name)
    num_loc_output = num_priors_per_location * 4
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_loc_output, kernel_size=3, pad=1, stride=1,lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    permute_name = "{}_perm".format(name) + add_str
    net[permute_name] = L.Permute(net[name + add_str], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name) + add_str
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layer = net[flatten_name]
    else:
        loc_layer = net[permute_name]
    # Create confidence prediction layer.
    name = "{}_mbox_conf".format(prefix_name)
    num_conf_output = num_priors_per_location * num_classes
    if use_focus_loss:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,init_xavier=False,bias_type='focal',sparse=num_classes,
                        lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    else:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    permute_name = "{}_perm".format(name) + add_str
    net[permute_name] = L.Permute(net[name + add_str], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name) + add_str
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layer = net[flatten_name]
    else:
        conf_layer = net[permute_name]
    # Create prior generation layer.
    name = "{}_mbox_priorbox".format(prefix_name) + add_str
    net[name] = L.PriorBox(net[from_layer], net[data_layer], pro_width=pro_widths, pro_height=pro_heights, \
        flip=flip, clip=clip, variance=prior_variance)
    priorbox_layer = net[name]
    return loc_layer,conf_layer,priorbox_layer
####################################################################################################
#####################################Create Multi Headers###########################################
def MultiLayersDetectorHeader(net, data_layer="data", num_classes=2, from_layers=[], \
        normalizations=[], use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], flip=True, clip=True, inter_layer_channels=[], \
        use_focus_loss=False, stage=1,lr_mult=1.0,decay_mult=1.0,flag_withparamname=False,add_str = ""):
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
        loc_layer,conf_layer,priorbox_layer = \
            UnitLayerDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                feature_layer=from_layer, normalization=normalization, use_batchnorm=use_batchnorm, \
                prior_variance = prior_variance, pro_widths=prowidths, pro_heights=proheights, \
                flip=flip, clip=clip, inter_layer_channels=inter_layer_depth, \
                flat=True, use_focus_loss=use_focus_loss, stage=stage,lr_mult=lr_mult,decay_mult=decay_mult,flag_withparamname=flag_withparamname,add_str=add_str)
        loc_layers.append(loc_layer)
        conf_layers.append(conf_layer)
        priorbox_layers.append(priorbox_layer)
    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_{}_loc".format(stage) + add_str
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_conf".format(stage) + add_str
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_priorbox".format(stage) + add_str
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    return mbox_layers
####################################################################################################
#####################################Create Unit DenseBoxHeader#####################################
def UnitLayerDenseDetectorHeader(net, data_layer="data", num_classes=2, feature_layer="conv5", \
        normalization=-1, use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], flip=True, clip=True, \
        inter_layer_channels=0, flat=False, use_focus_loss=False, stage=1,lr_mult=1.0,decay_mult=1.0,flag_withparamname=False,add_str = ""):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers."
    assert feature_layer + add_str in net_layers, "feature_layer is not in net's layers.(%s)"%(feature_layer + add_str )
    assert pro_widths, "Must provide proposed width/height."
    assert pro_heights, "Must provide proposed width/height."
    assert len(pro_widths) == len(pro_heights), "pro_widths/heights must have the same length."
    from_layer = feature_layer
    prefix_name = '{}_{}'.format(from_layer,stage)
    from_layer += add_str
    # Norm-Layer
    if normalization != -1:
        norm_name = "{}_norm".format(prefix_name)
        net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalization), \
            across_spatial=False, channel_shared=False)
        from_layer = norm_name
    # InterLayers
    if not inter_layer_channels==0:
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
                ConvBNUnitLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True,
                    num_output=inter_channel, kernel_size=inter_kernel, pad=inter_pad, stride=1,use_scale=True, leaky=False,
                                lr_mult=lr_mult, decay_mult=decay_mult,flag_withparamname=flag_withparamname,pose_string=add_str)
                from_layer = inter_name  + add_str
                start_inter_id = start_inter_id + 1
    # PriorBoxes
    num_priors_per_location = len(pro_widths)
    print "num_priors_per_location:",num_priors_per_location
    # LOC
    name = "{}_mbox_loc".format(prefix_name)
    num_loc_output = num_priors_per_location * 4 * (num_classes-1)
    ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
        num_output=num_loc_output, kernel_size=3, pad=1, stride=1,lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    permute_name = "{}_perm".format(name) + add_str
    net[permute_name] = L.Permute(net[name + add_str], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name) + add_str
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layer = net[flatten_name]
    else:
        loc_layer = net[permute_name]
    # CONF
    name = "{}_mbox_conf".format(prefix_name) + add_str
    num_conf_output = num_priors_per_location * num_classes
    if use_focus_loss:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,init_xavier=False,bias_type='focal',sparse=num_classes,
                        lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    else:
        ConvBNUnitLayer(net, from_layer, name, use_bn=False, use_relu=False, \
            num_output=num_conf_output, kernel_size=3, pad=1, stride=1,lr_mult=lr_mult, decay_mult=decay_mult,pose_string=add_str)
    permute_name = "{}_perm".format(name) + add_str
    net[permute_name] = L.Permute(net[name + add_str], order=[0, 2, 3, 1])
    if flat:
        flatten_name = "{}_flat".format(name) + add_str
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layer = net[flatten_name]
    else:
        conf_layer = net[permute_name]
    # PRIOR
    name = "{}_mbox_priorbox".format(prefix_name) + add_str
    net[name] = L.PriorBox(net[from_layer], net[data_layer], pro_width=pro_widths, pro_height=pro_heights, \
        flip=flip, clip=clip, variance=prior_variance)
    priorbox_layer = net[name]
    return loc_layer,conf_layer,priorbox_layer
####################################################################################################
#####################################Create Multi DenseHeaders######################################
def MultiLayersDenseDetectorHeader(net, data_layer="data", num_classes=2, from_layers=[], \
        normalizations=[], use_batchnorm=True, prior_variance = [0.1], \
        pro_widths=[], pro_heights=[], flip=True, clip=True, \
        inter_layer_channels=[], use_focus_loss=False, stage=1,lr_mult=1.0,decay_mult=1.0,flag_withparamname=False,add_str=""):
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
        loc_layer,conf_layer,priorbox_layer = \
            UnitLayerDenseDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                feature_layer=from_layer, normalization=normalization, use_batchnorm=use_batchnorm, \
                prior_variance = prior_variance, pro_widths=prowidths, pro_heights=proheights, \
                flip=flip, clip=clip, inter_layer_channels=inter_layer_depth, \
                flat=True, use_focus_loss=use_focus_loss, stage=stage,lr_mult=lr_mult,decay_mult=decay_mult,flag_withparamname=flag_withparamname,add_str=add_str)

        loc_layers.append(loc_layer)
        conf_layers.append(conf_layer)
        priorbox_layers.append(priorbox_layer)
    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_{}_loc".format(stage) + add_str
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_conf".format(stage) + add_str
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_{}_priorbox".format(stage) + add_str
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    return mbox_layers
####################################################################################################
#####################################Create SSD Headers ############################################
def SsdDetectorHeaders(net, net_width=300, net_height=300, data_layer="data", \
            num_classes=2, from_layers=[], use_batchnorm=True, boxsizes=[], \
            prior_variance = [0.1,0.1,0.2,0.2], normalizations=[], aspect_ratios=[], \
            flip=True, clip=True, inter_layer_channels=[], use_focus_loss=False, \
            use_dense_boxes=False, stage=1,lr_mult=1.0,decay_mult=1.0,flag_withparamname=False,add_str = "", AnchorFixed = False):
    assert from_layers, "Feature layers must be provided."
    pro_widths=[]
    pro_heights=[]
    for i in range(len(boxsizes)):
      boxsizes_per_layer = boxsizes[i]
      pro_widths_per_layer = []
      pro_heights_per_layer = []
      for j in range(len(boxsizes_per_layer)):
        boxsize = boxsizes_per_layer[j]
        # aspect_ratio = aspect_ratios[0]
        # if not len(aspect_ratios) == 1:
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
    print "num_classes:",num_classes
    if AnchorFixed:
        pro_widths=[]
        pro_heights=[]
        w_0 = [0.06,0.03,0.0424264068712,0.12,0.06,0.0848528137424]
        w_1 = [0.18,0.09,0.127279220614,0.24,0.12,0.169705627485,0.32,0.16,0.22627416998]
        w_2 = [0.4,0.2,0.282842712475,0.6,0.3,0.424264068712,0.8,0.565685424949,0.95]
        h_0 = [0.06,0.12,0.0848528137424,0.12,0.24,0.169705627485]
        h_1 = [0.18,0.36,0.254558441227,0.24,0.48,0.33941125497,0.32,0.64,0.452548339959]
        h_2 = [0.4,0.8,0.565685424949,0.6,1.0,0.848528137424,0.8,1.0,0.95]
        for i in xrange(3):
            pro_widths.append(eval('w_%d'%i))
            pro_heights.append(eval('h_%d'%i))
    if use_dense_boxes:
        mbox_layers = MultiLayersDenseDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
            from_layers=from_layers, normalizations=normalizations, use_batchnorm=use_batchnorm, \
            prior_variance = prior_variance, pro_widths=pro_widths, pro_heights=pro_heights, \
            flip=flip, clip=clip, inter_layer_channels=inter_layer_channels, \
            use_focus_loss=use_focus_loss, stage=stage,lr_mult=lr_mult,decay_mult=decay_mult,flag_withparamname=flag_withparamname,add_str=add_str)
    else:
        mbox_layers = MultiLayersDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
            from_layers=from_layers, normalizations=normalizations, use_batchnorm=use_batchnorm, \
            prior_variance = prior_variance, pro_widths=pro_widths, pro_heights=pro_heights, \
            flip=flip, clip=clip, inter_layer_channels=inter_layer_channels, \
            use_focus_loss=use_focus_loss, stage=stage,lr_mult=lr_mult,decay_mult=decay_mult,flag_withparamname=flag_withparamname,add_str=add_str)
    return mbox_layers
