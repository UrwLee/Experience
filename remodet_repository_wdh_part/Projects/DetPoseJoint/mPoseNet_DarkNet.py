import caffe

from caffe import layers as L
from caffe import params as P
from mPoseNet_Reduce import mPose_StageX_Train
from mPoseBaseNet import *
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.ConvBNLayer import *

############################################################
# Compress the YoloNet.
# Remove all the pool layers.
# Change the stride of the original conv layers before pool layers to 2.
############################################################
def  YoloNetPartBigStride(net, from_layer="data", use_bn=True, use_sub_layers=(1,1,3,3,5), leaky=False,
						 num_channels = (32,64,128,256,512),kernel_sizes = (7,3,3,3,3),strides = (4,2,2,1,1),deconv_channels = [256,128],
                          strides_last_flags = (True,True,True,True,True),pooling_flags = (False,)*5,ChangeNameAndChannel = {},lr=1, decay=1, addstrs = ""):
    assert len(use_sub_layers) == len(num_channels) == len(kernel_sizes) == len(strides) == len(strides_last_flags)
    for layer in xrange(len(num_channels)):
        for sub_layer in xrange(use_sub_layers[layer]):
            if use_sub_layers[layer] == 1:
                out_layer = "conv{}".format(layer + 1)
            else:
                out_layer = "conv{}_{}".format(layer + 1, sub_layer + 1)

            if strides_last_flags[layer]:
                if sub_layer == use_sub_layers[layer] - 1:
                    stride_i = strides[layer]
                else:
                    stride_i = 1
            else:
                if sub_layer == 0:
                    stride_i = strides[layer]
                else:
                    stride_i = 1
            if sub_layer%2 == 0:
                kernel_size_i = kernel_sizes[layer]
                num_channel_i = num_channels[layer]
            else:
                kernel_size_i = 1
                num_channel_i = num_channels[layer]/2
            if out_layer in ChangeNameAndChannel.keys():
                if ChangeNameAndChannel[out_layer] != 0:
                    num_channel_i = ChangeNameAndChannel[out_layer]
                out_layer += "_new"
            out_layer += addstrs
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=num_channel_i,
                            kernel_size=kernel_size_i, pad=(kernel_size_i-1)/2, stride=stride_i,
                            use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)
            if pooling_flags[layer]:
                out_layer = 'pool{}'.format(layer+1)
                net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                                           kernel_size=2, stride=2, pad=0)
                from_layer = out_layer
            else:
                from_layer = out_layer
    deconv_param = {
        'num_output': deconv_channels[0],
        'kernel_size': 2,
        'pad': 0,
        'stride': 2,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': deconv_param
    }
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
    }
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
        'filler': dict(type='constant', value=1.0),
        'bias_filler': dict(type='constant', value=0.2),
    }
    from_layer = "conv4_{}".format(use_sub_layers[-2]) + addstrs
    add_layer = from_layer + "_deconv"
    print from_layer, add_layer
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    bn_name = add_layer + '_bn'
    net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
    sb_name = add_layer + '_scale'
    net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
    relu_name = add_layer + '_relu'
    net[relu_name] = L.ReLU(net[add_layer], in_place=True)

    deconv_param1 = {
        'num_output': deconv_channels[1],
        'kernel_size': 4,
        'pad': 0,
        'stride': 4,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_deconv1 = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': deconv_param1
    }
    from_layer = "conv5_{}".format(use_sub_layers[-1]) + addstrs
    add_layer = from_layer + "_deconv"
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv1)
    bn_name = add_layer + '_bn'
    net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
    sb_name = add_layer + '_scale'
    net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
    relu_name = add_layer + '_relu'
    net[relu_name] = L.ReLU(net[add_layer], in_place=True)

    return net

def  Flexible_Base(net, from_layer="data", use_bn=True, leaky=False,
				   num_channels = ((32,),(64,),(64,32,256),(256,64,256,64,256,128,128)),
                   kernel_sizes = ((7,),(3,),(3,3,3),(3,1,3,1,3,1,3)),
                   strides = ((4,),(2,),(1,1,1),(2,1,1,1,1,1,1)),
                    lr=1, decay=1, flag_deconv=True,flag_deconv_relu=False,num_channel_deconv=128,scale_deconv=2,special_layers = "",addstrs = ""):
    assert len(num_channels) == len(kernel_sizes) == len(strides)
    for layer in xrange(len(num_channels)):
        assert len(num_channels[layer]) == len(kernel_sizes[layer]) == len(strides[layer])
        num_sublayer = len(num_channels[layer])
        for sub_layer in xrange(num_sublayer):
            if num_channels[layer] == 1:
                out_layer = "conv{}".format(layer + 1)
            else:
                out_layer = "conv{}_{}".format(layer + 1, sub_layer + 1)
            if out_layer == special_layers:
                flag_bninplace = False
            else:
                flag_bninplace = True
            out_layer += addstrs
            num_output = num_channels[layer][sub_layer]
            kernel_size = kernel_sizes[layer][sub_layer]
            stride = strides[layer][sub_layer]
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=use_bn, use_relu=True,num_output=num_output,
                            kernel_size=kernel_size, pad=(kernel_size-1)/2, stride=stride,
                            use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay,flag_bninplace=flag_bninplace)
            if flag_bninplace:
                from_layer = out_layer
            else:
                from_layer = out_layer + "_bn"
    if flag_deconv:
        deconv_param = {
            'num_output': num_channel_deconv,
            'kernel_size': scale_deconv,
            'pad': 0,
            'stride': scale_deconv,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),
            'group': 1,
        }
        kwargs_deconv = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'convolution_param': deconv_param
        }
        bn_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'eps': 0.001,
        }
        sb_kwargs = {
            'bias_term': True,
            'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
            'filler': dict(type='constant', value=1.0),
            'bias_filler': dict(type='constant', value=0.2),
        }
        from_layer = "conv{}_{}".format(len(num_channels), len(num_channels[-1])) + addstrs
        add_layer = from_layer + "_deconv"
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
        if flag_deconv_relu:
            bn_name = add_layer + '_bn'
            net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
            sb_name = add_layer + '_scale'
            net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
            relu_name = add_layer + '_relu'
            net[relu_name] = L.ReLU(net[add_layer], in_place=True)
    return net
def mPose_StageXDeconv_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True,base_layer="convf", lr=1.0, decay=1.0,
                       num_channels = 128, kernel_size=3,flag_hasloss_befdecov=True,flag_reducefirst = False,up_scale=4,addstrs = ''):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    cnt = -1
    if flag_reducefirst:
        conv_vec = "stage{}_conv{}_vec".format(stage, 0) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage, 0)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage, 0) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0,kernel_size=1, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage, 0)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat

    for layer in range(1, numlayers):
        # vec

        conv_vec = "stage{}_conv{}_vec".format(stage,layer) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    if use_1_layers > 0:
        for layer in range(1, use_1_layers):
            # vec
            conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+layer) + addstrs
            net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_vec = "stage{}_relu{}_vec".format(stage,use_3_layers+layer) + addstrs
            net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
            from1_layer = relu_vec
            # heat
            conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+layer) + addstrs
            net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0, kernel_size=1, **kwargs)
            relu_heat = "stage{}_relu{}_heat".format(stage,use_3_layers+layer) + addstrs
            net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
            from2_layer = relu_heat
        # output
        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers+use_1_layers) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=0, kernel_size=1, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers+use_1_layers) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=0, kernel_size=1, **kwargs)
    else:
        # output by 3x3

        conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)

    if flag_hasloss_befdecov:
        mask_vec_dn = mask_vec + "_DN"
        label_vec_dn = label_vec  + "_DN"
        mask_heat_dn = mask_heat + "_DN"
        label_heat_dn = label_heat + "_DN"
        weight_vec = "weight_stage{}_vec".format(stage)
        weight_heat = "weight_stage{}_heat".format(stage)
        loss_vec = "loss_stage{}_vec".format(stage)
        loss_heat = "loss_stage{}_heat".format(stage)
        net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec_dn], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec_dn], loss_weight=1)
        net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat_dn], eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat_dn], loss_weight=1)

    conv_param_vec = {
        'num_output': 34,
        'kernel_size': up_scale,
        'pad': 0,
        'stride': up_scale,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_vec = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param_vec
    }
    conv_param_heat = {
        'num_output': 18,
        'kernel_size': up_scale,
        'pad': 0,
        'stride': up_scale,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_heat = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param_heat
    }

    deconv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers) + '_deconv' + addstrs
    net[deconv_vec] = L.Deconvolution(net[conv_vec], **kwargs_vec)
    deconv_heat = "stage{}_conv{}_heat".format(stage, use_3_layers) + '_deconv' + addstrs
    net[deconv_heat] = L.Deconvolution(net[conv_heat], **kwargs_heat)
    weight_vec = "weight_stage{}_vec".format(stage) + '_deconv'
    weight_heat = "weight_stage{}_heat".format(stage) + '_deconv'
    loss_vec = "loss_stage{}_vec".format(stage) + '_deconv'
    loss_heat = "loss_stage{}_heat".format(stage) + '_deconv'
    net[weight_vec] = L.Eltwise(net[deconv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[deconv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)

    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net

def mPose_StageXDeconv_A_Train(net, from_layer="concat_stage1", out_layer="concat_stage2", stage=1, \
                       mask_vec="vec_mask", mask_heat="heat_mask",label_vec="vec_label", label_heat="heat_label", \
                       use_3_layers=5, use_1_layers=2, short_cut=True,base_layer="convf", lr=1.0, decay=1.0,
                       num_channels = 128, kernel_size=3,flag_reducefirst = False,scale_updown=1,addstrs = ''):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2*lr, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    from1_layer = from_layer
    from2_layer = from_layer
    if use_1_layers > 0:
        numlayers = use_3_layers + 1
    else:
        numlayers = use_3_layers
    cnt = -1
    if flag_reducefirst:
        conv_vec = "stage{}_conv{}_vec".format(stage, 0) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=0,kernel_size=1, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage, 0)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage, 0) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=0,kernel_size=1, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage, 0)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat

    for layer in range(1, numlayers):
        # vec

        conv_vec = "stage{}_conv{}_vec".format(stage,layer) + addstrs
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_vec = "stage{}_relu{}_vec".format(stage,layer)
        net[relu_vec] = L.ReLU(net[conv_vec], in_place=True)
        from1_layer = relu_vec
        # heat
        conv_heat = "stage{}_conv{}_heat".format(stage,layer) + addstrs
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=num_channels, pad=(kernel_size-1)/2, kernel_size=kernel_size, **kwargs)
        relu_heat = "stage{}_relu{}_heat".format(stage,layer)
        net[relu_heat] = L.ReLU(net[conv_heat], in_place=True)
        from2_layer = relu_heat
    conv_param_vec = {
        'num_output': 34,
        'kernel_size': scale_updown,
        'pad': 0,
        'stride': scale_updown,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_vec = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param_vec
    }
    conv_param_heat = {
        'num_output': 18,
        'kernel_size': scale_updown,
        'pad': 0,
        'stride': scale_updown,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_heat = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param_heat
    }
    conv_vec = "stage{}_conv{}_vec".format(stage,use_3_layers) + '_deconv' + addstrs
    net[conv_vec] = L.Deconvolution(net[from1_layer], **kwargs_vec)
    conv_heat = "stage{}_conv{}_heat".format(stage,use_3_layers) +  '_deconv' + addstrs
    net[conv_heat] = L.Deconvolution(net[from2_layer], **kwargs_heat)


    weight_vec = "weight_stage{}_vec".format(stage) + '_deconv'
    weight_heat = "weight_stage{}_heat".format(stage) + '_deconv'
    loss_vec = "loss_stage{}_vec".format(stage) + '_deconv'
    loss_heat = "loss_stage{}_heat".format(stage) + '_deconv'
    net[weight_vec] = L.Eltwise(net[conv_vec], net[mask_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_vec] = L.EuclideanLoss(net[weight_vec], net[label_vec], loss_weight=1)
    net[weight_heat] = L.Eltwise(net[conv_heat], net[mask_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
    net[loss_heat] = L.EuclideanLoss(net[weight_heat], net[label_heat], loss_weight=1)

    if short_cut:
        fea_layers = []
        fea_layers.append(net[conv_vec])
        fea_layers.append(net[conv_heat])
        add_layer = "stage{}_concatvecandheat".format(stage)
        net[add_layer] = L.Concat(*fea_layers, axis=1)
        from_layer = add_layer
        add_layer = add_layer + "_pool"
        net[add_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                                   kernel_size=scale_updown, stride=scale_updown, pad=0)
        fea_layers = []
        fea_layers.append(net[add_layer])
        assert base_layer in net.keys()
        fea_layers.append(net[base_layer])
        net[out_layer] = L.Concat(*fea_layers, axis=1)
    return net
def YoloNetPart_StrideRemove1x1(net, num_sublayers, num_channels  ,from_layer="data", lr=1, decay=1,alpha=1,fix_layer=-1,fix_sublayer = -1):
    leaky = False
    num_channels_fourlayers = [32,64,128,256]
    flag_lr_zero = True
    for layer in xrange(len(num_channels_fourlayers)):
        for sublayer in xrange(num_sublayers[layer]):
            if num_sublayers[layer] == 1:
                out_layer = "conv{}".format(layer + 1)
            else:
                out_layer = "conv{}_{}".format(layer + 1, sublayer + 1)
            if sublayer == num_sublayers[layer] - 1 and layer != len(num_channels_fourlayers) - 1:
                stride = 2
            else:
                stride = 1
            if layer != 0:
                scale = alpha
            else:
                scale = 1
            if fix_layer != -1:
                if layer < fix_layer - 1:
                    flag_lr_zero = True
                else:
                    flag_lr_zero = False
            else:
                flag_lr_zero = False

            if flag_lr_zero:
                ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                                num_output=(num_channels_fourlayers[layer] / scale), kernel_size=3, pad=1,
                                stride=stride, use_scale=True, leaky=leaky, lr_mult=0, decay_mult=0)
            else:
                ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                                num_output=(num_channels_fourlayers[layer]/scale), kernel_size=3, pad=1, stride=stride,
                                use_scale=True, leaky=leaky, lr_mult=lr, decay_mult=decay)
            from_layer = out_layer

    out_layer = 'pool4'
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                               kernel_size=2, stride=2, pad=0)


    for sublayer in xrange(len(num_channels)):
        if sublayer != len(num_channels) - 1:
            scale = alpha
        else:
            scale = 1
        from_layer = out_layer
        out_layer = "conv5_{}".format(sublayer + 1)
        if fix_layer != -1:
            if sublayer < fix_sublayer:
                flag_lr_zero = True
            else:
                flag_lr_zero = False
        else:
            flag_lr_zero = False
        if flag_lr_zero:
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                            num_output=(num_channels[sublayer] / scale), kernel_size=3, pad=1, stride=1,
                            use_scale=True,leaky=leaky, lr_mult=0, decay_mult=0)
        else:
            ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                            num_output=(num_channels[sublayer]/scale), kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky, lr_mult=lr,
                            decay_mult=decay)

    return net

def YoloNetPart_StrideRemove1x1AndPooling(net, from_layer="data", lr=1, decay=1):
    leaky = False
    out_layer = 'conv1'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, \
        num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=leaky,lr_mult=lr,decay_mult=decay)

    from_layer = out_layer
    out_layer = 'conv2'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=64, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=leaky, lr_mult=lr,
                    decay_mult=decay)

    from_layer = out_layer
    out_layer = 'conv3_1'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=128, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=leaky, lr_mult=lr,
                    decay_mult=decay)

    from_layer = out_layer
    out_layer = 'conv3_2'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=128, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky, lr_mult=lr,
                    decay_mult=decay)

    from_layer = out_layer
    out_layer = 'conv4_1'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky, lr_mult=lr,
                    decay_mult=decay)

    from_layer = out_layer
    out_layer = 'conv4_2'
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=256, kernel_size=3, pad=1, stride=1, use_scale=True, leaky=leaky, lr_mult=lr,
                    decay_mult=decay)

    num_channels = [512,256,512,256,512]
    for sublayer in xrange(len(num_channels)):
        from_layer = out_layer
        out_layer = "conv5_{}".format(sublayer + 1)
        if sublayer == 0:
            stride = 2
        else:
            stride = 1
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                        num_output=num_channels[sublayer], kernel_size=3, pad=1, stride=stride, use_scale=True, leaky=leaky, lr_mult=lr,
                        decay_mult=decay)

    return net
def mPoseNet_COCO_DarkNetMultiStage_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19

    net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False, lr=1, decay=1)

    baselayer = "convf"
    net = UnifiedMultiScaleLayers(net, layers=["conv4_3","conv5_5"], tags=["Ref","Up"],unifiedlayer=baselayer, upsampleMethod="Reorg")
    # Stages

    net['convf_drop'] = L.SpatialDropout(net[baselayer], in_place=True,dropout_param=dict(dropout_ratio=0.2))
    baselayer = 'convf_drop'
    use_stage = 6
    use_3_layers = 7
    use_1_layers = 0
    n_channel = 128
    lrdecay = 1.0
    kernel_size = 7
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size)
    # for Test
    if not train:
        print(net.keys())
        conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers)
        conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetMultiStageB_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_hasdrop = False
    leaky = False
    net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False, leaky=leaky,lr=1, decay=1)

    add_layer = "conv5_5_upsample"
    conv_param = {"kernel_size":2,"stride":2,"num_output":128,"group":1,"pad":0,
                  "weight_filler":dict(type="bilinear"),"bias_term":False}
    net[add_layer] = L.Deconvolution(net["conv5_5"],convolution_param=conv_param,param=dict(lr_mult=0, decay_mult=0))

    fea_layers = []
    fea_layers.append(net["conv4_3"])
    fea_layers.append(net[add_layer])
    add_layer = "multiscale_concat"
    net[add_layer] = L.Concat(*fea_layers, axis=1)
    # Stages
    lrdecay = 1.0
    baselayer = "convf"
    ConvBNUnitLayer(net, add_layer, baselayer, use_bn=True, use_relu=True,num_output=128, kernel_size=3, pad=1, stride=1,
                    use_scale=True, leaky=leaky,lr_mult = lrdecay, decay_mult = lrdecay)
    if flag_hasdrop:
        net['convf_drop'] = L.SpatialDropout(net[baselayer], in_place=True, dropout_param=dict(dropout_ratio=0.2))
    use_stage = 6
    use_3_layers = 7
    use_1_layers = 0
    n_channel = 128

    kernel_size = 7
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        if stage == use_stage - 1:
            flag_change_layer = True
        else:
            flag_change_layer = False
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,
                                 flag_change_layer=flag_change_layer)
    # for Test
    if not train:
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64 * 64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
        }
        for stage in xrange(use_stage):


            conv_vec = "stage{}_conv{}_vec".format(stage + 1, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(stage + 1, use_3_layers + use_1_layers)
            net["vec_out{}".format(stage + 1)] = L.Eltwise(net.vec_mask, net[conv_vec],
                                                           eltwise_param=dict(operation=P.Eltwise.PROD))
            net["heat_out{}".format(stage + 1)] = L.Eltwise(net.heat_mask, net[conv_heat],
                                                            eltwise_param=dict(operation=P.Eltwise.PROD))
            feaLayers = []
            feaLayers.append(net["heat_out{}".format(stage + 1)])
            feaLayers.append(net["vec_out{}".format(stage + 1)])
            outlayer = "concateval_stage{}".format(stage + 1)
            net[outlayer] = L.Concat(*feaLayers, axis=1)
            net["resized_map{}".format(stage + 1)] = L.ImResize(net[outlayer], name="resize",
                                                                imresize_param=resize_kwargs)
            net["joints{}".format(stage + 1)] = L.Nms(net["resized_map{}".format(stage + 1)], name="nms",
                                                      nms_param=nms_kwargs)
            net["limbs{}".format(stage + 1)] = L.Connectlimb(net["resized_map{}".format(stage + 1)],
                                                             net["joints{}".format(stage + 1)],
                                                             connect_limb_param=connect_kwargs)
            net["eval{}".format(stage + 1)] = L.PoseEval(net["limbs{}".format(stage + 1)], net.gt,
                                                         pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetMultiStageChangeBase_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_concat_lowlevel = True
    flag_output_sigmoid  = False
    num_sublayers = [1, 1, 2, 3] #number of sublayers in conv1 to conv4
    num_channels = [512, 128, 512, 256, 512] #number of channels in conv5
    alpha = 1
    leaky = False
    num_chan_deconv = 128
    ChangeNameAndChannel = {"conv4_3":128,"conv5_1":512}
    # ChangeNameAndChannel = {}
    net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False,
                      leaky=leaky, lr=1, decay=1,ChangeNameAndChannel=ChangeNameAndChannel)

    # net = YoloNetPart_StrideRemove1x1(net, num_sublayers=num_sublayers,num_channels=num_channels,from_layer=data_layer,
    #                                   lr=1, decay=1,alpha=alpha,fix_layer=5,fix_sublayer=1)

    conv_param = {
        'num_output': num_chan_deconv,
        'kernel_size': 2,
        'pad': 0,
        'stride': 2,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        'group': 1,
    }
    # conv_param = {"kernel_size": 4, "stride": 2, "num_output": 128, "group": 128, "pad": 1,
    #               "weight_filler": dict(type="bilinear"), "bias_term": False}
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param
    }

    from_layer = "conv5_{}".format(len(num_channels))
    out_layer = from_layer + "_Upsample"
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)

    if flag_concat_lowlevel:
        baselayer = "convf"
        feature_layers = []
        if "conv4_3" in ChangeNameAndChannel.keys():
            feature_layers.append(net["conv4_3_new"])
        else:
            feature_layers.append(net["conv4_3"])
        feature_layers.append(net[out_layer])
        net[baselayer] = L.Concat(*feature_layers, axis=1)
    else:
        baselayer = out_layer
    # Stages
    use_stage = 3
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,flag_sigmoid=flag_output_sigmoid)
    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetSmallFeatMap_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input

    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_use_deconv = True
    flag_concat_lowlevel = False
    flag_has_other_loss = False
    strid_convs = [1, 1, 1, 0, 0]
    kernel_size_first = 7
    stride_first = 4
    channel_divides = (1, 1, 1, 1, 1)
    num_channel_conv5_5 = 512
    net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,leaky=False,
                              strid_conv=strid_convs, final_pool=False, lr=1, decay=1,kernel_size_first=kernel_size_first,
                              stride_first = stride_first,channel_divides = channel_divides,num_channel_conv5_5=num_channel_conv5_5)
    # if flag_use_deconv:
    #     down_sample_layers = ["vec_label", "heat_label","vec_mask","heat_mask"]
    #     for layer_name in down_sample_layers:
    #         out_layer = layer_name + "_DN"
    #         net[out_layer] = L.Pooling(net[layer_name], pool=P.Pooling.MAX,
    #                                    kernel_size=4, stride=4, pad=0)
    if flag_concat_lowlevel:
        from_layer = "conv4_3"
        out_layer = from_layer + "_DN"
        net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE,
                                   kernel_size=2, stride=2, pad=0)
        baselayer = "convf"
        feature_layers = []
        feature_layers.append(net[out_layer])
        feature_layers.append(net["conv5_5"])
        net[baselayer] = L.Concat(*feature_layers, axis=1)
    else:
        baselayer = "conv5_5"
    # Stages
    flag_hasloss_befdecov = False
    flag_reducefirst = True

    use_stage = 3
    use_3_layers = [5,5,5]
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        # net = mPose_StageXDeconv_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
        #                    mask_vec="vec_mask", mask_heat="heat_mask", \
        #                    label_vec="vec_label", label_heat="heat_label", \
        #                    use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
        #                    base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,
        #                    flag_hasloss_befdecov=flag_hasloss_befdecov,flag_reducefirst=flag_reducefirst)
        net = mPose_StageXDeconv_A_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1, \
                                       mask_vec="vec_mask", mask_heat="heat_mask", \
                                       label_vec="vec_label", label_heat="heat_label", \
                                       use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
                                       base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
                                       kernel_size=kernel_size,flag_reducefirst=flag_reducefirst)
    if train:
        if flag_has_other_loss:
            kwargs = {'param': [dict(lr_mult=lrdecay, decay_mult=lrdecay), dict(lr_mult=2 * lrdecay, decay_mult=0)],
                      'weight_filler': dict(type='gaussian', std=0.01),
                      'bias_filler': dict(type='constant', value=0)}
            if not flag_concat_lowlevel:
                from_layer = "conv4_3"
                out_layer = from_layer + "_DN"
                net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE,
                                           kernel_size=2, stride=2, pad=0)
            add_layers = ["conv4",]
            from_layers = ["conv4_3_DN",]
            kernel_size = 23
            for i in xrange(len(add_layers)):
                conv_vec = "%s_vec"%add_layers[i]

                net[conv_vec] = L.Convolution(net[from_layers[i]], num_output=34, pad=(kernel_size - 1) / 2, kernel_size=kernel_size,
                                              **kwargs)
                conv_heat = "%s_heat"%add_layers[i]
                net[conv_heat] = L.Convolution(net[from_layers[i]], num_output=18, pad=(kernel_size - 1) / 2, kernel_size=kernel_size,
                                               **kwargs)
                weight_vec = "weight_%s_vec"%add_layers[i]
                weight_heat = "weight_%s_heat"%add_layers[i]
                loss_vec = "loss_%s_vec"%add_layers[i]
                loss_heat = "loss_%s_heat"%add_layers[i]
                net[weight_vec] = L.Eltwise(net[conv_vec], net.vec_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
                net[loss_vec] = L.EuclideanLoss(net[weight_vec], net.vec_label, loss_weight=0.3)
                net[weight_heat] = L.Eltwise(net[conv_heat], net.heat_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
                net[loss_heat] = L.EuclideanLoss(net[weight_heat], net.heat_label, loss_weight=0.5)


    # for Test
    if not train:
        if flag_use_deconv:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers)

        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetSmallFeatMap_A_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input

    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_use_deconv = True
    flag_concat_lowlevel = False
    flag_has_other_loss = False
    strid_convs = [1, 1, 1, 0, 0]
    kernel_size_first = 7
    stride_first = 4
    channel_divides = (1, 1, 1, 1, 1)
    num_channel_conv5_5 = 128
    net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,leaky=False,
                              strid_conv=strid_convs, final_pool=False, lr=1, decay=1,kernel_size_first=kernel_size_first,
                              stride_first = stride_first,channel_divides = channel_divides,num_channel_conv5_5=num_channel_conv5_5)
    deconv_param = {
        'num_output': 128,
        'kernel_size': 2,
        'pad': 0,
        'stride': 2,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'group': 1,
    }
    kwargs_deconv = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': deconv_param
    }
    from_layer = "conv5_5"
    add_layer = from_layer + "_deconv"
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)

    if flag_concat_lowlevel:
        from_layer = "conv4_3"
        baselayer = "convf"
        feature_layers = []
        feature_layers.append(net[from_layer])
        feature_layers.append(net[add_layer])
        net[baselayer] = L.Concat(*feature_layers, axis=1)
    else:
        baselayer = add_layer

    up_scale = 2
    if flag_use_deconv:
        down_sample_layers = ["vec_label", "heat_label","vec_mask","heat_mask"]
        for layer_name in down_sample_layers:
            out_layer = layer_name + "_DN"
            net[out_layer] = L.Pooling(net[layer_name], pool=P.Pooling.MAX,
                                       kernel_size=up_scale, stride=up_scale, pad=0)
    # Stages
    flag_hasloss_befdecov = True
    flag_reducefirst = True

    use_stage = 3
    use_3_layers = [5,5,5]
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3

    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageXDeconv_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,
                           flag_hasloss_befdecov=flag_hasloss_befdecov,flag_reducefirst=flag_reducefirst,up_scale=up_scale)

    if train:
        if flag_has_other_loss:
            kwargs = {'param': [dict(lr_mult=lrdecay, decay_mult=lrdecay), dict(lr_mult=2 * lrdecay, decay_mult=0)],
                      'weight_filler': dict(type='gaussian', std=0.01),
                      'bias_filler': dict(type='constant', value=0)}
            if not flag_concat_lowlevel:
                from_layer = "conv4_3"
                out_layer = from_layer + "_DN"
                net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE,
                                           kernel_size=2, stride=2, pad=0)
            add_layers = ["conv4",]
            from_layers = ["conv4_3_DN",]
            kernel_size = 23
            for i in xrange(len(add_layers)):
                conv_vec = "%s_vec"%add_layers[i]

                net[conv_vec] = L.Convolution(net[from_layers[i]], num_output=34, pad=(kernel_size - 1) / 2, kernel_size=kernel_size,
                                              **kwargs)
                conv_heat = "%s_heat"%add_layers[i]
                net[conv_heat] = L.Convolution(net[from_layers[i]], num_output=18, pad=(kernel_size - 1) / 2, kernel_size=kernel_size,
                                               **kwargs)
                weight_vec = "weight_%s_vec"%add_layers[i]
                weight_heat = "weight_%s_heat"%add_layers[i]
                loss_vec = "loss_%s_vec"%add_layers[i]
                loss_heat = "loss_%s_heat"%add_layers[i]
                net[weight_vec] = L.Eltwise(net[conv_vec], net.vec_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
                net[loss_vec] = L.EuclideanLoss(net[weight_vec], net.vec_label, loss_weight=0.3)
                net[weight_heat] = L.Eltwise(net[conv_heat], net.heat_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
                net[loss_heat] = L.EuclideanLoss(net[weight_heat], net.heat_label, loss_weight=0.5)


    # for Test
    if not train:
        if flag_use_deconv:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers)

        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetSmallFeatMap_B_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input

    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_use_label_DN = False
    scale_updown = 2
    flag_concat_lowlevel = True
    flag_concat_conv3 = True # default false
    flag_upscale_base = True
    flag_base_reduce = False
    flag_concat_conv4_last = False# default False
    use_sub_layers = (1, 1, 3, 3, 5)
    num_channels = (32, 64, 128, 128, 256)
    kernel_sizes = (7, 3, 3, 3, 3)
    strides = (4,2,2,1,1)
    num_channel_final = 128
    strides_last_flags = (True,True,True,True,False)# default All True
    ChangeNameLayers = ["conv1",]
    for i in xrange(3):
        ChangeNameLayers.append("conv{}_{}".format(4,i+1))
    for i in xrange(5):
        ChangeNameLayers.append("conv{}_{}".format(5,i+1))
    net = YoloNetPartBigStride(net, from_layer=data_layer, use_bn=True, use_sub_layers=use_sub_layers, leaky=False,
						 num_channels = num_channels,kernel_sizes = kernel_sizes,strides = strides,
                               num_channel_final=num_channel_final,strides_last_flags=strides_last_flags,ChangeNameLayers=ChangeNameLayers,
                               lr=1, decay=1)

    if flag_use_label_DN:
        down_sample_layers = ["vec_label", "heat_label","vec_mask","heat_mask"]
        for layer_name in down_sample_layers:
            out_layer = layer_name + "_DN"
            net[out_layer] = L.Pooling(net[layer_name], pool=P.Pooling.MAX,
                                       kernel_size=scale_updown, stride=scale_updown, pad=0)
    if flag_concat_lowlevel:
        baselayer = "convf"
        feature_layers = []
        feature_layers.append(net["conv4_3_new"])
        feature_layers.append(net["conv5_5_new"])
        net[baselayer] = L.Concat(*feature_layers, axis=1)
    else:
        baselayer = "conv5_5_new"
    if flag_base_reduce:
        from_layer = baselayer
        out_layer = from_layer + "_reduce"
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, num_output=128,
                        kernel_size=1, pad=0, stride=1,
                        use_scale=True, leaky=False, lr_mult=1, decay_mult=1)
        baselayer = out_layer

    if flag_upscale_base:
        deconv_param = {
            'num_output': 128,
            'kernel_size': scale_updown,
            'pad': 0,
            'stride': scale_updown,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),
            'group': 1,
        }
        kwargs_deconv = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'convolution_param': deconv_param
        }
        bn_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'eps': 0.001,
        }
        sb_kwargs = {
            'bias_term': True,
            'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
            'filler': dict(type='constant', value=1.0),
            'bias_filler': dict(type='constant', value=0.0),
        }
        from_layer = baselayer
        add_layer = from_layer + "_deconv"
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)
        baselayer = add_layer
    if flag_concat_conv3:
        feature_layers = []
        feature_layers.append(net["conv3_2"])
        feature_layers.append(net[baselayer])
        add_layer = "concat345"
        net[add_layer] = L.Concat(*feature_layers, axis=1)
        baselayer = add_layer
    if flag_concat_conv4_last:
        feature_layers = []
        feature_layers.append(net["conv4_3"])
        feature_layers.append(net[baselayer])
        add_layer = "concat4and5"
        net[add_layer] = L.Concat(*feature_layers, axis=1)
        baselayer = add_layer

    # Stages
    flag_hasloss_befdecov = True
    flag_reducefirst = True
    use_stage = 3
    use_3_layers = [7,7,7]
    use_1_layers = 0
    n_channel = 128
    lrdecay = 1.0
    kernel_size = 3
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        ##################################################################################################################################
        # net = mPose_StageXDeconv_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
        #                    mask_vec="vec_mask", mask_heat="heat_mask", \
        #                    label_vec="vec_label", label_heat="heat_label", \
        #                    use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
        #                    base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,
        #                    flag_hasloss_befdecov=flag_hasloss_befdecov,flag_reducefirst=flag_reducefirst)
        ###################################################################################################################################
        # net = mPose_StageXDeconv_A_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1, \
        #                                mask_vec="vec_mask", mask_heat="heat_mask", \
        #                                label_vec="vec_label", label_heat="heat_label", \
        #                                use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
        #                                base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
        #                                kernel_size=kernel_size,flag_reducefirst=flag_reducefirst,scale_updown=scale_updown)
        ################################################################################################################################
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1, \
                                 mask_vec="vec_mask", mask_heat="heat_mask", \
                                 label_vec="vec_label", label_heat="heat_label", \
                                 use_3_layers=use_3_layers[stage], use_1_layers=use_1_layers, short_cut=short_cut, \
                                 base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
                                 kernel_size=kernel_size)
    # for Test
    # for Test
    if not train:
        if not flag_hasloss_befdecov:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers) + '_deconv'
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers[-1] + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers[-1] + use_1_layers)

        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net
def mPoseNet_COCO_DarkNetMultiStageChangeBaseA_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):

    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_concat_lowlevel = True
    flag_output_sigmoid  = False
    num_sublayers = [1, 1, 2, 3]
    num_channels = [512, 256, 512, 256, 512]
    alpha = 1
    net = YoloNetPart_StrideRemove1x1(net, num_sublayers=num_sublayers,num_channels=num_channels,from_layer=data_layer,
                                      lr=1, decay=1,alpha=alpha)

    conv_param = {
        'num_output': 256,
        'kernel_size': 2,
        'pad': 0,
        'stride': 2,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        'group': 1,
    }
    # conv_param = {"kernel_size": 4, "stride": 2, "num_output": 128, "group": 128, "pad": 1,
    #               "weight_filler": dict(type="bilinear"), "bias_term": False}
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param
    }

    from_layer = "conv5_{}".format(len(num_channels))
    out_layer = from_layer + "_Upsample"
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)

    if flag_concat_lowlevel:
        add_layer = "multi_scale_concat"
        feature_layers = []
        feature_layers.append(net["conv4_3"])
        feature_layers.append(net[out_layer])
        net[add_layer] = L.Concat(*feature_layers, axis=1)
    else:
        add_layer = out_layer

    baselayer = "convf"
    ConvBNUnitLayer(net, add_layer, baselayer, use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1,
                    stride=1,use_scale=True, leaky=False, lr_mult=1, decay_mult=1)


    # Stages
    use_stage = 3
    use_3_layers = 6
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,flag_sigmoid=flag_output_sigmoid)
    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_DarkNetMultiStageChangeBaseB_Train(net, data_layer="data", label_layer="label", train=True, **pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    # Darknet19
    flag_concat_lowlevel = True
    flag_output_sigmoid  = False
    num_sublayers = [1, 1, 2, 2]
    num_channels = [512, 256, 512, 256, 512]
    alpha = 1
    net = YoloNetPart_StrideRemove1x1(net, num_sublayers=num_sublayers,num_channels=num_channels,from_layer=data_layer,
                                      lr=1, decay=1,alpha=alpha)

    conv_param = {
        'num_output': 128,
        'kernel_size': 2,
        'pad': 0,
        'stride': 2,
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        'group': 1,
    }
    # conv_param = {"kernel_size": 4, "stride": 2, "num_output": 128, "group": 128, "pad": 1,
    #               "weight_filler": dict(type="bilinear"), "bias_term": False}
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'convolution_param': conv_param
    }

    from_layer = "conv5_{}".format(len(num_channels))
    out_layer = from_layer + "_Upsample"
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)

    if flag_concat_lowlevel:
        add_layer = "multi_scale_concat"
        feature_layers = []
        feature_layers.append(net["conv4_2"])
        feature_layers.append(net[out_layer])
        net[add_layer] = L.Concat(*feature_layers, axis=1)
    else:
        add_layer = out_layer

    baselayer = add_layer


    # Stages
    use_stage = 3
    use_3_layers = 6
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage+1, \
                           mask_vec="vec_mask", mask_heat="heat_mask", \
                           label_vec="vec_label", label_heat="heat_label", \
                           use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                           base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,flag_sigmoid=flag_output_sigmoid)
    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage,use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage,use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64*64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)
    return net

def mPoseNet_COCO_SmallFeatMap_ReconBase_Train(net, data_layer="data",flag_withTea = True,loss_weight=1.0):

    # Darknet19
    use_sub_layers = (1, 1, 3, 3, 5)
    num_channels = (32, 64, 128, 384, 512)
    kernel_sizes = (7, 3, 3, 3, 3)
    strides = (4,2,2,1,2)
    strides_last_flags = (True,True,True,True,False)# default All True
    pooling_flags = (False,) * 5
    ChangeNameAndChannel = {}
    deconv_channels = [128,128]
    addstrs = "_recon"
    net = YoloNetPartBigStride(net, from_layer=data_layer, use_bn=True, use_sub_layers=use_sub_layers, leaky=False,deconv_channels=deconv_channels,
						 num_channels = num_channels,kernel_sizes = kernel_sizes,strides = strides,strides_last_flags=strides_last_flags,
                        ChangeNameAndChannel=ChangeNameAndChannel,pooling_flags=pooling_flags,lr=1, decay=1,addstrs=addstrs)



    if flag_withTea:
        #### Teacher 15F
        # strid_convs = [1, 1, 1, 0, 0]
        # net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,
        #                           strid_conv=strid_convs, final_pool=False, lr=0, decay=0, leaky=True)
        # add_layer = 'conv5_5_upsample'
        # net[add_layer] = L.Reorg(net["conv5_5"], reorg_param=dict(up_down=P.Reorg.UP))

        ## Teach DarkTea8B
        leaky = False
        ChangeNameAndChannel = {"conv4_3": 128, "conv5_1": 512}
        net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False,
                          leaky=leaky, lr=0, decay=0, ChangeNameAndChannel=ChangeNameAndChannel)
        ####Both Teach DarkTea8B and DarkTea4A use the following deconv
        conv_param = {
            'num_output': 128,
            'kernel_size': 2,
            'pad': 0,
            'stride': 2,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
            'group': 1,
        }
        # conv_param = {"kernel_size": 4, "stride": 2, "num_output": 128, "group": 128, "pad": 1,
        #               "weight_filler": dict(type="bilinear"), "bias_term": False}
        kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0)],
            'convolution_param': conv_param
        }

        from_layer = "conv5_5"
        out_layer = from_layer + "_Upsample"
        net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)
        ref_layer1 = "conv4_3_new"
        ref_layer2 = "conv5_5_Upsample"
        recon_layer1 = "conv4_{}_recon_deconv".format(use_sub_layers[3])
        recon_layer2 = "conv5_{}_recon_deconv".format(use_sub_layers[4])

        net['loss1'] = L.EuclideanLoss(net[ref_layer1], net[recon_layer1], loss_weight=loss_weight)
        net['loss2'] = L.EuclideanLoss(net[ref_layer2], net[recon_layer2], loss_weight=loss_weight)
    return net

def mPoseNet_COCO_SmallFeatMap_ReconBase_A_Train(net, data_layer="data", flag_withTea = True):
    ### For Student
    num_channels = ((32,), (64,), (112, 56, 128, 64, 128), (176, 88, 176, 88, 208))
    kernel_sizes = ((7,), (3,), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
    strides = ((4,), (2,), (1, 1, 1, 1, 1), (2, 1, 1, 1, 1))
    num_channel_deconv = 128
    scale_deconv = 2
    addstrs = "_recon"
    flag_deconv = True
    flag_deconv_relu = False
    special_layers = ""
    net = Flexible_Base(net, from_layer="data", use_bn=True, leaky=False, num_channels=num_channels,
                        kernel_sizes=kernel_sizes, strides=strides, lr=1, decay=1, flag_deconv=flag_deconv,
                        flag_deconv_relu=flag_deconv_relu,num_channel_deconv=num_channel_deconv, scale_deconv=scale_deconv, special_layers=special_layers,
                        addstrs=addstrs)
    recon_layer1 = "conv3_{}".format(len(num_channels[-2])) + addstrs
    recon_layer2 = "conv4_{}".format(len(num_channels[-1])) + addstrs + "_deconv"

    if flag_withTea:
        ###Teacher DarkNetTea8B
        leaky = False
        ChangeNameAndChannel = {"conv4_3":128,"conv5_1":512}
        net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False,
                          leaky=leaky, lr=0, decay=0, ChangeNameAndChannel=ChangeNameAndChannel)

        ### Teacher DarkNetTea4A
        # num_sublayers_tea = [1, 1, 2, 3]
        # num_channels_tea = [512, 256,512, 256,128]
        # alpha = 1
        # net = YoloNetPart_StrideRemove1x1(net, num_sublayers=num_sublayers_tea, num_channels=num_channels_tea,
        #                                   from_layer=data_layer,lr=0, decay=0,alpha=alpha,fix_layer=5,fix_sublayer=1)

        conv_param = {
            'num_output': 128,
            'kernel_size': 2,
            'pad': 0,
            'stride': 2,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
            'group': 1,
        }
        # conv_param = {"kernel_size": 4, "stride": 2, "num_output": 128, "group": 128, "pad": 1,
        #               "weight_filler": dict(type="bilinear"), "bias_term": False}
        kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0)],
            'convolution_param': conv_param
        }

        from_layer = "conv5_5"
        out_layer = from_layer + "_Upsample"
        net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)
        ref_layer1 = "conv4_3_new"
        ref_layer2 = "conv5_5_Upsample"


        net['loss1'] = L.EuclideanLoss(net[ref_layer1], net[recon_layer1], loss_weight=1.0)
        net['loss2'] = L.EuclideanLoss(net[ref_layer2], net[recon_layer2], loss_weight=1.0)
    return net, recon_layer1, recon_layer2
def mPoseNet_COCO_SmallFeatMap_PoseFromRecon_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    flag_deconv_explicit = False
    # Darknet19
    # use_sub_layers = (1, 1, 3, 3, 5)
    # num_channels = (32, 64, 128, 256, 512)
    # kernel_sizes = (7, 3, 3, 3, 3)
    # strides = (4,2,2,1,2)
    # strides_last_flags = (True,True,True,True,False)# default All True
    # pooling_flags = (False,) * 5
    # ChangeNameAndChannel = {"conv1":32}
    # addstrs = ""
    # net = YoloNetPartBigStride(net, from_layer=data_layer, use_bn=True, use_sub_layers=use_sub_layers, leaky=False,
		# 				 num_channels = num_channels,kernel_sizes = kernel_sizes,strides = strides,strides_last_flags=strides_last_flags,
    #                     ChangeNameAndChannel=ChangeNameAndChannel,pooling_flags=pooling_flags,lr=1, decay=1,addstrs=addstrs)

    num_channels = ((32,), (64,), (64, 32, 128), (256, 64, 256, 64, 256, 128, 128))
    kernel_sizes = ((7,), (3,), (3, 3, 3), (3, 1, 3, 1, 3, 1, 3))
    strides = ((4,), (2,), (1, 1, 1), (2, 1, 1, 1, 1, 1, 1))
    num_channel_deconv = 128
    scale_deconv = 2
    addstrs = "_recon"
    flag_deconv = True
    flag_deconv_relu = False
    special_layers = ""
    net = Flexible_Base(net, from_layer=data_layer, use_bn=True, leaky=False, num_channels=num_channels,
                        kernel_sizes=kernel_sizes, strides=strides, lr=1, decay=1, flag_deconv=flag_deconv,
                        flag_deconv_relu=flag_deconv_relu,
                        num_channel_deconv=num_channel_deconv, scale_deconv=scale_deconv, special_layers=special_layers,
                        addstrs=addstrs)

    if flag_deconv_explicit:
        deconv_param = {
            'num_output': 256,
            'kernel_size': 2,
            'pad': 0,
            'stride': 2,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),
            'group': 1,
        }
        kwargs_deconv = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'convolution_param': deconv_param
        }
        bn_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'eps': 0.001,
        }
        sb_kwargs = {
            'bias_term': True,
            'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
            'filler': dict(type='constant', value=1.0),
            'bias_filler': dict(type='constant', value=0.2),
        }
        from_layer = "conv4_3" + addstrs
        add_layer = from_layer + "_deconv"
        print from_layer, add_layer
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)

        deconv_param1 = {
            'num_output': 128,
            'kernel_size': 4,
            'pad': 0,
            'stride': 4,
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),
            'group': 1,
        }
        kwargs_deconv1 = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'convolution_param': deconv_param1
        }
        from_layer = "conv5_5" + addstrs
        add_layer = from_layer + "_deconv"
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv1)
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)

    feaLayers = []
    feaLayers.append(net["conv3_3_recon"])
    feaLayers.append(net["conv4_7_recon_deconv"])
    baselayer = "convf"
    net[baselayer] = L.Concat(*feaLayers, axis=1)

    use_stage = 3
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    flag_output_sigmoid = False
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1,
                                 mask_vec="vec_mask", mask_heat="heat_mask", \
                                 label_vec="vec_label", label_heat="heat_label", \
                                 use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                                 base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
                                 kernel_size=kernel_size, flag_sigmoid=flag_output_sigmoid)

    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64 * 64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)


    return net
def mPoseNet_COCO_SmallFeatMap_PoseFromReconNew_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))

    net, ref_layer1, ref_layer2= mPoseNet_COCO_SmallFeatMap_ReconBase_A_Train(net, data_layer="data", flag_withTea=False)
    feaLayers = []
    feaLayers.append(net[ref_layer1])
    feaLayers.append(net[ref_layer2])
    baselayer = "convf"
    net[baselayer] = L.Concat(*feaLayers, axis=1)
    use_stage = 3
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    flag_output_sigmoid = False
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1,
                                 mask_vec="vec_mask", mask_heat="heat_mask", \
                                 label_vec="vec_label", label_heat="heat_label", \
                                 use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                                 base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
                                 kernel_size=kernel_size, flag_sigmoid=flag_output_sigmoid)

    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64 * 64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)


    return net
def mPoseNet_COCO_SmallFeatMap_PoseFromReconWithRecon_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34,52,86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34,52,86,104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))

    net = mPoseNet_COCO_SmallFeatMap_ReconBase_Train(net,data_layer,flag_withTea=train,loss_weight=0.2)

    feaLayers = []
    feaLayers.append(net["conv4_3_recon_deconv"])
    feaLayers.append(net["conv5_5_recon_deconv"])
    baselayer = "convf"
    net[baselayer] = L.Concat(*feaLayers, axis=1)

    use_stage = 3
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    flag_output_sigmoid = False
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            short_cut = False
        else:
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1,
                                 mask_vec="vec_mask", mask_heat="heat_mask", \
                                 label_vec="vec_label", label_heat="heat_label", \
                                 use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=short_cut, \
                                 base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,
                                 kernel_size=kernel_size, flag_sigmoid=flag_output_sigmoid)

    # for Test
    if not train:
        if flag_output_sigmoid:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers) + "_sig"
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers) + "_sig"
        else:
            conv_vec = "stage{}_conv{}_vec".format(use_stage, use_3_layers + use_1_layers)
            conv_heat = "stage{}_conv{}_heat".format(use_stage, use_3_layers + use_1_layers)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(3)
        net[outlayer] = L.Concat(*feaLayers, axis=1)
        # Resize
        resize_kwargs = {
            'factor': pose_test_kwargs.get("resize_factor", 8),
            'scale_gap': pose_test_kwargs.get("resize_scale_gap", 0.3),
            'start_scale': pose_test_kwargs.get("resize_start_scale", 1.0),
        }
        net.resized_map = L.ImResize(net[outlayer], name="resize", imresize_param=resize_kwargs)
        # Nms
        nms_kwargs = {
            'threshold': pose_test_kwargs.get("nms_threshold", 0.05),
            'max_peaks': pose_test_kwargs.get("nms_max_peaks", 100),
            'num_parts': pose_test_kwargs.get("nms_num_parts", 18),
        }
        net.joints = L.Nms(net.resized_map, name="nms", nms_param=nms_kwargs)
        # ConnectLimbs
        connect_kwargs = {
            'is_type_coco': pose_test_kwargs.get("conn_is_type_coco", True),
            'max_person': pose_test_kwargs.get("conn_max_person", 10),
            'max_peaks_use': pose_test_kwargs.get("conn_max_peaks_use", 20),
            'iters_pa_cal': pose_test_kwargs.get("conn_iters_pa_cal", 10),
            'connect_inter_threshold': pose_test_kwargs.get("conn_connect_inter_threshold", 0.05),
            'connect_inter_min_nums': pose_test_kwargs.get("conn_connect_inter_min_nums", 8),
            'connect_min_subset_cnt': pose_test_kwargs.get("conn_connect_min_subset_cnt", 3),
            'connect_min_subset_score': pose_test_kwargs.get("conn_connect_min_subset_score", 0.4),
        }
        net.limbs = L.Connectlimb(net.resized_map, net.joints, connect_limb_param=connect_kwargs)
        # Eval
        eval_kwargs = {
            'stride': 8,
            'area_thre': pose_test_kwargs.get("eval_area_thre", 64 * 64),
            'oks_thre': pose_test_kwargs.get("eval_oks_thre", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
        }
        net.eval = L.PoseEval(net.limbs, net.gt, pose_eval_param=eval_kwargs)



    return net