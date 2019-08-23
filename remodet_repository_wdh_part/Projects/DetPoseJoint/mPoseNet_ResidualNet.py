import caffe

from caffe import layers as L
from caffe import params as P
from PyLib.NetLib.MultiScaleLayer import *
from PyLib.NetLib.ConvBNLayer import *
from PyLib.NetLib.PoseNet import *
from PyLib.NetLib.VggNet import *
from mPoseBaseNet import *
from mPoseNet_Reduce import *
from mPoseNet_DarkNet import *
from BaseNet import *
from DAPNet import pose_string
def ResNet_UnitA(net, base_layer, name_prefix, stride, num_channel,bridge = False,num_channel_change = 0,
                     flag_hasresid = True,channel_scale = 4,check_macc = False,flag_withparamname = False):
    add_layer = name_prefix + '_1x1Conv1'
    ConvBNUnitLayer(net, base_layer, add_layer, use_bn=True, use_relu=True,
					num_output=num_channel/channel_scale, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=False,
					check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    from_layer = add_layer
    add_layer = name_prefix + '_3x3Conv'
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=num_channel / channel_scale, kernel_size=3, pad=1, stride=stride, use_scale=True,
                    n_group=1,check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    from_layer = add_layer
    add_layer = name_prefix + '_1x1Conv2'
    if num_channel_change != 0:
        num_channel = num_channel_change
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,
                    num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,
                    check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)

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
                                num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,check_macc=check_macc,
                flag_withparamname = flag_withparamname,pose_string=pose_string)
                add_layer2 = add_layer
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)

def ResNetTwoLayers_UnitA(net, base_layer, name_prefix, stride, num_channel,bridge = False,num_channel_change = 0,
                     flag_hasresid = True,channel_scale = 4,check_macc = False,lr_mult=0.1,decay_mult=1.0,flag_withparamname=False):
    add_layer = name_prefix + '_1x1Conv'
    ConvBNUnitLayer(net, base_layer, add_layer, use_bn=True, use_relu=True,lr_mult=lr_mult, decay_mult=decay_mult,
					num_output=num_channel/channel_scale, kernel_size=1, pad=0, stride=1, use_scale=True, leaky=False,
					check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    from_layer = add_layer+pose_string

    add_layer = name_prefix + '_3x3Conv'
    if num_channel_change != 0:
        num_channel = num_channel_change
    ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,lr_mult=lr_mult, decay_mult=decay_mult,
                    num_output=num_channel, kernel_size=3, pad=1, stride=stride, use_scale=True,
                    n_group=1,check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
    # for old_name in net.keys():
    #     print old_name,'$$$$$'
    if flag_hasresid:
        from_layer = add_layer+pose_string
        if stride == 2:
            feature_layers = []
            feature_layers.append(net[from_layer])
            add_layer = name_prefix + '_AVEPool'+pose_string
            net[add_layer] = L.Pooling(net[base_layer], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)

            feature_layers.append(net[add_layer])
            add_layer = name_prefix + '_Concat'+pose_string
            net[add_layer] = L.Concat(*feature_layers, axis=1)
        # for old_name in net.keys():
        #     print old_name,'^^^'
        else:
            add_layer1 = from_layer
            if bridge:
                from_layer = base_layer
                add_layer = name_prefix + '_bridge'
                # for old_name in net.keys():
                #     print old_name,'!!!'
                ConvBNUnitLayer(net, from_layer, add_layer, use_bn=True, use_relu=False, leaky=False,lr_mult=lr_mult, decay_mult=decay_mult,
                                num_output=num_channel, kernel_size=1, pad=0, stride=1, use_scale=True,check_macc=check_macc,flag_withparamname=flag_withparamname,pose_string=pose_string)
                # for old_name in net.keys():
                #     print old_name,'~~~'
                add_layer2 = add_layer+pose_string
            else:
                add_layer2 = base_layer
            add_layer = name_prefix + '_Add'+pose_string
            net[add_layer] = L.Eltwise(net[add_layer1], net[add_layer2], eltwise_param=dict(operation=P.Eltwise.SUM))

    from_layer = add_layer
    add_layer = name_prefix + '_relu'+pose_string
    print from_layer,add_layer
    net[add_layer] = L.ReLU(net[from_layer], in_place=True)
def resnext_block(bottom, base_output=64, card=32,kernel_size = 3):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers

    Args:
        card:
        card:
    """
    conv1 = L.Convolution(bottom, num_output=base_output * (card / 16), kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output * (card / 16), kernel_size=kernel_size, stride=1, pad=(kernel_size - 1)/2, group=card,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'),engine=P.Convolution.CAFFE)
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output * 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(bottom, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, eltwise, eltwise_relu

def resnet_block(bottom, base_output=64,kernel_size = 3):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers

    Args:
        card:
    """
    conv1 = L.Convolution(bottom, num_output=base_output, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output , kernel_size=kernel_size, stride=1, pad=(kernel_size - 1)/2,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output*4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(bottom, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, eltwise, eltwise_relu
def match_block(bottom, base_output=64, stride=2, card=32, kernel_size = 3):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    conv1 = L.Convolution(bottom, num_output=base_output * (card / 16), kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output * (card / 16), kernel_size=kernel_size, stride=stride, pad=(kernel_size-1)/2, group=card,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'),engine=P.Convolution.CAFFE)
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output * 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    match = L.Convolution(bottom, num_output=base_output * 4, kernel_size=1, stride=stride, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    match_bn = L.BatchNorm(match, use_global_stats=False, in_place=True)
    match_scale = L.Scale(match, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(match, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, match, match_bn, match_scale, eltwise, eltwise_relu

def match_block_stage(bottom, base_output=64, stride=2, card=32, kernel_size = 3):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    conv1 = L.Convolution(bottom, num_output=base_output, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv1_bn = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    conv1_scale = L.Scale(conv1, scale_param=dict(bias_term=True), in_place=True)
    conv1_relu = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(conv1, num_output=base_output, kernel_size=kernel_size, stride=stride, pad=(kernel_size-1)/2,
                          bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'),engine=P.Convolution.CAFFE)
    conv2_bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    conv2_scale = L.Scale(conv2, scale_param=dict(bias_term=True), in_place=True)
    conv2_relu = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(conv2, num_output=base_output * 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    conv3_bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    conv3_scale = L.Scale(conv3, scale_param=dict(bias_term=True), in_place=True)

    match = L.Convolution(bottom, num_output=base_output * 4, kernel_size=1, stride=stride, pad=0, bias_term=False,
                          param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'),group=card)
    match_bn = L.BatchNorm(match, use_global_stats=False, in_place=True)
    match_scale = L.Scale(match, scale_param=dict(bias_term=True), in_place=True)

    eltwise = L.Eltwise(match, conv3, eltwise_param=dict(operation=1))
    eltwise_relu = L.ReLU(eltwise, in_place=True)

    return conv1, conv1_bn, conv1_scale, conv1_relu, conv2, conv2_bn, conv2_scale, conv2_relu, \
           conv3, conv3_bn, conv3_scale, match, match_bn, match_scale, eltwise, eltwise_relu
def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_term = False)
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_term=False)
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1)  # base_output x n x n
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)  # base_output x n x n
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)  # 4*base_output x n x n

    residual, residual_relu = \
        eltwize_relu(bottom, branch2c)  # 4*base_output x n x n

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64):
    """

    :param stride: stride
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch1, branch1_bn, branch1_scale = \
        conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, stride=stride)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)

    residual, residual_relu = \
        eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


branch_shortcut_string = 'net.res(stage)a_branch1, net.res(stage)a_branch1_bn, net.res(stage)a_branch1_scale, \
        net.res(stage)a_branch2a, net.res(stage)a_branch2a_bn, net.res(stage)a_branch2a_scale, net.res(stage)a_branch2a_relu, \
        net.res(stage)a_branch2b, net.res(stage)a_branch2b_bn, net.res(stage)a_branch2b_scale, net.res(stage)a_branch2b_relu, \
        net.res(stage)a_branch2c, net.res(stage)a_branch2c_bn, net.res(stage)a_branch2c_scale, net.res(stage)a, net.res(stage)a_relu = \
            residual_branch_shortcut((bottom), stride=(stride), base_output=(num))'

branch_string = 'net.res(stage)b(order)_branch2a, net.res(stage)b(order)_branch2a_bn, net.res(stage)b(order)_branch2a_scale, \
        net.res(stage)b(order)_branch2a_relu, net.res(stage)b(order)_branch2b, net.res(stage)b(order)_branch2b_bn, \
        net.res(stage)b(order)_branch2b_scale, net.res(stage)b(order)_branch2b_relu, net.res(stage)b(order)_branch2c, \
        net.res(stage)b(order)_branch2c_bn, net.res(stage)b(order)_branch2c_scale, net.res(stage)b(order), net.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num))'

resnext_string = 'net.resx(n)_conv1, net.resx(n)_conv1_bn, net.resx(n)_conv1_scale, net.resx(n)_conv1_relu, \
         net.resx(n)_conv2, net.resx(n)_conv2_bn, net.resx(n)_conv2_scale, net.resx(n)_conv2_relu, net.resx(n)_conv3, \
         net.resx(n)_conv3_bn, net.resx(n)_conv3_scale, net.resx(n)_elewise, net.resx(n)_elewise_relu = \
             resnext_block((bottom), base_output=(base), card=(c), kernel_size=(k))'
resnext_string_stage = 'net.stage#(n)_(m)_conv1, net.stage#(n)_(m)_conv1_bn, net.stage#(n)_(m)_conv1_scale, net.stage#(n)_(m)_conv1_relu, \
         net.stage#(n)_(m)_conv2, net.stage#(n)_(m)_conv2_bn, net.stage#(n)_(m)_conv2_scale, net.stage#(n)_(m)_conv2_relu, net.stage#(n)_(m)_conv3, \
         net.stage#(n)_(m)_conv3_bn, net.stage#(n)_(m)_conv3_scale, net.stage#(n)_(m)_elewise, net.stage#(n)_(m)_elewise_relu = \
             resnext_block((bottom), base_output=(base), card=(c), kernel_size=(k))'
resnet_string_stage = 'net.stage#(n)_(m)_conv1, net.stage#(n)_(m)_conv1_bn, net.stage#(n)_(m)_conv1_scale, net.stage#(n)_(m)_conv1_relu, \
         net.stage#(n)_(m)_conv2, net.stage#(n)_(m)_conv2_bn, net.stage#(n)_(m)_conv2_scale, net.stage#(n)_(m)_conv2_relu, net.stage#(n)_(m)_conv3, \
         net.stage#(n)_(m)_conv3_bn, net.stage#(n)_(m)_conv3_scale, net.stage#(n)_(m)_elewise, net.stage#(n)_(m)_elewise_relu = \
             resnet_block((bottom), base_output=(base), kernel_size=(k))'
match_string = 'net.resx(n)_conv1, net.resx(n)_conv1_bn, net.resx(n)_conv1_scale, net.resx(n)_conv1_relu, \
     net.resx(n)_conv2, net.resx(n)_conv2_bn, net.resx(n)_conv2_scale, net.resx(n)_conv2_relu, net.resx(n)_conv3, \
     net.resx(n)_conv3_bn, net.resx(n)_conv3_scale, net.resx(n)_match_conv, net.resx(n)_match_conv_bn, net.resx(n)_match_conv_scale,\
     net.resx(n)_elewise, net.resx(n)_elewise_relu = match_block((bottom), base_output=(base), stride=(s), card=(c), kernel_size=(k))'
match_string_stage = 'net.stage#(n)_(m)_conv1, net.stage#(n)_(m)_conv1_bn, net.stage#(n)_(m)_conv1_scale, net.stage#(n)_(m)_conv1_relu, \
     net.stage#(n)_(m)_conv2, net.stage#(n)_(m)_conv2_bn, net.stage#(n)_(m)_conv2_scale, net.stage#(n)_(m)_conv2_relu, net.stage#(n)_(m)_conv3, \
     net.stage#(n)_(m)_conv3_bn, net.stage#(n)_(m)_conv3_scale, net.stage#(n)_(m)_match_conv, net.stage#(n)_(m)_match_conv_bn, net.stage#(n)_(m)_match_conv_scale,\
     net.stage#(n)_(m)_elewise, net.stage#(n)_(m)_elewise_relu = match_block_stage((bottom), base_output=(base), stride=(s), card=1, kernel_size=(k))'

def ResNeXt_layers(net, from_layer, card=32, stages=(3, 4, 6, 3)):
    """

    :param batch_size: the batch_size of train and test phase
    :param phase: TRAIN or TEST
    :param stages: the num of layers = 2 + 3*sum(stages), layers would better be chosen from [50, 101, 152]
                   {every stage is composed of 1 residual_branch_shortcut module and stage[i]-1 residual_branch
                   modules, each module consists of 3 conv layers}
                    (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
    """
    net.conv1 = L.Convolution(net[from_layer], num_output=64, kernel_size=7, stride=2, pad=3, bias_term=False,
                            param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type='xavier'))
    net.conv1_bn = L.BatchNorm(net.conv1, use_global_stats=False, in_place=True)
    net.conv1_scale = L.Scale(net.conv1, scale_param=dict(bias_term=True), in_place=True)
    net.conv1_relu = L.ReLU(net.conv1, in_place=True)  # 64x112x112
    net.pool1 = L.Pooling(net.conv1, kernel_size=3, stride=2, pad=0, pool=P.Pooling.MAX)  # 64x56x56

    for num in xrange(len(stages)):  # num = 0, 1, 2, 3
        for i in xrange(stages[num]):
            if i == 0:
                stage_string = match_string
                bottom_string = ['net.pool1', 'net.resx{}_elewise'.format(str(sum(stages[:1]))),
                                 'net.resx{}_elewise'.format(str(sum(stages[:2]))),
                                 'net.resx{}_elewise'.format(str(sum(stages[:3])))][num]
            else:
                stage_string = resnext_string
                bottom_string = 'net.resx{}_elewise'.format(str(sum(stages[:num]) + i))
            print num, i
            exec (stage_string.replace('(bottom)', bottom_string).
                  replace('(base)', str(2 ** num * 64)).
                  replace('(n)', str(sum(stages[:num]) + i + 1)).
                  replace('(s)', str(int(num > 0) + 1)).
                  replace('(c)', str(card)).
                  replace('(k)', str(3)))

    return net

def ResNet_layers(net, from_layer, stages=(3, 4, 6, 3)):
    """

    :param batch_size: the batch_size of train and test phase
    :param phase: TRAIN or TEST
    :param stages: the num of layers = 2 + 3*sum(stages), layers would better be chosen from [50, 101, 152]
                   {every stage is composed of 1 residual_branch_shortcut module and stage[i]-1 residual_branch
                   modules, each module consists of 3 conv layers}
                    (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
    """
    net.conv1, net.conv1_bn, net.conv1_scale, net.conv1_relu = \
        conv_bn_scale_relu(net[from_layer], num_output=64, kernel_size=7, stride=2, pad=3)  # 64x112x112
    net.pool1 = L.Pooling(net.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56

    for num in xrange(len(stages)):  # num = 0, 1, 2, 3
        for i in xrange(stages[num]):
            if i == 0:
                stage_string = branch_shortcut_string
                bottom_string = ['net.pool1', 'net.res2b%s' % str(stages[0] - 1), 'net.res3b%s' % str(stages[1] - 1),
                                 'net.res4b%s' % str(stages[2] - 1)][num]
            else:
                stage_string = branch_string
                if i == 1:
                    bottom_string = 'net.res%sa' % str(num + 2)
                else:
                    bottom_string = 'net.res%sb%s' % (str(num + 2), str(i - 1))
            exec (stage_string.replace('(stage)', str(num + 2)).replace('(bottom)', bottom_string).
                  replace('(num)', str(2 ** num * 64)).replace('(order)', str(i)).
                  replace('(stride)', str(int(num > 0) + 1)))

    return net

def mPoseNet_ResNeXt_MultiStages_Train(net, data_layer="data", label_layer="label", train=True, lr = 1, decay = 1,**pose_test_kwargs):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2 * lr, decay_mult=0)],
                   'weight_filler': dict(type='gaussian', std=0.01),
                   'bias_filler': dict(type='constant', value=0)}
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
    stages = (3, 4, 8)
    net = ResNeXt_layers(net, from_layer=data_layer, card=32, stages=stages)
    from_layer = 'resx{}_elewise'.format(str(sum(stages)))
    add_layer = 'upsample'
    net[add_layer] = L.Reorg(net[from_layer], reorg_param=dict(up_down=P.Reorg.UP))
    base_layer = add_layer
    bottom_string = 'net.{}'.format(base_layer)

    use_stages = 4
    use_sub_layers = 5
    num_output = 32
    kernel_size = 5
    for i_stage in xrange(use_stages):
        for i_sub in xrange(use_sub_layers):
            for str_i in ['vec', 'heat']:
                if i_sub == 0:
                    stage_string = match_string_stage.replace('#', str_i)
                else:
                    stage_string = resnext_string_stage.replace('#', str_i)
                if i_sub != 0:
                    bottom_string = 'net.stage#(n)_(m)_elewise'.\
                        replace('#', str_i).replace('(n)',str(i_stage + 1)).\
                        replace('(m)', str(i_sub))
                exec (stage_string.replace('(bottom)', bottom_string).
                      replace('(base)', str(num_output)).
                      replace('(n)', str(i_stage+1)).
                      replace('(m)', str(i_sub+1)).
                      replace('(s)', str(1)).
                      replace('(c)', str(32)).
                      replace('(k)', str(kernel_size)))


        from1_layer = 'stage#(n)_(m)_elewise'.replace('#', 'vec').replace('(n)', str(i_stage + 1)).replace(
                '(m)', str(use_sub_layers))
        conv_vec = "stage{}_conv{}_vec".format(i_stage + 1, use_sub_layers + 1)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)

        from2_layer = 'stage#(n)_(m)_elewise'.replace('#', 'heat').replace('(n)', str(i_stage + 1)).replace(
            '(m)', str(use_sub_layers))
        conv_heat = "stage{}_conv{}_heat".format(i_stage + 1, use_sub_layers + 1)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)

        weight_vec = "weight_stage{}_vec".format(i_stage+ 1)
        weight_heat = "weight_stage{}_heat".format(i_stage+1)
        loss_vec = "loss_stage{}_vec".format(i_stage+1)
        loss_heat = "loss_stage{}_heat".format(i_stage+1)
        net[weight_vec] = L.Eltwise(net[conv_vec], net.vec_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_vec] = L.EuclideanLoss(net[weight_vec], net.vec_label, loss_weight=1)
        net[weight_heat] = L.Eltwise(net[conv_heat], net.heat_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_heat] = L.EuclideanLoss(net[weight_heat], net.heat_label, loss_weight=1)

        if i_stage != use_stages - 1:
            out_layer = 'concat_stage{}'.format(str(i_stage + 1))
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
            bottom_string = 'net.{}'.format(out_layer)


    if not train:
        print(net.keys())
        conv_vec = "stage{}_conv{}_vec".format(use_stages,use_sub_layers + 1)
        conv_heat = "stage{}_conv{}_heat".format(use_stages,use_sub_layers + 1)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(use_stages)
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

def mPoseNet_ResNet_MultiStages_Train(net, data_layer="data", label_layer="label", train=True, lr = 1, decay = 1,**pose_test_kwargs):
    kwargs = {'param': [dict(lr_mult=lr, decay_mult=decay), dict(lr_mult=2 * lr, decay_mult=0)],
                   'weight_filler': dict(type='gaussian', std=0.01),
                   'bias_filler': dict(type='constant', value=0)}
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
    stages = (3, 4, 6)
    net = ResNet_layers(net, from_layer=data_layer, stages=stages)
    from_layer = 'res%sb%s' % (str(len(stages) + 1), str(stages[-1] - 1))
    add_layer = 'upsample'
    net[add_layer] = L.Reorg(net[from_layer], reorg_param=dict(up_down=P.Reorg.UP))
    base_layer = add_layer
    bottom_string = 'net.{}'.format(base_layer)

    use_stages = 4
    use_sub_layers = 5
    num_output = 32
    kernel_size = 5
    for i_stage in xrange(use_stages):
        for i_sub in xrange(use_sub_layers):
            for str_i in ['vec', 'heat']:
                print i_stage, i_sub, str_i
                if i_sub == 0:
                    stage_string = match_string_stage.replace('#', str_i)
                else:
                    stage_string = resnet_string_stage.replace('#', str_i)
                if i_sub != 0:
                    bottom_string = 'net.stage#(n)_(m)_elewise'.\
                        replace('#', str_i).replace('(n)',str(i_stage + 1)).\
                        replace('(m)', str(i_sub))
                exec (stage_string.replace('(bottom)', bottom_string).
                      replace('(base)', str(num_output)).
                      replace('(n)', str(i_stage+1)).
                      replace('(m)', str(i_sub+1)).
                      replace('(s)', str(1)).
                      replace('(k)', str(kernel_size)))


        from1_layer = 'stage#(n)_(m)_elewise'.replace('#', 'vec').replace('(n)', str(i_stage + 1)).replace(
                '(m)', str(use_sub_layers))
        conv_vec = "stage{}_conv{}_vec".format(i_stage + 1, use_sub_layers + 1)
        net[conv_vec] = L.Convolution(net[from1_layer], num_output=34, pad=1, kernel_size=3, **kwargs)

        from2_layer = 'stage#(n)_(m)_elewise'.replace('#', 'heat').replace('(n)', str(i_stage + 1)).replace(
            '(m)', str(use_sub_layers))
        conv_heat = "stage{}_conv{}_heat".format(i_stage + 1, use_sub_layers + 1)
        net[conv_heat] = L.Convolution(net[from2_layer], num_output=18, pad=1, kernel_size=3, **kwargs)

        weight_vec = "weight_stage{}_vec".format(i_stage+ 1)
        weight_heat = "weight_stage{}_heat".format(i_stage+1)
        loss_vec = "loss_stage{}_vec".format(i_stage+1)
        loss_heat = "loss_stage{}_heat".format(i_stage+1)
        net[weight_vec] = L.Eltwise(net[conv_vec], net.vec_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_vec] = L.EuclideanLoss(net[weight_vec], net.vec_label, loss_weight=1)
        net[weight_heat] = L.Eltwise(net[conv_heat], net.heat_mask, eltwise_param=dict(operation=P.Eltwise.PROD))
        net[loss_heat] = L.EuclideanLoss(net[weight_heat], net.heat_label, loss_weight=1)

        if i_stage != use_stages - 1:
            out_layer = 'concat_stage{}'.format(str(i_stage + 1))
            fea_layers = []
            fea_layers.append(net[conv_vec])
            fea_layers.append(net[conv_heat])
            assert base_layer in net.keys()
            fea_layers.append(net[base_layer])
            net[out_layer] = L.Concat(*fea_layers, axis=1)
            bottom_string = 'net.{}'.format(out_layer)

    for key in net.keys():
        print key
    if not train:
        print(net.keys())
        conv_vec = "stage{}_conv{}_vec".format(use_stages,use_sub_layers + 1)
        conv_heat = "stage{}_conv{}_heat".format(use_stages,use_sub_layers + 1)
        net.vec_out = L.Eltwise(net.vec_mask, net[conv_vec], eltwise_param=dict(operation=P.Eltwise.PROD))
        net.heat_out = L.Eltwise(net.heat_mask, net[conv_heat], eltwise_param=dict(operation=P.Eltwise.PROD))
        feaLayers = []
        feaLayers.append(net.heat_out)
        feaLayers.append(net.vec_out)
        outlayer = "concat_stage{}".format(use_stages)
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

def ResidualReduce_Base_A(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 0, 0, 0),
    channel_scale = 3,num_channel_deconv = (128,128),lr=1,decay=1,add_strs=""):
    num_output = 32
    out_layer = 'conv1' + add_strs
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,num_output=num_output,
                    kernel_size=7, pad=3, stride=4, use_scale=True, leaky=False, lr_mult=lr,decay_mult=decay,pose_string=pose_string)

    from_layer = out_layer
    out_layer = 'pool1' + add_strs
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=3, stride=2, pad=0)
    num_output = 64
    kernel_size = 3
    out_layer = "conv2_1" + add_strs
    ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=num_output, kernel_size=kernel_size, pad=(kernel_size - 1) / 2, stride=2, use_scale=True,
                    leaky=False, lr_mult=lr, decay_mult=decay,pose_string=pose_string)
    from_layer = out_layer
    feat_layers = []
    feat_layers.append(net["pool1" + add_strs])
    feat_layers.append(net[from_layer])
    out_layer = "conv2_1_concat" + add_strs
    net[out_layer] = L.Concat(*feat_layers, axis=1)


    for sublayer in xrange(use_sub_layers[0]):
        base_layer = out_layer
        name_prefix = 'conv2_{}'.format(sublayer + 2) + add_strs
        ResNet_UnitA(net, base_layer, name_prefix, 1, num_channels[0], bridge=True,
                     num_channel_change=0, flag_hasresid=True,channel_scale=channel_scale)
        out_layer = name_prefix + '_relu'

    for layer in xrange(1, len(use_sub_layers)):
        num_output_layer = num_channels[layer]
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
            ResNet_UnitA(net, base_layer, name_prefix, stride, num_output_layer,bridge = bridge,
                         num_channel_change = num_channel_change,flag_hasresid = True,channel_scale=channel_scale)
            out_layer = name_prefix + '_relu'
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
    if len(num_channel_deconv) == 2:
        deconv_param = {
            'num_output': num_channel_deconv[0],
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
        from_layer = "conv3_6{}_Add".format(add_strs)
        add_layer = from_layer + "_deconv"
        net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)

    deconv_param1 = {
        'num_output': num_channel_deconv[-1],
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
    from_layer = "conv4_7{}_Add".format(add_strs)
    add_layer = from_layer + "_deconv"
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv1)
    bn_name = add_layer + '_bn'
    net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
    sb_name = add_layer + '_scale'
    net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
    relu_name = add_layer + '_relu'
    net[relu_name] = L.ReLU(net[add_layer], in_place=True)
    return net

def ResidualShuffleVariant_Base_A(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,num_channel_deconv = 128,lr=1,decay=1,flag_deconvwithrelu = True,add_strs=""):
    out_layer = 'conv1' + add_strs
    ConvBNUnitLayer(net, data_layer, out_layer, use_bn=True, use_relu=True,
                    num_output=32, kernel_size=3, pad=1, stride=2, use_scale=True, leaky=False, lr_mult=lr,
                    decay_mult=decay,pose_string=pose_string)
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

            ResNet_UnitA(net, base_layer, name_prefix, stride, num_channel_layer, bridge=bridge, num_channel_change=num_channel_change,
                         flag_hasresid=True, channel_scale=channel_scale, check_macc=False)
            out_layer = name_prefix + '_relu'

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
    deconv_param = {
        'num_output': num_channel_deconv,
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
    from_layer = "conv3_{}{}_Add".format(use_sub_layers[-1],add_strs)
    add_layer = from_layer + "_deconv"
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    if flag_deconvwithrelu:
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)
    return net
def ResidualVariant_Base_A(net, data_layer="data",use_sub_layers = (2, 6, 7),num_channels = (128, 144, 288),output_channels = (0, 256,128),
    channel_scale = 4,num_channel_deconv = 128,lr=0.1,decay=1.0,flag_deconvwithrelu = True,add_strs="",flag_withparamname=False):

  ####
    # global pose_string
    # pose_string='_pose'
    # net = ResidualVariant_Base_A_base(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
    #                       output_channels=output_channels,channel_scale=channel_scale,lr=lr, decay=1, add_strs=add_strs,flag_withparamname=flag_withparamname,pose_string=pose_string)

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
    deconv_param = {
        'num_output': num_channel_deconv,
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
    from_layer = "conv3_{}{}_Add".format(use_sub_layers[-1],add_strs)
    add_layer = from_layer + "_deconv"
    from_layer= "conv3_{}{}_Add".format(use_sub_layers[-1],add_strs)+pose_string
    net[add_layer] = L.Deconvolution(net[from_layer], **kwargs_deconv)
    if flag_deconvwithrelu:
        bn_name = add_layer + '_bn'
        net[bn_name] = L.BatchNorm(net[add_layer], in_place=True, **bn_kwargs)
        sb_name = add_layer + '_scale'
        net[sb_name] = L.Scale(net[add_layer], in_place=True, **sb_kwargs)
        relu_name = add_layer + '_relu'
        net[relu_name] = L.ReLU(net[add_layer], in_place=True)
    return net

def mPoseNet_COCO_ShuffleVariant_ReconBase_Train(net, data_layer="data",flag_withTea = True,loss_weight=0.2):####
    use_sub_layers = (6, 7)
    num_channels = (144, 288)
    output_channels = (128, 0)
    channel_scale = 4
    num_channel_deconv = 128
    lr = 0.1
    decay = 1.0
    add_strs = "_recon"
    flag_deconvwithrelu = False
    flag_withparamname=True
    pose_string='_pose'
    ############################# NOTE TO CHANGE THE BASE FUNCTION!!!!!!!!!!!!!
    net = ResidualVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                                  output_channels=output_channels,channel_scale=channel_scale, num_channel_deconv=num_channel_deconv,
                                  lr=lr, decay=decay, add_strs=add_strs,flag_deconvwithrelu=flag_deconvwithrelu,flag_withparamname=flag_withparamname)

    # net = ResidualShuffleVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
    #                               output_channels=output_channels,channel_scale=channel_scale, num_channel_deconv=num_channel_deconv,
    #                               lr=lrdecay, decay=lrdecay, add_strs=add_strs,flag_deconvwithrelu=flag_deconvwithrelu)

    recon_layer1 = "conv2_{}{}_Add".format(use_sub_layers[0], add_strs)
    recon_layer2 = "conv3_{}{}_Add".format(use_sub_layers[1], add_strs) + "_deconv"

    strid_convs = [1, 1, 1, 0, 0]
    if flag_withTea:
        ## Teacher 15F
        # net = YoloNetPartCompress(net, from_layer="data", use_bn=True, use_layers=5, use_sub_layers=5,
        #                           strid_conv=strid_convs, final_pool=False, lr=0, decay=0, leaky=True)
        # add_layer = 'conv5_5_upsample'
        # net[add_layer] = L.Reorg(net["conv5_5"], reorg_param=dict(up_down=P.Reorg.UP))

        ## Teach DarkTea8B
        leaky = False
        ChangeNameAndChannel = {"conv4_3": 128, "conv5_1": 512}
        net = YoloNetPart(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5, final_pool=False,
                          leaky=leaky, lr=0, decay=0, ChangeNameAndChannel=ChangeNameAndChannel)

        ### Teacher DarkNetTea4A
        # num_sublayers_tea = [1, 1, 2, 3]
        # num_channels_tea = [512, 256,512, 256,128]
        # alpha = 1
        # net = YoloNetPart_StrideRemove1x1(net, num_sublayers=num_sublayers_tea, num_channels=num_channels_tea,
        #                                   from_layer=data_layer,lr=0, decay=0,alpha=alpha,fix_layer=5,fix_sublayer=1)

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
        ref_layer1 = "conv4_3"
        ref_layer2 = "conv5_5_Upsample"
        net['loss1'] = L.EuclideanLoss(net[recon_layer1], net[ref_layer1], loss_weight=loss_weight)
        net['loss2'] = L.EuclideanLoss(net[recon_layer2], net[ref_layer2], loss_weight=loss_weight)
    return net, recon_layer1, recon_layer2

def mPoseNet_COCO_ShuffleVariant_ReconStage_Train(net, data_layer="data",loss_weight=1.0):
    use_sub_layers = (6, 7)
    num_channels = (128, 256)
    output_channels = (256, 0)
    channel_scale = 4
    num_channel_deconv = 128
    lrdecay = 1
    add_strs = "_recon"
    net = ResidualShuffleVariant_Base_A(net, data_layer=data_layer, use_sub_layers=use_sub_layers, num_channels=num_channels,
                                  output_channels=output_channels,channel_scale=channel_scale, num_channel_deconv=num_channel_deconv,
                                  lr=lrdecay, decay=lrdecay, add_strs=add_strs)

    recon_layer1 = "conv2_{}{}_Add".format(use_sub_layers[0], add_strs)
    recon_layer2 = "conv3_{}{}_Add".format(use_sub_layers[1], add_strs) + "_deconv"
    concat_layer = []
    concat_layer.append(net[recon_layer1])
    concat_layer.append(net[recon_layer2])
    baselayer = "convf" + add_strs
    net[baselayer] = L.Concat(*concat_layer, axis=1)

    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    lrdecay = 1.0
    kernel_size = 3
    flag_output_sigmoid = False
    net = mPose_StageX_Train(net, from_layer=baselayer, stage=1,use_3_layers=use_3_layers, use_1_layers=use_1_layers, short_cut=False,
                                 base_layer=baselayer, lr=lrdecay, decay=lrdecay, num_channels=n_channel,kernel_size=kernel_size,
                             flag_sigmoid=flag_output_sigmoid,flag_hasoutput=False,addstrs=add_strs,flag_hasloss=False)

############################### Teacher
    strid_convs = [1, 1, 1, 0, 0]

    net = YoloNetPartCompress(net, from_layer=data_layer, use_bn=True, use_layers=5, use_sub_layers=5,
                              strid_conv=strid_convs, final_pool=False, lr=0, decay=0, leaky=False)
    add_layer = 'conv5_5_upsample'
    net[add_layer] = L.Reorg(net["conv5_5"], reorg_param=dict(up_down=P.Reorg.UP))
    concat_layer = []
    concat_layer.append(net['conv4_3'])
    concat_layer.append(net['conv5_5_upsample'])
    baselayer = "convf"
    net[baselayer] = L.Concat(*concat_layer, axis=1)

    use_stage = 3
    use_3_layers = 5
    use_1_layers = 0
    n_channel = 64
    kernel_size = 3
    flag_output_sigmoid = False
    for stage in xrange(use_stage):
        if stage == 0:
            from_layer = baselayer
        else:
            from_layer = "concat_stage{}".format(stage)
        outlayer = "concat_stage{}".format(stage + 1)
        if stage == use_stage - 1:
            flag_hasoutput = False
            short_cut = False
        else:
            flag_hasoutput = True
            short_cut = True
        net = mPose_StageX_Train(net, from_layer=from_layer, out_layer=outlayer, stage=stage + 1,mask_vec="vec_mask", mask_heat="heat_mask", \
                                 label_vec="vec_label", label_heat="heat_label",use_3_layers=use_3_layers, use_1_layers=use_1_layers,
                                 short_cut=short_cut,base_layer=baselayer, lr=0, decay=0, num_channels=n_channel,
                                 kernel_size=kernel_size, flag_sigmoid=flag_output_sigmoid,flag_hasoutput=flag_hasoutput,flag_hasloss=False)

    recon_layer1 = "stage1_conv{}_heat".format(use_3_layers-1) + add_strs
    recon_layer2 = "stage1_conv{}_vec".format(use_3_layers - 1) + add_strs
    ref_layer1 = "stage3_conv{}_heat".format(use_3_layers-1)
    ref_layer2 = "stage3_conv{}_vec".format(use_3_layers - 1)
    net['loss1'] = L.EuclideanLoss(net[recon_layer1], net[ref_layer1], loss_weight=loss_weight)
    net['loss2'] = L.EuclideanLoss(net[recon_layer2], net[ref_layer2], loss_weight=loss_weight)
    return net

def mPoseNet_COCO_ShuffleVariant_PoseFromReconBase_Train(net, data_layer="data", label_layer="label", train=True,**pose_test_kwargs):####
    # input
    # input
    if train:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp = \
            L.Slice(net[label_layer], ntop=4, slice_param=dict(slice_point=[34, 52, 86], axis=1))
    else:
        net.vec_mask, net.heat_mask, net.vec_temp, net.heat_temp, net.gt = \
            L.Slice(net[label_layer], ntop=5, slice_param=dict(slice_point=[34, 52, 86, 104], axis=1))
    # label
    net.vec_label = L.Eltwise(net.vec_mask, net.vec_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    net.heat_label = L.Eltwise(net.heat_mask, net.heat_temp, eltwise_param=dict(operation=P.Eltwise.PROD))
    flag_concat = True
    net, ref_layer1, ref_layer2 = mPoseNet_COCO_ShuffleVariant_ReconBase_Train(net, data_layer=data_layer,flag_withTea = False)
    ref_layer1=ref_layer1+pose_string
    if flag_concat:
        feaLayers = []
        feaLayers.append(net[ref_layer1])
        feaLayers.append(net[ref_layer2])
        baselayer = "convf"
        net[baselayer] = L.Concat(*feaLayers, axis=1)
    else:
        baselayer = ref_layer2

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
                                 base_layer=baselayer, lr=0.1, decay=lrdecay, num_channels=n_channel,
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
