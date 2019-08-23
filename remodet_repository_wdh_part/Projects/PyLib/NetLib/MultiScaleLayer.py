# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from ConvBNLayer import *

sys.dont_write_bytecode = True

from SpatialTransLayer import *

# 使用多个尺度的检测层合并
# feature_layers = [...]
# Unified-Layer to combine multi-layers
# we have the following methods to concat different scale
# /**注意： 如果卷积层中已包含BN，则特征在拼接前无需Norm处理，否则需要加Norm，类似于SDD的处理方式
# /**yolo卷积层加入了BN，所以直接拼接，ResNet/GoogLeNet/PVANet都具有BN层，可以直接进行拼接
# /**VGG没有使用BN，在拼接前，最好先使用Norm进行标准化处理
# 1. down-sample [type]
#   : reorg by 2 (/2)
#     /** up_down = DOWN, and stride = 2, keep default and DO NOT MODIFIED. **/
#     /** Note: input_size % 2 == 0 should be considered. **/
#   : MaxPool: 3/2 (/2)
#     /** ksize = 3, and stride = 2, pad = 0 **/
#     /** input = 2k+1 -> output = k **/
#     /** input = 2k -> output = k   **/
#   : Conv: 3/2 (/2)
#     /** ksize = 3, and stride = 2, pad = 0 **/
#     /** 2k/2k+1 -> k **/
#     /** 卷积层需要再加入BN层 **/
# 2. up-sample [type]
#   : Deconv ： ksize=4, pad=1, stride = 1, keep output channels the same
#   /**
#   output_size = (input_size - 1)*stride + ksize - 2*pad
#   we make ksize = 4, and pad = 1, stride = 2, then output_size = 2*input_size
#   Note: group must equal to the channels.
#   **/
# 3. nonlinear transform
# /**在平面尺度相同的情况下，可以增加一层1x1-卷积层
# we need to implement these funcs first.
# layers: 特征层列表
# tags： 每一层对应的标签，"Ref"/"Down"/"Up"
# unifiedlayer: 拼接后的特征层名
# dnsampleMethod/upsampleMethod: 上下采样方法
# normalizetions: 每一层是否需要Norm操作
# interLayerChannels: 上下采样结束后，是否增加1x1-conv层
# dnsampleChannels： 下采样，使用卷积方式的输出通道数
# upsampleChannels： 上采样的deconv层的输出通道，一定要与该层的输入channels相同！！！！
# 注意：upsampleChannels参数一般是需要设置的，与上采样的特征层具有相同的通道数
# Down     Ref     Up
# [Norm]  [Norm]  [Norm]
# <dnSp>          <upSp>
# [inter] [inter] [inter]
# <       concat        >
def UnifiedMultiScaleLayers(net,layers=[], tags=[], unifiedlayer="msfMap", \
                            dnsampleMethod=[],upsampleMethod="Deconv", \
                            normalizetions=[], interLayerChannels=[],
                            dnsampleChannels=512, upsampleChannels=512,pad=False):
    assert layers, "layers must be specified."
    assert tags, "tags must be specified."
    num_layers = len(layers)
    assert len(tags) == num_layers, "layers and tags must have the same length."
    for i in range(0,num_layers):
        assert tags[i] in ["Ref","Down","Up"], "Only Ref/Down/Up tags is supported."
    if normalizetions:
        assert len(normalizetions) == num_layers
    if interLayerChannels:
        assert len(interLayerChannels) == num_layers

    feature_layers = []
    # 开始遍历各个层，并创建
    for i in range(0,num_layers):
        from_layer = layers[i]
        tag = tags[i]
        if tag == "Ref":
            if normalizetions and normalizations[i] > 0:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]), \
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name
            if interLayerChannels and interLayerChannels[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNUnitLayer(net, from_layer, inter_name, use_bn=False, use_relu=True, \
                    num_output=interLayerChannels[i], kernel_size=1, pad=0, stride=1)
                from_layer = inter_name
            feature_layers.append(net[from_layer])
        elif tag == "Down":
            down_methods = dnsampleMethod[i]
            for j in range(len(down_methods)):
                down_method = down_methods[j]
                if normalizetions and normalizations[i] > 0:
                    norm_name = "{}_norm".format(from_layer)
                    net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]), \
                        across_spatial=False, channel_shared=False)
                    from_layer = norm_name
                out_layer = "{}_downsample_{}".format(from_layer,j)
                DownSampleLayer(net, from_layer, out_layer, down_sample_method=down_method, num_output=dnsampleChannels,pad=pad)
                from_layer = out_layer
                if interLayerChannels and interLayerChannels[i] > 0:
                    inter_name = "{}_inter".format(from_layer)
                    ConvBNUnitLayer(net, from_layer, inter_name, use_bn=False, use_relu=True, \
                        num_output=interLayerChannels[i], kernel_size=1, pad=0, stride=1)
                    from_layer = inter_name
            feature_layers.append(net[from_layer])
        else:
            if normalizetions and normalizations[i] > 0:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]), \
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name
            out_layer = "{}_upsample".format(from_layer)
            UpSampleLayer(net, from_layer, out_layer, up_sample_method=upsampleMethod, num_output=upsampleChannels)
            from_layer = out_layer
            if interLayerChannels and interLayerChannels[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNUnitLayer(net, from_layer, inter_name, use_bn=False, use_relu=True, \
                    num_output=interLayerChannels[i], kernel_size=1, pad=0, stride=1)
                from_layer = inter_name
            feature_layers.append(net[from_layer])
    # concat
    if len(feature_layers) > 1:
        net[unifiedlayer] = L.Concat(*feature_layers, axis=1)
    else:
        net[unifiedlayer] = feature_layers[0]

    return net
