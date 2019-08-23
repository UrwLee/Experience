# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

from ConvBNLayer import *

def DeconvLayerForUpSample(net, from_layer, out_layer, num_output, \
                           kernel_size=4, pad=1, stride=2, \
                           filler_type="bilinear"):
    kwargs = {
    'param': [dict(lr_mult=0, decay_mult=0)],
    'convolution_param': {
        'num_output': num_output,
        'kernel_size': kernel_size,
        'pad': pad,
        'stride': stride,
        'weight_filler': dict(type=filler_type),
        'bias_term': False,
        'group': num_output,
        }
    }
    net[out_layer] = L.Deconvolution(net[from_layer], **kwargs)
    return net

def DownSampleLayer(net, from_layer, out_layer, down_sample_method="MaxPool", num_output=0,pad=False):
    assert from_layer in net.keys(), "From_layer is not found in net."
    if down_sample_method == "Reorg":
        # use the default value: stride = 2 and DOWN-mode
        net[out_layer] = L.Reorg(net[from_layer])
    elif down_sample_method == "MaxPool":
        # ksize = 3, pad = 0 and stride = 2
        if pad:
            net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2,pad=1)
        else:
            net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=3, stride=2,pad=0)
    elif down_sample_method == "Conv":
        # use conv with stride = 2
        # use BN Layer
        ConvBNUnitLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, num_output=num_output,
            kernel_size=1, pad=0, stride=2)
    else:
        raise ValueError("the down_sample_method is invalid.")
    return net

def UpSampleLayer(net, from_layer, out_layer, up_sample_method="Deconv", num_output=1):
    """
    num_output should be equal to the channels of the from_layer.
    """
    assert from_layer in net.keys(), "From_layer is not found in net."
    if up_sample_method == "Deconv":
        return DeconvLayerForUpSample(net,from_layer,out_layer,num_output)
    elif up_sample_method == "Reorg":
       net[out_layer] = L.Reorg(net[from_layer], reorg_param=dict(up_down=P.Reorg.UP))
    else:
        raise ValueError("Only deconv upsample method is supported.")
