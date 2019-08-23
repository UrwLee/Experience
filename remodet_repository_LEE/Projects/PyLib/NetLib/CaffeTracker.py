# -*- coding: utf-8 -*-
import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

def CaffeTrackerNet(net, from_layer="data", label_layer="label"):
    # CaffeNet
    kwargs = {
            'param': [dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),}
    # conv1
    net.conv1 = L.Convolution(net[from_layer], num_output=96, stride=4, kernel_size=11, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    # pool1
    net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # norm1
    net.norm1 = L.LRN(net.pool1, lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))
    # conv2
    net.conv2 = L.Convolution(net.norm1, num_output=256, pad=2, group=2, kernel_size=5, **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    # pool2
    net.pool2 = L.Pooling(net.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # norm2
    net.norm2 = L.LRN(net.pool2, lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))
    # conv3
    net.conv3 = L.Convolution(net.norm2, num_output=384, pad=1, kernel_size=3, **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    # conv4
    #net.conv4 = L.Convolution(net.relu3, num_output=384, pad=1, group=2, kernel_size=3, **kwargs)
    #net.relu4 = L.ReLU(net.conv4, in_place=True)
    # conv5
    #net.conv5 = L.Convolution(net.relu4, num_output=256, pad=1, group=2, kernel_size=3, **kwargs)
    #net.relu5 = L.ReLU(net.conv5, in_place=True)
    # pool5
    net.pool5 = L.Pooling(net.relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    # HalfMerge
    net.convf = L.Halfmerge(net.pool5)
    # FC layers
    fc_kwargs = {
            'param': [dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.005),
            'bias_filler': dict(type='constant', value=1),}
    net.fc6 = L.InnerProduct(net.convf, name="fc6-new1", num_output=4096, **fc_kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.drop6 = L.Dropout(net.relu6, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    net.fc7 = L.InnerProduct(net.drop6, name="fc7-new1", num_output=4096, **fc_kwargs)
    net.relu7 = L.ReLU(net.fc7, in_place=True)
    net.drop7 = L.Dropout(net.relu7, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    net.fc7b = L.InnerProduct(net.drop7, name="fc7-newb1", num_output=4096, **fc_kwargs)
    net.relu7b = L.ReLU(net.fc7b, in_place=True)
    net.drop7b = L.Dropout(net.relu7b, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    fc_kwargs = {
            'param': [dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0),}
    net.fc8 = L.InnerProduct(net.drop7b, name="fc8-shapes1", num_output=4, **fc_kwargs)
    # GT layers
    net.neg = L.Power(net[label_layer],power_param=dict(power=1,scale=-1,shift=0))
    net.neg_flat = L.Flatten(net.neg, name="flatten1")
    # add
    net.out_diff = L.Eltwise(net.fc8,net.neg_flat,name="subtract1")
    # loss
    net.loss = L.Reduction(net.out_diff,name="abssum1",loss_weight=1,reduction_param=dict(operation=2))
    return net
