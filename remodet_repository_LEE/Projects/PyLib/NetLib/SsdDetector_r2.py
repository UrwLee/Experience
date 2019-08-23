# -*- coding: utf-8 -*-
import os
import sys
import caffe
import math

sys.dont_write_bytecode = True

sys.path.append('../')

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from DetectHeaderLayer import *
from ConvBNLayer import *

from MultiScaleLayer import *

def SsdDetectorHeaders(net, \
                      boxsizes=[], \
                      min_ratio=15, max_ratio=90, \
                      net_width=300, net_height=300, \
                      data_layer="data", num_classes=2, \
                      from_layers=[], \
                      use_batchnorm=True, \
                      prior_variance = [0.1,0.1,0.2,0.2], \
                      normalizations=[], \
                      aspect_ratios=[], \
                      flip=True, clip=False, \
                      inter_layer_channels=[], \
                      kernel_size=3,pad=1,dropout=False,loc_postfix=''):
    assert from_layers, "Feature layers must be provided."
    pro_widths=[]
    pro_heights=[]
    if loc_postfix=='parts':
      #[0.036,0.05][0.05,0.05]
      pro_widths = [[0.086,0.086],[0.155,0.157],[0.278,0.491]]
      pro_heights = [[0.066,0.12],[0.23,0.125],[0.333,0.581]]
    else:      
      # i is layer
      #
      #
      # pro_widths = [[0.086,0.086],[0.155,0.157],[0.278,0.491]]
      # pro_heights = [[0.066,0.12],[0.23,0.125],[0.333,0.581]]
      # pro_widths = [[0.03125,0.0625,0.125],[0.25],[0.5]]
      # pro_heights = [[0.0555,0.11,0.222],[0.44],[0.88]]
      # pro_widths = [[0.02,0.048,0.064,0.093,0.117,0.125],[0.153,0.181,0.187,0.259,0.290,0.293],[0.394,0.432,0.557,0.599,0.672,0.895]]
      # pro_heights = [[0.045,0.01,0.21,0.351,0.128,0.545],[0.233,0.386,0.698,0.507,0.786,0.283],[0.546,0.828,0.323,0.888,0.598,0.89]]
      pro_widths = []
      pro_heights = []
      for i in range(len(boxsizes)):
        #  boxsizes_per_layer -> []
        boxsizes_per_layer = boxsizes[i]
        pro_widths_per_layer = []
        pro_heights_per_layer = []
        # scan all box size
        for j in range(len(boxsizes_per_layer)):
          #   boxsize
          boxsize = boxsizes_per_layer[j]
          # print aspect_ratios
          # aspect_ratio = aspect_ratios[0]
          aspect_ratio = aspect_ratios[i][j]
          # if not len(aspect_ratios) == 1:
          #   aspect_ratio = aspect_ratios[i][j]
          for each_aspect_ratio in aspect_ratio:
              w = boxsize * math.sqrt(each_aspect_ratio)
              h = boxsize / math.sqrt(each_aspect_ratio)
              w = min(w,1.0)
              h = min(h,1.0)
              pro_widths_per_layer.append(w)
              pro_heights_per_layer.append(h)
        pro_widths.append(pro_widths_per_layer)
        pro_heights.append(pro_heights_per_layer)
      

    mbox_layers = MultiLayersDetectorHeader(net, data_layer=data_layer, num_classes=num_classes, \
                                            from_layers=from_layers, \
                                            normalizations=normalizations, \
                                            use_batchnorm=use_batchnorm, \
                                            prior_variance = prior_variance, \
                                            pro_widths=pro_widths, pro_heights=pro_heights, \
                                            aspect_ratios=aspect_ratios, \
                                            flip=flip, clip=clip, \
                                            inter_layer_channels=inter_layer_channels, \
                                            kernel_size=kernel_size, pad=pad,loc_postfix=loc_postfix)
    return mbox_layers
