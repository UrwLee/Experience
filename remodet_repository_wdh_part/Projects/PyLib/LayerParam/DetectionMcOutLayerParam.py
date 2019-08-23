# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import params as P
from caffe import layers as L
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import os
import sys

sys.dont_write_bytecode = True

# YOLO use:
# 0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741
# [0.0618, w/h~1]    [0.195, w/h~1]     [0.424, w/h~0.5]     [0.527, w/h~2]   [0.944, w/h~1]
# 注意，Layer自带[0.95,0.95]的box，只需定义较小的box即可。
def get_detection_mc_out_param(num_classes=1, \
                            conf_threshold=0.5, \
                            nms_threshold=0.45, \
                            clip=True, \
                            boxsize_threshold=0.001, \
                            top_k=100, \
                            boxsize=[0.5, 0.25, 0.12, 0.06], \
                            aspect_ratio=[1, 0.5, 2], \
                            pwidth=[], \
                            pheight=[], \
                            visualize=False, \
                            visual_conf_threshold=0.5, \
                            visual_size_threshold=0, \
                            display_maxsize=1000, \
                            line_width=4, \
                            color=[[0,255,0],], \
                            code_loc_type=P.McBoxLoss.SSD):
    if boxsize:
        return {
            'num_classes': num_classes,
            'conf_threshold': conf_threshold,
            'nms_threshold': nms_threshold,
            'clip': clip,
            'boxsize_threshold': boxsize_threshold,
            'top_k': top_k,
            'boxsize': boxsize,
            'aspect_ratio': aspect_ratio,
            'visual_param': {
                    'visualize': visualize,
                    'conf_threshold':visual_conf_threshold,
                    'size_threshold':visual_size_threshold,
                    'display_maxsize': display_maxsize,
                    'line_width': line_width,
                    'color_param': {
                        'rgb': {
                            'val': color[0],
                        }
                    }
            },
            'code_loc_type':code_loc_type,
        }
    else:
        return {
            'num_classes': num_classes,
            'conf_threshold': conf_threshold,
            'nms_threshold': nms_threshold,
            'clip': clip,
            'boxsize_threshold': boxsize_threshold,
            'top_k': top_k,
            'pwidth': pwidth,
            'pheight': pheight,
            'visual_param': {
                    'visualize': visualize,
                    'conf_threshold':visual_conf_threshold,
                    'size_threshold':visual_size_threshold,
                    'display_maxsize': display_maxsize,
                    'line_width': line_width,
                    'color_param': {
                        'rgb': {
                            'val': color[0],
                        }
                    }
            },
            'code_loc_type':code_loc_type,
        }
