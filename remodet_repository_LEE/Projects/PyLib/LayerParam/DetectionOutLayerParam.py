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

def get_detection_out_param(num_classes=2, \
                            share_location=True, \
                            background_label_id=0, \
                            code_type=P.PriorBox.CENTER_SIZE, \
                            variance_encoded_in_target=False, \
                            conf_threshold=0.01, \
                            nms_threshold=0.45, \
                            boxsize_threshold=0.001, \
                            top_k=200, \
                            visualize=False, \
                            visual_conf_threshold=0.5, \
                            visual_size_threshold=0, \
                            display_maxsize=1000, \
                            line_width=4, \
                            color=[[0,255,0],]):
    return {
        # 0-bg and 1-body
        'num_classes': num_classes,
        # all classes share the loc-boxes
        'share_location': share_location,
        # bg->0
        'background_label_id': background_label_id,
        # box-encoding: CENTER
        'code_type': code_type,
        # variance pre-encoding
        'variance_encoded_in_target': variance_encoded_in_target,
        # NMS params: conf and nms-iou
        'conf_threshold': conf_threshold,
        'nms_threshold': nms_threshold,
        # size-filtering
        'size_threshold': boxsize_threshold,
        # keep maxmium objects
        'top_k': top_k,
        # visualize param
        'visual_param': {
                # enable or disable the visualize function
                'visualize': visualize,
                'conf_threshold':visual_conf_threshold,
                'size_threshold':visual_size_threshold,
                # image display size
                'display_maxsize': display_maxsize,
                # rectangle line width
                'line_width': line_width,
                # rectangle color
                'color_param': {
                    'rgb': {
                        'val': color[0],
                    }
                }
        }
    }
