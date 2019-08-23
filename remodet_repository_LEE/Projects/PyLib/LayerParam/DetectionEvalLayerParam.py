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

def get_detection_eval_param(num_classes=2, \
                             background_label_id=0, \
                             evaluate_difficult_gt=False, \
                             boxsize_threshold=[0,0.01,0.05,0.1,0.15,0.2,0.25], \
                             iou_threshold=[0.9,0.75,0.5], \
                             name_size_file=""):
    return {
        'num_classes': num_classes,
        'background_label_id': background_label_id,
        'evaluate_difficult_gt': evaluate_difficult_gt,
        'boxsize_threshold': boxsize_threshold,
        'iou_threshold': iou_threshold,
        'name_size_file': "",
    }
