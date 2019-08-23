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

# Layer自带[0.95,0.95]的box
# 使用boxsize * aspect_ratio来进行定义
def get_mcboxloss_param(num_classes=1, \
                       overlap_threshold=0.6, \
                       use_prior_for_matching=True, \
                       use_prior_for_init=False, \
                       use_difficult_gt=True, \
                       rescore=True, \
                       clip=True, \
                       iters=0, \
                       iter_using_bgboxes=10000, \
                       background_box_loc_scale=0.01, \
                       object_scale=5, \
                       noobject_scale=1, \
                       class_scale=1, \
                       loc_scale=1, \
                       boxsize=[], \
                       aspect_ratio=[], \
                       pwidth=[], \
                       pheight=[], \
                       background_label_id=0,\
                       code_loc_type=P.McBoxLoss.SSD):
    if boxsize:
        return {
            'num_classes': num_classes,
            'overlap_threshold': overlap_threshold,
            'use_prior_for_matching': use_prior_for_matching,
            'use_prior_for_init': use_prior_for_init,
            'use_difficult_gt': use_difficult_gt,
            'rescore': rescore,
            'clip': clip,
            'iters': iters,
            'iter_using_bgboxes': iter_using_bgboxes,
            'background_box_loc_scale': background_box_loc_scale,
            'object_scale': object_scale,
            'noobject_scale': noobject_scale,
            'class_scale': class_scale,
            'loc_scale':loc_scale,
            'boxsize': boxsize,
            'aspect_ratio': aspect_ratio,
            'background_label_id': background_label_id,
            'code_loc_type':code_loc_type,
        }
    else:
        return {
            'num_classes': num_classes,
            'overlap_threshold': overlap_threshold,
            'use_prior_for_matching': use_prior_for_matching,
            'use_prior_for_init': use_prior_for_init,
            'use_difficult_gt': use_difficult_gt,
            'rescore': rescore,
            'clip': clip,
            'iters': iters,
            'iter_using_bgboxes': iter_using_bgboxes,
            'background_box_loc_scale': background_box_loc_scale,
            'object_scale': object_scale,
            'noobject_scale': noobject_scale,
            'class_scale': class_scale,
            'loc_scale':loc_scale,
            'pwidth': pwidth,
            'pheight': pheight,
            'background_label_id': background_label_id,
            'code_loc_type':code_loc_type,
        }
