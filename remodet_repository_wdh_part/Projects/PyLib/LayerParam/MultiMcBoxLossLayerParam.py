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

def get_multimcboxloss_param(loc_loss_type=P.MultiBoxLoss.SMOOTH_L1, \
                           loc_weight=1.0, \
                           conf_weight=1.0, \
                           num_classes=2, \
                           share_location=True, \
                           match_type=P.MultiBoxLoss.PER_PREDICTION, \
                           overlap_threshold=0.5, \
                           use_prior_for_matching=True, \
                           background_label_id=0, \
                           use_difficult_gt=False, \
                           do_neg_mining=True, \
                           neg_pos_ratio=3, \
                           neg_overlap=0.5, \
                           code_type=P.PriorBox.CENTER_SIZE, \
                           encode_variance_in_target=False, \
                           map_object_to_agnostic=False, \
                           name_to_label_file="",\
                           rescore=True,\
                           object_scale=1, \
                           noobject_scale=1, \
                           class_scale=1, \
                           loc_scale=1,):
    return {
        # loc loss type
        'loc_loss_type': loc_loss_type,
        # loc weight
        'loc_weight': loc_weight,
        # body conf weight
        'conf_weight': conf_weight,
        # conf vector length
        'num_classes': num_classes,
        # loc-share
        'share_location': share_location,
        # match type
        'match_type': match_type,
        # match threshold
        'overlap_threshold': overlap_threshold,
        # use prior or real boxes for matching
        'use_prior_for_matching': use_prior_for_matching,
        # bg id
        'background_label_id': background_label_id,
        # use diff boxes for trainning ?
        'use_difficult_gt': use_difficult_gt,
        # hard negative mining
        'do_neg_mining':do_neg_mining,
        # negative samples ratio vers. match samples
        'neg_pos_ratio': neg_pos_ratio,
        # negative max-iou threshold
        'neg_overlap': neg_overlap,
        # boxes encoding type
        'code_type': code_type,
        # variance pre-encoding
        'encode_variance_in_target': encode_variance_in_target,
        # all objects map to 1
        'map_object_to_agnostic': map_object_to_agnostic,
        # loc_class must be 1.
        'loc_class': 1,
        # rsvd
        'name_to_label_file': name_to_label_file,

        'rescore': rescore,
        'object_scale': object_scale,
        'noobject_scale': noobject_scale,
        'class_scale': class_scale,
        'loc_scale':loc_scale,

    }

