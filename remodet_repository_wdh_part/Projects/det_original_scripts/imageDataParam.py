# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True

def get_imageDataParam(root_folder="", \
                       train_source="", \
                       train_batchsize=1, \
                       test_source="", \
                       test_batchsize=1):
    return {
        'mean_values': [104,117,123],
        'boxsize_threshold': 0.001,
        # batch-sampler
        'bs_min_scales': [0.5,0.5,0.5,0.5],
        'bs_max_scales': [1.0,1.0,1.0,1.0],
        'bs_min_aspect_ratios': [0.5,0.5,0.5,0.5],
        'bs_max_aspect_ratios': [2.0,2.0,2.0,2.0],
        'bs_min_overlaps': [0.3,0.5,0.7,0.9],
        'bs_max_trials': [50,50,50,50],
        'bs_max_samples': [1,1,1,1],
        'bs_constraint_method': 'min_jaccard_overlap',
        # min_jaccard_overlap / min_object_coverage
        # distortion
        'dis_brightness_prob': 0.2,
        'dis_brightness_delta': 32,
        'dis_contrast_prob': 0.2,
        'dis_contrast_lower': 0.5,
        'dis_contrast_upper': 1.5,
        'dis_hue_prob': 0.2,
        'dis_hue_delta': 18,
        'dis_saturation_prob': 0.2,
        'dis_saturation_lower': 0.5,
        'dis_saturation_upper': 1.5,
        'dis_random_order_prob': 0,
        # expansion
        'ex_expand_prob': 0,
        'ex_max_expand_ratio': 3,
        # emit type
        'emit_type': caffe_pb2.EmitConstraint.CENTER,
        'emit_overlap': 0.3,
        # input directory
        'root_dir': root_folder,
        'train_source': train_source,
        'train_batchsize': train_batchsize,
        'test_source': test_source,
        'test_batchsize': test_batchsize,
    }
