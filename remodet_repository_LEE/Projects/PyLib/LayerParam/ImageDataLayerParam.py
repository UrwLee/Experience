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

def get_imageDataLayer_param(root_dir="",src="",batchsize=1):
    return {
        'root_folder': root_dir,
        'source': src,
        'batch_size': batchsize,
        'rand_skip': 1000,
        'shuffle': True,
    }

def get_resize_param(train=True,resize_width=100,resize_height=100):
    return {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [P.Resize.LINEAR,P.Resize.AREA,P.Resize.NEAREST, \
                        P.Resize.CUBIC,P.Resize.LANCZOS4] if train else [P.Resize.LINEAR],
    }

def get_sampler(min_scale=1.0,max_scale=1.0, \
                min_aspect_ratio=1.0,max_aspect_ratio=1.0, \
                min_overlap=0.5,max_trial=1,max_sample=1, \
                constraint_method="min_object_coverage"):
    return {
        'sampler': {'min_scale': min_scale,
                    'max_scale': max_scale,
                    'min_aspect_ratio': min_aspect_ratio,
                    'max_aspect_ratio': max_aspect_ratio,},
        'sample_constraint': {constraint_method: min_overlap},
        'max_trials': max_trial,
        'max_sample': max_sample,
    }

def get_batchsampler(min_scales=[1.0],max_scales=[1.0], \
                min_aspect_ratios=[1.0],max_aspect_ratios=[1.0], \
                min_overlaps=[0.5],max_trials=[1],max_samples=[1], \
                constraint_method="min_object_coverage"):
    batchsamplers=[]
    assert len(min_scales) == len(max_scales)
    assert len(min_scales) == len(min_aspect_ratios)
    assert len(min_scales) == len(max_aspect_ratios)
    assert len(min_scales) == len(min_overlaps)
    assert len(min_scales) == len(max_trials)
    assert len(min_scales) == len(max_samples)
    for min_scale,max_scale,min_aspect_ratio,max_aspect_ratio,min_overlap, \
        max_trial,max_sample in zip(min_scales,max_scales, \
        min_aspect_ratios,max_aspect_ratios,min_overlaps, \
        max_trials,max_samples):
        batchsamplers.append(get_sampler(min_scale=min_scale, \
        max_scale=max_scale,min_aspect_ratio=min_aspect_ratio, \
        max_aspect_ratio=max_aspect_ratio,min_overlap=min_overlap, \
        max_trial=max_trial,max_sample=max_sample,constraint_method=constraint_method))
    return batchsamplers

def get_distort_param(brightness_prob=0,brightness_delta=32, \
                      contrast_prob=0,contrast_lower=0.5,contrast_upper=1.5, \
                      hue_prob=0,hue_delta=18, \
                      saturation_prob=0,saturation_lower=0.5,saturation_upper=1.5, \
                      random_order_prob=0):
    return {
        'brightness_prob': brightness_prob,
        'brightness_delta': brightness_delta,
        'contrast_prob': contrast_prob,
        'contrast_lower': contrast_lower,
        'contrast_upper': contrast_upper,
        'hue_prob': hue_prob,
        'hue_delta': hue_delta,
        'saturation_prob': saturation_prob,
        'saturation_lower': saturation_lower,
        'saturation_upper': saturation_upper,
        'random_order_prob': random_order_prob,
    }

def get_expand_param(expand_prob=0.5,max_expand_ratio=2):
    return {
        'prob': expand_prob,
        'max_expand_ratio': max_expand_ratio,
        }

def get_emit_constraints(emit_type=caffe_pb2.EmitConstraint.CENTER, \
                         emit_overlap=0.5):
    return {
        'emit_type': emit_type,
        'emit_overlap': emit_overlap,
             }

def get_datatransformer_param(train=True, \
                              mean_values=[104, 117, 123], \
                              boxsize_threshold=0.0001, \
                              resize_width=100, resize_height=100, \
                              bs_min_scales=[1.0], bs_max_scales=[1.0], \
                              bs_min_aspect_ratios=[1.0], bs_max_aspect_ratios=[1.0], \
                              bs_min_overlaps=[0.5], bs_max_trials=[1], bs_max_samples=[1], \
                              bs_constraint_method="min_object_coverage", \
                              dis_brightness_prob=0,dis_brightness_delta=32, \
                              dis_contrast_prob=0,dis_contrast_lower=0.5,
                              dis_contrast_upper=1.5, dis_hue_prob=0,dis_hue_delta=18, \
                              dis_saturation_prob=0,dis_saturation_lower=0.5,
                              dis_saturation_upper=1.5, dis_random_order_prob=0, \
                              ex_expand_prob=0, ex_max_expand_ratio=2, \
                              emit_type=caffe_pb2.EmitConstraint.CENTER, \
                              emit_overlap=0.5):
    if train:
        return {
                'mirror': True,
                'mean_value': mean_values,
                'boxsize_threshold': boxsize_threshold,
                'resize_param': get_resize_param(train=True, \
                                                 resize_width=resize_width, \
                                                 resize_height=resize_height),
                'batch_sampler': get_batchsampler(min_scales=bs_min_scales, \
                                                  max_scales=bs_max_scales, \
                                                  min_aspect_ratios=bs_min_aspect_ratios, \
                                                  max_aspect_ratios=bs_max_aspect_ratios, \
                                                  min_overlaps=bs_min_overlaps, \
                                                  max_trials=bs_max_trials, \
                                                  max_samples=bs_max_samples, \
                                                  constraint_method=bs_constraint_method),
                'distort_param': get_distort_param(brightness_prob=dis_brightness_prob, \
                                                   brightness_delta=dis_brightness_delta, \
                                                   contrast_prob=dis_contrast_prob, \
                                                   contrast_lower=dis_contrast_lower, \
                                                   contrast_upper=dis_contrast_upper, \
                                                   hue_prob=dis_hue_prob, \
                                                   hue_delta=dis_hue_delta, \
                                                   saturation_prob=dis_saturation_prob, \
                                                   saturation_lower=dis_saturation_lower, \
                                                   saturation_upper=dis_saturation_upper, \
                                                   random_order_prob=dis_random_order_prob),
                'expand_param': get_expand_param(expand_prob=ex_expand_prob, \
                                                 max_expand_ratio=ex_max_expand_ratio),
                'emit_constraint': get_emit_constraints(emit_type=emit_type, \
                                                        emit_overlap=emit_overlap),
        }
    else:
        return {
            'mirror': False,
            'mean_value': mean_values,
            'boxsize_threshold': boxsize_threshold,
            'resize_param': get_resize_param(train=False, \
                                             resize_width=resize_width, \
                                             resize_height=resize_height),
        }
