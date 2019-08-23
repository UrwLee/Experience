# -*- coding: utf-8 -*-
import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

sys.path.append('../')

from PyLib.LayerParam.ImageDataLayerParam import *

def ImageDataLayer(train=True, output_label=True, \
                  resize_width=300, resize_height=300, **imgparam):
    transform_param = get_datatransformer_param( \
          train=train, \
          mean_values=imgparam.get("mean_values",[104,117,123]), \
          boxsize_threshold=imgparam.get("boxsize_threshold",0.0001), \
          resize_width=resize_width, \
          resize_height=resize_height, \
          bs_min_scales=imgparam.get("bs_min_scales",[1.0]), \
          bs_max_scales=imgparam.get("bs_max_scales",[1.0]), \
          bs_min_aspect_ratios=imgparam.get("bs_min_aspect_ratios",[1.0]), \
          bs_max_aspect_ratios=imgparam.get("bs_max_aspect_ratios",[1.0]), \
          bs_min_overlaps=imgparam.get("bs_min_overlaps",[0.5]), \
          bs_max_trials=imgparam.get("bs_max_trials",[1]), \
          bs_max_samples=imgparam.get("bs_max_samples",[1]), \
          bs_constraint_method=imgparam.get("bs_constraint_method","min_object_coverage"), \
          dis_brightness_prob=imgparam.get("dis_brightness_prob",0), \
          dis_brightness_delta=imgparam.get("dis_brightness_delta",32), \
          dis_contrast_prob=imgparam.get("dis_contrast_prob",0), \
          dis_contrast_lower=imgparam.get("dis_contrast_lower",0.5), \
          dis_contrast_upper=imgparam.get("dis_contrast_upper",1.5), \
          dis_hue_prob=imgparam.get("dis_hue_prob",0), \
          dis_hue_delta=imgparam.get("dis_hue_delta",18), \
          dis_saturation_prob=imgparam.get("dis_saturation_prob",0), \
          dis_saturation_lower=imgparam.get("dis_saturation_lower",0.5), \
          dis_saturation_upper=imgparam.get("dis_saturation_upper",1.5), \
          dis_random_order_prob=imgparam.get("dis_random_order_prob",0), \
          ex_expand_prob=imgparam.get("ex_expand_prob",0), \
          ex_max_expand_ratio=imgparam.get("ex_max_expand_ratio",2), \
          emit_type=imgparam.get("emit_type",caffe_pb2.EmitConstraint.CENTER), \
          emit_overlap=imgparam.get("emit_overlap",0.5))
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
        imageDataLayer_param = get_imageDataLayer_param( \
                root_dir=imgparam.get("root_dir",""), \
                src=imgparam.get("train_source",""), \
                batchsize=imgparam.get("train_batchsize",1))
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
        imageDataLayer_param = get_imageDataLayer_param( \
                root_dir=imgparam.get("root_dir",""), \
                src=imgparam.get("test_source",""), \
                batchsize=imgparam.get("test_batchsize",1))
    if output_label:
        data, label = L.ImageData(name="data",
                                  image_data_param=imageDataLayer_param,
                                  ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.ImageData(name="data",
                           image_data_param=imageDataLayer_param,
                           ntop=1, **kwargs)
        return data
