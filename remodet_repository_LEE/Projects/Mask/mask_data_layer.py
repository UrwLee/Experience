# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True

# train_list = '/home/%s/REMO_DATASET/Layout/Common/train.txt' % USERNAME
# val_list = '/home/%s/REMO_DATASET/Layout/Common/val.txt' % USERNAME
# parts_xml_dir='/home/%s/REMO_DATASET/XML/parts' % USERNAME
# root_dir = '/home/%s/REMO_DATASET' % USERNAME
train_list = '/home/%s/Datasets/WiderFace/Layout/train.txt' % USERNAME
val_list = '/home/%s/Datasets/WiderFace/Layout/val.txt' % USERNAME
root_dir = '/home/%s/Datasets/WiderFace' % USERNAME
save_dir = '/home/%s/Datasets/WiderFace/vis_aug' % USERNAME

def get_unifiedTransParam(train=True):
    if train:
        return{
                'emit_coverage_thre': 0.25,
                # 'kps_min_visible': 1,
                'flip_prob': 0.5,
                'resized_width': 512,
                'resized_height': 288,
                'visualize': True,
                'save_dir': save_dir,
                'dis_param': {
                    'brightness_prob': 0.2,
                    'brightness_delta': 20,
                    'contrast_prob': 0.2,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.2,
                    'hue_delta': 18,
                    'saturation_prob': 0.2,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0,
                    },
                'batch_sampler': [
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': 0.75,
                    'max_scale': 1.0,
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.9,
                    }
                },
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': 0.75,
                    'max_scale': 1.0,
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.7,
                    }
                },
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': 0.75,
                    'max_scale': 1.0,
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.5,
                    }
                },
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': 0.75,
                    'max_scale': 1.0,
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.3,
                    }
                }
            ]
        }
    else:
        return{
                # 'kps_min_visible': 3,
                'resized_width': 512,
                'resized_height': 288,
                'visualize': False,
                'save_dir': save_dir,
        }

def get_unified_data_param(train=True,batchsize=1):
    return {
        'xml_list': train_list if train else val_list,
        'xml_root': root_dir,
        # 'parts_xml_dir': parts_xml_dir,
        'shuffle': True,
        'rand_skip': 500 if train else 1,
        'batch_size': batchsize if train else 1,
        'mean_value': [104,117,123],
        'add_parts': False,
        # 'add_kps':False,
        # 'add_mask':False,
    }

def get_UnifiedDataLayer(net, train=True, batchsize=1):
    unifiedDataParam = get_unified_data_param(train=train,batchsize=batchsize)
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            }
        unifiedTransParam=get_unifiedTransParam(train=True)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            }
        unifiedTransParam=get_unifiedTransParam(train=False)
    net.data, net.label = L.UnifiedData(name="data", unified_data_param=unifiedDataParam, unified_data_transform_param=unifiedTransParam, ntop=2, **kwargs)
    return net
def get_FaceDataLayer(net, train=True, batchsize=1):
    unifiedDataParam = get_unified_data_param(train=train,batchsize=batchsize)
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            }
        unifiedTransParam=get_unifiedTransParam(train=True)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            }
        unifiedTransParam=get_unifiedTransParam(train=False)
    net.data, net.label = L.BBoxData(name="data", unified_data_param=unifiedDataParam, unified_data_transform_param=unifiedTransParam, ntop=2, **kwargs)
    return net
