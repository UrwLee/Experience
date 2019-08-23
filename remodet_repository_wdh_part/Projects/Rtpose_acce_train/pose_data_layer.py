# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True

# if use trainval
use_trainval = True
# if use color_distortion
use_distorted = False

def get_source_file(train=True):
    if train:
        if use_trainval:
            return '/home/%s/Datasets/Layout_pose/trainval.txt' % USERNAME
        else:
            return '/home/%s/Datasets/Layout_pose/train.txt' % USERNAME
    else:
        return '/home/%s/Datasets/Layout_pose/val.txt' % USERNAME

def get_poseDataParam(train=True, batch_size=1):
    if train:
        xml_list = get_source_file(train=True)
        shuffle = True
        rand_skip = 1000
        batch_size_real = batch_size
        out_kps = False
    else:
        xml_list = get_source_file(train=False)
        shuffle = False
        rand_skip = 0
        batch_size_real = 1
        out_kps = True
    return {
        'xml_list': xml_list,
        'xml_root': '/home/%s/Datasets/' % USERNAME,
        'shuffle': shuffle,
        'rand_skip': rand_skip,
        'batch_size': batch_size_real,
        'out_kps': out_kps,
    }

def get_poseDataTransParam(train = True):
    if use_distorted:
        return {
            'mirror': True if train else False,
            'stride': 8,
            'max_rotate_degree': 40,
            'visualize': False,
            'crop_size_x': 368,
            'crop_size_y': 368,
            'scale_prob': 1,
            'scale_min': 0.5,
            'scale_max': 1.1,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '/home/%s/Datasets/temp/visual/' % USERNAME,
            'root_dir': '/home/%s/Datasets/' % USERNAME,
            'resized_width': 416,
            'resized_height': 416,
            'dis_param': {
                # brightness
                'brightness_prob': 0.5,
                'brightness_delta': 18,
                # contrast
                'contrast_prob': 0.5,
                'contrast_lower': 0.7,
                'contrast_upper': 1.3,
                # hue
                'hue_prob': 0.5,
                'hue_delta': 18,
                # sat
                'saturation_prob': 0.5,
                'saturation_lower': 0.7,
                'saturation_upper': 1.3,
                # random swap the channels
                'random_order_prob': 0,
            },
            # if True -> (x-128)/256 or False -> x - mean_value[c], default is True
            'normalize': False,
            'mean_value': [104,117,123],
        }
    else:
        return {
            'mirror': True if train else False,
            'stride': 8,
            'max_rotate_degree': 40,
            'visualize': False,
            'crop_size_x': 368,
            'crop_size_y': 368,
            'scale_prob': 1,
            'scale_min': 0.5,
            'scale_max': 1.1,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '/home/%s/Datasets/temp/visual/' % USERNAME,
            'root_dir': '/home/%s/Datasets/' % USERNAME,
            'resized_width': 416,
            'resized_height': 416,
            # if True -> (x-128)/256 or False -> x - mean_value[c], default is True
            'normalize': False,
            'mean_value': [104,117,123],
        }

def get_poseDataLayer(net, train=True, batch_size=1):
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            'pose_data_transform_param': get_poseDataTransParam(train=train),
            }
        posedata_kwargs = get_poseDataParam(train=train, batch_size=batch_size)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'pose_data_transform_param': get_poseDataTransParam(train=train),
            }
        posedata_kwargs = get_poseDataParam(train=train, batch_size=batch_size)
    net.data, net.label = L.PoseData(name="data", pose_data_param=posedata_kwargs, \
                                     ntop=2, **kwargs)
    return net
