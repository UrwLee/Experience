# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
import os
sys.dont_write_bytecode = True

# if use trainval
use_trainval = False
# if use color_distortion
use_distorted = True
if os.path.exists('/home/%s/DataDisk/PoseDatasets'%USERNAME):
    root_dir = '/home/%s/DataDisk/PoseDatasets/'%USERNAME
else:
    root_dir = '/home/%s/PoseDatasets/' % USERNAME
crop_using_resize = False
scale_max = 1.1
pose_stride = 8
def get_source_file(train=True):
    if train:
        if use_trainval:
            return '%sLayout/trainval.txt' % root_dir
        else:
            return '%sLayout/train85.txt' % root_dir
    else:
        return '%sLayout/val15.txt' % root_dir

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
        'xml_root': root_dir,
        'shuffle': shuffle,
        'rand_skip': rand_skip,
        'batch_size': batch_size_real,
        'out_kps': out_kps,
    }

def get_poseDataTransParam(train = True):
    if use_distorted and train:
        return {
            'mirror': True if train else False,
            'stride': pose_stride,
            'max_rotate_degree': 40,
            'visualize': False,
            'crop_using_resize': crop_using_resize,
            'crop_size_x': 384,
            'crop_size_y': 384,
            'scale_prob': 1.0,
            'scale_min': 0.5,
            'scale_max': scale_max,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '%stemp/visual/' % root_dir,
            'root_dir': root_dir,
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
            'stride': pose_stride,
            'max_rotate_degree': 40,
            'visualize': False,
            'crop_using_resize': crop_using_resize,
            'crop_size_x': 368,
            'crop_size_y': 368,
            'scale_prob': 1.0,
            'scale_min': 0.5,
            'scale_max': scale_max,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '%stemp/visual/' % root_dir,
            'root_dir': root_dir,
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
    net.data_pose, net.label_pose = L.PoseData(name="data_pose", pose_data_param=posedata_kwargs, \
                                     ntop=2, **kwargs)

    return net
