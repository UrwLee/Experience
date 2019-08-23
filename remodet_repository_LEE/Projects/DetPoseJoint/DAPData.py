# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
import os
sys.dont_write_bytecode = True
# USERNAME='zhangming'
# '/home/%s/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt' % USERNAME,
# '/home/%s/Datasets/RemoCoco' % USERNAME,
# '/home/%s/Datasets/RemoCoco/Layout/AllParts/train_ImgAreaAbove10000.txt' % USERNAME,
#'/home/%s/Datasets/RemoCoco' % USERNAME,
if USERNAME == "zhangming":

    train_list = ['/home/zhangming/Datasets/RemoCoco/Layout/AllParts/train.txt','/home/zhangming/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt','/home/zhangming/Datasets/AIC_Data/trainval_V1New.txt','/home/zhangming/Datasets/OtherBackGround_Images/background_list.txt']
    root_dir_train = ['/home/zhangming/Datasets/RemoCoco','/home/zhangming/Datasets/RemoCoco','/home/zhangming/Datasets/AIC_Data','/home/zhangming/Datasets/OtherBackGround_Images']
#'/home/%s/Datasets/RemoCoco/Layout/AllParts/val_ImgAreaAbove10000.txt' % USERNAME,
#'/home/%s/Datasets/RemoCoco' % USERNAME,

    val_list = '/home/zhangming/Datasets/AIC_Data/val_PersonWithFace.txt'
    root_dir_val = '/home/zhangming/Datasets/AIC_Data'
else:
    train_list = ['/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/trainval_V1New.txt',]
    root_dir_train = ["/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/AIC_SRC",]
    val_list = '/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/val_PersonWithFace.txt'
    root_dir_val = '/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/AIC_SRC'

# train_list = '/home/%s/Datasets/RemoCoco/Layout/AllParts/train.txt' % USERNAME
# root_dir_train = '/home/%s/Datasets/RemoCoco' % USERNAME
# val_list = '/home/%s/Datasets/RemoCoco/Layout/AllParts/val.txt' % USERNAME
# root_dir_val = '/home/%s/Datasets/RemoCoco' % USERNAME



save_dir = '/home/zhangming/Datasets/RemoCoco/vis_aug'

##det
flag_noperson = True
flag_sample_sixteennine = True
resized_width = 512
resized_height = 288
lr_basenet = 0.1
use_trainval = False
##pose
use_distorted = True
if os.path.exists('/home/%s/PoseDatasets'%USERNAME):
    root_dir = '/home/%s/PoseDatasets/'%USERNAME
else:
    root_dir = '/home/zhangming/PoseDatasets/'
crop_using_resize = False
scale_max = 1.1
pose_stride = 8
def get_unifiedTransParam(train=True):
    if train:
        return{
                'emit_coverage_thre_multiple': [1.0, 0.75, 0.5,0.25], #
                'sample_sixteennine':flag_sample_sixteennine,
                'emit_coverage_thre':0.25,
                'emit_area_check':[0.02, 0.1,0.3, 1.0],    # default is [1.0,]
                'flip_prob': 0.5,
                'resized_width': resized_width,
                'resized_height': resized_height,
                'visualize': False,
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
                    'min_aspect_ratio':0.5, # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio':2.0, # is not used when flag_sample_sixteennine is true
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
                    'min_aspect_ratio': 0.5,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 2.0,  # is not used when flag_sample_sixteennine is true
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
                    'min_aspect_ratio': 0.5,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 2.0,  # is not used when flag_sample_sixteennine is true
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
                    'min_aspect_ratio': 0.5,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 2.0,  # is not used when flag_sample_sixteennine is true
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.3,
                    }
                }
            ]
        }
    else:
        return {
                'sample_sixteennine': flag_sample_sixteennine,
                'resized_width': resized_width,
                'resized_height': resized_height,
                'visualize': False,
                'save_dir': save_dir,
        }

def get_unified_data_param(train=True,batchsize=1):
    if train:
        if isinstance(train_list,list):
            return {
                'xml_list_multiple': train_list,
                'xml_root_multiple': root_dir_train,
                'shuffle': True,
                'rand_skip': 500,
                'batch_size': batchsize,
                'mean_value': [104, 117, 123],
                'add_parts': True,
            }
        else:
            return {
                'xml_list': train_list,
                'xml_root': root_dir_train,
                'shuffle': True,
                'rand_skip': 500,
                'batch_size': batchsize,
                'mean_value': [104, 117, 123],
                'add_parts': True,
            }
    else:
        if isinstance(val_list,list):
            return {
                'xml_list_multiple': val_list,
                'xml_root_multiple': root_dir_val,
                'shuffle': True,
                'rand_skip': 1,
                'batch_size': 1,
                'mean_value': [104, 117, 123],
                'add_parts': True,
            }
        else:
            return {
                'xml_list': val_list,
                'xml_root': root_dir_val,
                'shuffle': True,
                'rand_skip': 1,
                'batch_size': 1,
                'mean_value': [104, 117, 123],
                'add_parts': True,
            }

def get_source_file(train=True):
    if train:
        if use_trainval:
            return '%sLayout/trainval.txt' % root_dir
        else:
            return '%sLayout/train85.txt' % root_dir
    else:
        return '%sLayout/val15.txt' % root_dir
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
def get_detDAPDataLayer(net, train=True, batchsize=1):
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
def get_DAPDataLayer(net,train=True,batchsize_det=1,batchsize_pose=1):
    if train:
        net=get_detDAPDataLayer(net, train=True, batchsize=batchsize_det)
        net=get_poseDataLayer(net, train=True, batch_size=batchsize_pose)
    else:
        net=get_detDAPDataLayer(net, train=False, batchsize=1)
        net=get_poseDataLayer(net, train=False, batch_size=1)
    return net