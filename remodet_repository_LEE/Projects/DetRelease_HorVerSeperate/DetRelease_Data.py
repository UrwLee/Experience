# -*- coding: utf-8 -*-
from username import USERNAME
import sys
# sys.path.insert(0, "/home/%s/work/minihand/remodet_repository/python")
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from DetRelease_General import *
sys.dont_write_bytecode = True

########################################################
###############General Informaion#######################
########################################################


save_path = '/home/%s/tmp'%USERNAME
#######Body and Part Detection Data######
if train_net_id == 0:
    if data_str_body == "OnlyAIC":
        train_list_bdpd169 = ['/home/%s/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt'% USERNAME]
        root_dir_train_bdpd = ['/home/%s/Datasets/AIC_Data' % USERNAME]
    else:
        train_list_bdpd169 = ['/home/%s/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt'% USERNAME,
                           '/home/zhangming/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt',
                           '/home/zhangming/Datasets/OtherBackGround_Images/background_list.txt',
                           '/home/zhangming/Datasets/Google_BodyDet_20180817/Layout/trainval_Google_BodyDet_20180817.txt']
        root_dir_train_bdpd = ['/home/%s/SSD_DATA/Datasets/AIC_Data' % USERNAME,
                           '/home/zhangming/SSD_DATA/Datasets/RemoCoco',
                           '/home/zhangming/SSD_DATA/Datasets/OtherBackGround_Images',
                           '/home/zhangming/SSD_DATA/Datasets']
    train_list_bdpd916 = ['/home/%s/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt'% USERNAME,
                       '/home/zhangming/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt',
                       '/home/zhangming/Datasets/OtherBackGround_Images/background_list.txt',
                       '/home/zhangming/Datasets/Google_BodyDet_20180817/Layout/trainval_Google_BodyDet_20180817_HWGE2_3.txt']
else:
    train_list_bdpd169 = ["/home/%s/Datasets/AIC_REMOCapture/trainval_AIC_remocap2018053008070827.txt"% USERNAME,
   "/home/%s/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt"% USERNAME,
    "/home/%s/Datasets/OtherBackGround_Images/background_list.txt"% USERNAME]

    train_list_bdpd916 = train_list_bdpd169

    root_dir_train_bdpd = ["/home/%s/Datasets/AIC_REMOCapture" % USERNAME,
                           "/home/%s/Datasets/RemoCoco" % USERNAME,
                           "/home/%s/Datasets/OtherBackGround_Images" % USERNAME]
val_list_bdpd = '/home/%s/Datasets/AIC_Data/Layout/val_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt' % USERNAME
root_dir_val_bdpd = '/home/%s/Datasets/AIC_Data' % USERNAME

#######Part and Minihand Detection Data######
train_list_minihand = ["/home/%s/Datasets/AIC_REMOCapture/trainval_AIC_remocap2018053008070827.txt"% USERNAME,
    "/home/%s/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt"% USERNAME,
    "/home/%s/Datasets/OtherBackGround_Images/background_list.txt"% USERNAME]
if flag_miniresizeddata:
    train_root_minihand = ["/home/%s/Datasets/AIC_REMOCapture_resized"% USERNAME,
         "/home/%s/Datasets/RemoCoco_resized"% USERNAME,
         "/home/%s/Datasets/OtherBackGround_Images_resized"% USERNAME]
else:
    train_root_minihand = ["/home/%s/Datasets/AIC_REMOCapture" % USERNAME,
                           "/home/%s/Datasets/RemoCoco" % USERNAME,
                           "/home/%s/Datasets/OtherBackGround_Images" % USERNAME]
val_list_minihand = '/home/%s/SSD_DATA/Datasets/AIC_Data/Layout/val_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt' % USERNAME
val_root_minihand = '/home/%s/SSD_DATA/Datasets/AIC_Data' % USERNAME
######Pose Estimation Data
root_dir_pose = '/home/%s/SSD_DATA/PoseDatasets/'%USERNAME
train_lists_pose = "/home/%s/SSD_DATA/PoseDatasets/Google_PoseEstimation_20180817/Layout/trainval_Google20180817Coco.txt"% USERNAME
#train_lists_pose = "/home/%s/PoseDatasets/Layout/train85.txt"% USERNAME
val_lists_pose = "/home/%s/SSD_DATA/PoseDatasets/Google_PoseEstimation_20180817/Layout/val15.txt"% USERNAME

########################################################
###############Body and Part Information#######################
########################################################

#### start of default dis_param ####################
# 'dis_param': {
#             'brightness_prob': 0.2,
#             'brightness_delta': 20,
#             'contrast_prob': 0.2,
#             'contrast_lower': 0.5,
#             'contrast_upper': 1.5,
#             'hue_prob': 0.2,
#             'hue_delta': 18,
#             'saturation_prob': 0.2,
#             'saturation_lower': 0.5,
#             'saturation_upper': 1.5,
#             'random_order_prob': 0,
#             },
#### end of default dis_param ####################
def get_unifiedTransParam(train=True,flag_169=True):
    if flag_169:
        sample_sixteennine=True
        sample_ninesixteen=False
        resized_width = 512
        resized_height = 288

    else:
        sample_sixteennine = False
        sample_ninesixteen = True
        resized_width = 288
        resized_height = 512
    if train:
        param_dic = {
		'single_person_size':single_person_size,
		'merge_single_person_prob':merge_single_person_prob,
                # 'bboxsample_classid':bboxsample_classid,
                'emit_coverage_thre_multiple': [1.0, 0.75, 0.5,0.25], #
                'sample_sixteennine':sample_sixteennine,
                'sample_ninesixteen':sample_ninesixteen,
                'emit_coverage_thre':0.25,
                'emit_area_check':[0.02, 0.1,0.3, 1.0],    # default is [1.0,]
                'flip_prob': 0.5,
                'for_body':for_body,
                'resized_width': resized_width,
                'resized_height': resized_height,
                'visualize': False,
                'save_dir': save_path,
                'dis_param': {
                            'brightness_prob': dist_prob,
                            'brightness_delta': brightness_delta,
                            'contrast_prob': dist_prob,
                            'contrast_lower': contrast_lower,
                            'contrast_upper': contrast_upper,
                            'hue_prob': dist_prob,
                            'hue_delta': hue_delta,
                            'saturation_prob': dist_prob,
                            'saturation_lower': saturation_lower,
                            'saturation_upper': saturation_upper,
                            'random_order_prob': 0,
                            },
                'batch_sampler': [
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': min_scale[0],
                    'max_scale': max_scale[0],
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
                    'min_scale': min_scale[1],
                    'max_scale': max_scale[1],
                    'min_aspect_ratio': 0.98,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 1.0,  # is not used when flag_sample_sixteennine is true
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.7,
                    }
                },
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': min_scale[2],
                    'max_scale': max_scale[2],
                    'min_aspect_ratio': 0.98,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 1.0,  # is not used when flag_sample_sixteennine is true
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.5,
                    }
                },
                {
                'max_sample': 1,
                'max_trials': 50,
                'sampler': {
                    'min_scale': min_scale[3],
                    'max_scale': max_scale[3],
                    'min_aspect_ratio': 0.98,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 1.0,  # is not used when flag_sample_sixteennine is true
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.3,
                    }
                }
            ]
        }
        return param_dic
    else:
        return {
                'sample_sixteennine':sample_sixteennine,
                'sample_ninesixteen':sample_ninesixteen,
                'resized_width': resized_width,
                'resized_height': resized_height,
                'visualize': False,
                'save_dir': save_path,
        }

def get_unified_data_param(train=True,batchsize=1,id_val=0,flag_169=False):
    if flag_169:
        train_list = train_list_bdpd169
    else:
        train_list = train_list_bdpd916
    if train:
        if isinstance(train_list,list):
            return {
                'xml_list_multiple': train_list,
                'xml_root_multiple': root_dir_train_bdpd,
                'shuffle': True,
                'rand_skip': 500,
                'batch_size': batchsize,
                'mean_value': [104, 117, 123],
                'add_parts': True,
                'base_bindex':base_bindex
            }
        else:
            return {
                'xml_list': train_list,
                'xml_root': root_dir_train_bdpd,
                'shuffle': True,
                'rand_skip': 500,
                'batch_size': batchsize,
                'mean_value': [104, 117, 123],
                'add_parts': True,
                'base_bindex': base_bindex
            }
    else:
        if isinstance(val_list_bdpd, list):
            return {
                'xml_list': val_list_bdpd[id_val],
                'xml_root': root_dir_val_bdpd[id_val],
                'shuffle': True,
                'rand_skip': 1,
                'batch_size': 1,
                'mean_value': [104, 117, 123],
                'add_parts': True,
                'base_bindex': base_bindex
            }
        else:
            return {
                'xml_list': val_list_bdpd,
                'xml_root': root_dir_val_bdpd,
                'shuffle': True,
                'rand_skip': 1,
                'batch_size': 1,
                'mean_value': [104, 117, 123],
                'add_parts': True,
                'base_bindex': base_bindex
            }


def get_DAPDataLayer(net, train=True, batchsize=1,id_val=0,data_name = "data",label_name = "label",flag_169=True):
    if train:
        unifiedDataParam = get_unified_data_param(train=train, batchsize=batchsize,flag_169=flag_169)
    else:
        unifiedDataParam = get_unified_data_param(train=train, batchsize=batchsize, id_val=id_val,flag_169=flag_169)
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            }
        unifiedTransParam=get_unifiedTransParam(train=True,flag_169=flag_169)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            }
        unifiedTransParam=get_unifiedTransParam(train=False,flag_169=flag_169)
    net[data_name], net[label_name] = L.BBoxData(name=data_name, unified_data_param=unifiedDataParam, unified_data_transform_param=unifiedTransParam, ntop=2, **kwargs)
    return net


#######################################################
###############Minihand InFormation############################
########################################################
# List

def get_MinihandTransParam(train=True,flag_169=True):
    if flag_169:
        sample_sixteennine=True
        sample_ninesixteen=False
        resized_width = 512
        resized_height = 288

    else:
        sample_sixteennine = False
        sample_ninesixteen = True
        resized_width = 288
        resized_height = 512
    if train:
        return {
            'do_flip': True,
            'flip_prob': 0.5,
            'resized_width': resized_width,
            'resized_height': resized_height,
            'save': False,
            'save_path': save_path,
            'cov_limits': [0.3, 0.5, 0.7, 0.9],
            'dis_param': {
                'brightness_prob': dist_prob,
                'brightness_delta': brightness_delta,
                'contrast_prob': dist_prob,
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper,
                'hue_prob': dist_prob,
                'hue_delta': hue_delta,
                'saturation_prob': dist_prob,
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper,
                'random_order_prob': 0,
            },
            'flag_eqhist':flag_eqhist,
            'sample_sixteennine':sample_sixteennine,
            'sample_ninesixteen':sample_ninesixteen
        }
    else:
        return {
            'do_flip': False,
            'resized_width': resized_width,
            'resized_height': resized_height,
            'save': False,
            'save_path': save_path,
            'flag_eqhist': flag_eqhist
        }

def get_MinihandDataParam(train=True):
    if train:
        return {
            'xml_list_multiple': train_list_minihand,
            'xml_root_multiple': train_root_minihand,
            'shuffle': True,
            'rand_skip': 500,
            'batch_size': batch_size,
            'mean_value': [104,117,123],
        }
    else:
        return {
            'xml_list': val_list_minihand,
            'xml_root': val_root_minihand,
            'shuffle': False,
            'rand_skip': 1,
            'batch_size': 1,
            'mean_value': [104,117,123],
        }

def get_MinihandDataLayer(net, train=True,data_name = "data",label_name = "label",flag_169=True):
    data_param = get_MinihandDataParam(train=train)
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            }
        trans_param = get_MinihandTransParam(train=True,flag_169=flag_169)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            }
        trans_param = get_MinihandTransParam(train=False,flag_169=flag_169)
    net[data_name], net[label_name] = L.MinihandData(name=data_name, minihand_data_param=data_param, minihand_transform_param=trans_param, ntop=2, **kwargs)
    return net
#######################################################
###############Pose InFormation############################
########################################################

# if use trainval
use_trainval = False
# if use color_distortion
use_distorted = True
crop_using_resize = False
scale_max = 1.1
pose_stride = 8
def get_poseDataParam(train=True, batch_size=1):
    if train:
        xml_list = train_lists_pose
        shuffle = True
        rand_skip = 1000
        batch_size_real = batch_size_pose
        out_kps = False
    else:
        xml_list = val_lists_pose
        shuffle = False
        rand_skip = 0
        batch_size_real = 1
        out_kps = True
    return {
        'xml_list': xml_list,
        'xml_root': root_dir_pose,
        'shuffle': shuffle,
        'rand_skip': rand_skip,
        'batch_size': batch_size_real,
        'out_kps': out_kps,
    }

def get_poseDataTransParam(train = True):
    if pose_img_w == 368:
        resized_width = 384
        resized_height = 384
    else:
        resized_width = pose_img_w
        resized_height = pose_img_h
    if use_distorted and train:
        return {
            'mirror': True if train else False,
            'stride': pose_stride,
            'max_rotate_degree': 40,
            'visualize': False,
            'crop_using_resize': crop_using_resize,
            'crop_size_x': pose_img_w,
            'crop_size_y': pose_img_h,
            'scale_prob': 1.0,
            'scale_min': 0.5,
            'scale_max': scale_max,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '%stemp/visual/' % root_dir_pose,
            'root_dir': root_dir_pose,
            'resized_width': resized_width,
            'resized_height': resized_height,
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
            'crop_size_x': pose_img_w,
            'crop_size_y': pose_img_h,
            'scale_prob': 1.0,
            'scale_min': 0.5,
            'scale_max': scale_max,
            'target_dist': 0.6,
            'center_perterb_max': 40,
            'sigma': 7,
            'transform_body_joint': True,
            'mode': 5,
            'save_dir': '%stemp/visual/' % root_dir_pose,
            'root_dir': root_dir_pose,
            'resized_width': resized_width,
            'resized_height': resized_height,
            # if True -> (x-128)/256 or False -> x - mean_value[c], default is True
            'normalize': False,
            'mean_value': [104,117,123],
        }

def get_poseDataLayer(net, train=True, batch_size=1,data_name = "data",label_name = "label"):
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
    net[data_name], net[label_name] = L.PoseData(name=data_name, pose_data_param=posedata_kwargs, \
                                     ntop=2, **kwargs)
    return net




