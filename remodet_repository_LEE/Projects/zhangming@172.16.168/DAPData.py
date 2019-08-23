# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True
# '/home/%s/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt' % USERNAME,
# '/home/%s/Datasets/RemoCoco' % USERNAME,
# train_list = ['/home/%s/Datasets/RemoCoco/Layout/AllParts/train_ImgAreaAbove10000.txt' % USERNAME,
#               '/home/%s/Datasets/AIC_Data/train_V1New_AreaAbove10000.txt'% USERNAME,
#               '/home/%s/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt' % USERNAME,]
# root_dir_train = ['/home/%s/Datasets/RemoCoco' % USERNAME,
#                   '/home/%s/Datasets/AIC_Data' % USERNAME,
#                   '/home/%s/Datasets/RemoCoco' % USERNAME]
if USERNAME == "zhangming":
    train_list = ['/home/%s/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt'% USERNAME,]
    root_dir_train = ['/home/%s/Datasets/AIC_Data' % USERNAME,]
    val_list = ['/home/%s/Datasets/AIC_Data/Layout/val_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt' % USERNAME,]
    root_dir_val = ['/home/%s/Datasets/AIC_Data' % USERNAME]
else:
    train_list = ['/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt',]
    root_dir_train = ['/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/AIC_SRC',]
    val_list = ['/home/ethan/DataSets/DataFromZhangming/AIC_DataSet/Layout/val_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt',]
    root_dir_val = ['/home/%s/Datasets/AIC_Data']

# val_list = ['/home/%s/Datasets/RemoCoco/Layout/AllParts/val_ImgAreaAbove10000.txt' % USERNAME,
#             '/home/%s/Datasets/AIC_Data/val_V1New_AreaAbove10000.txt' % USERNAME]
# root_dir_val = ['/home/%s/Datasets/RemoCoco' % USERNAME,'/home/%s/Datasets/AIC_Data' % USERNAME]

# val_list = ['/home/%s/Datasets/AIC_Data/Layout/val_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt' % USERNAME,]
# root_dir_val = ['/home/%s/Datasets/AIC_Data' % USERNAME]



# train_list = '/home/%s/Datasets/RemoCoco/Layout/AllParts/train.txt' % USERNAME
# root_dir_train = '/home/%s/Datasets/RemoCoco' % USERNAME
# val_list = '/home/%s/Datasets/RemoCoco/Layout/AllParts/val.txt' % USERNAME
# root_dir_val = '/home/%s/Datasets/RemoCoco' % USERNAME



save_dir = '/home/%s/Datasets/RemoCoco/vis_aug_dj' % USERNAME




# train_list = '/home/%s/Datasets/AIC_Data/train_V1.txt' % USERNAME
# val_list = '/home/%s/Datasets/AIC_Data/val_V1.txt' % USERNAME
# root_dir = '/home/%s/Datasets/AIC_Data' % USERNAME
# save_dir = '/home/%s/Datasets/AIC_Data/vis_aug' % USERNAME
flag_noperson = False
flag_sample_sixteennine = True
resized_width = 512
resized_height = 288
flag_use_expansion = True
bboxsample_classid = 0
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
def get_unifiedTransParam(train=True):
    if train:
        param_dic = {
                # 'bboxsample_classid':bboxsample_classid,
                'emit_coverage_thre_multiple': [1.0, 0.75, 0.5,0.25], #
                'sample_sixteennine':flag_sample_sixteennine,
                'emit_coverage_thre':0.5,
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
                    'min_scale': 0.75,
                    'max_scale': 1.0,
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
                    'min_scale': 0.75,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.98,  # is not used when flag_sample_sixteennine is true
                    'max_aspect_ratio': 1.0,  # is not used when flag_sample_sixteennine is true
                    },
                'sample_constraint': {
                    'min_object_coverage': 0.3,
                    }
                }
            ]
        }
        if flag_use_expansion:
            param_dic["expand_param"] = {"prob":0.5,"max_expand_ratio":2.5}
        return param_dic
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


def get_DAPDataLayer(net, train=True, batchsize=1):
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
