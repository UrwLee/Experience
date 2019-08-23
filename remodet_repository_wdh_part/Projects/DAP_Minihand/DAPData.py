# -*- coding: utf-8 -*-
from username import USERNAME
import sys
# sys.path.insert(0, "/home/zhangming/work/minihand/remodet_repository/python")
sys.path.insert(0, '/home/xjx/work/remodet_repository_DJ/python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.dont_write_bytecode = True

# List
train_list = '/home/zhangming/Datasets/AIC_REMOCapture/trainval_remocapture_AIC_handface_20180530_union_remocapture_handface_20180807.txt'
#train_list = '/home/zhangming/Datasets/AIC_Data/Layout/train_Hand_ClusterModMiniHand_MultiScale_A1Featconv2_hand_maxNclus200_CNTThre1500_allkeeptrue1AndremoveHandInIou0.1WHmax2.1.txt'
# train_list = '/home/zhangming/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt'
val_list = '/home/zhangming/Datasets/AIC_Data/Layout/val_handclean_20180427_and_remocapture_20180427.txt'
# Root
xml_root = "/home/zhangming/Datasets/AIC_REMOCapture"
image_root = "/home/zhangming/Datasets/AIC_REMOCapture"

xml_root_val = '/home/zhangming/Datasets/AIC_Data'
image_root_val = '/home/zhangming/Datasets/AIC_Data'

# Save
save_path = '/home/zhangming/tmp'
# Rsz Dim
resized_width = 288
resized_height = 512
# BatchSize
batch_size = 24
# ### default
# 'dis_param': {
#     'brightness_prob': 0.2,
#     'brightness_delta': 20,
#     'contrast_prob': 0.2,
#     'contrast_lower': 0.5,
#     'contrast_upper': 1.5,
#     'hue_prob': 0.2,
#     'hue_delta': 18,
#     'saturation_prob': 0.2,
#     'saturation_lower': 0.5,
#     'saturation_upper': 1.5,
#     'random_order_prob': 0,
# }
# ########
if float(resized_width)/float(resized_height)==16.0/9.0:
    sample_sixteennine = True
    sample_ninesixteen = False
else:
    sample_sixteennine = False
    sample_ninesixteen = True

dist_prob = 0.5
flag_eqhist = False
brightness_delta = 16
contrast_lower = 0.8
contrast_upper = 1.2
hue_delta = 12
saturation_lower = 0.8
saturation_upper = 1.2
def get_MinihandTransParam(train=True):
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
            'xml_list': train_list,
            'xml_root': xml_root,
            'image_root': image_root,
            'shuffle': True,
            'rand_skip': 500,
            'batch_size': batch_size,
            'mean_value': [104,117,123],
        }
    else:
        return {
            'xml_list': val_list,
            'xml_root': xml_root_val,
            'image_root': image_root_val,
            'shuffle': False,
            'rand_skip': 1,
            'batch_size': 1,
            'mean_value': [104,117,123],
        }

def get_MinihandDataLayer(net, train=True):
    data_param = get_MinihandDataParam(train=train)
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            }
        trans_param = get_MinihandTransParam(train=True)
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            }
        trans_param = get_MinihandTransParam(train=False)
    net.data, net.label = L.MinihandData(name="data", minihand_data_param=data_param, minihand_transform_param=trans_param, ntop=2, **kwargs)
    return net
