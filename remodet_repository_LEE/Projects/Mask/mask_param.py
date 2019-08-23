# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
#import inputParam
import os
import sys
import math
sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True
# Net input size
Input_Width = 1024
Input_Height = 576
# RoiAlign Output size
Roi_Width = 16
Roi_Height = 16
# scale for loss of kps & mask
# Note: scale for kps: 1.0
loss_scale_kps = 1
# Note: scale for mask: 0.1
loss_scale_mask = 0.02
# kps
kps_use_conv_layers = [1,1,1,1,1,1,1,1,0,2]
channels_of_kps = [512,512,512,512,512,512,512,512,512,512]
kernel_size_of_kps = [3,3,3,3,3,3,3,3,3,3]
pad_of_kps = 1
def use_num_deconv(kps_use_conv_layers):
    num=len(kps_use_conv_layers)
    for x in kps_use_conv_layers:
        if x==1:
            num-=1
    return num
kps_use_deconv_layers = use_num_deconv(kps_use_conv_layers)
# mask
mask_use_conv_layers = 6
channels_of_mask = 256
kernel_size_of_mask = 3
pad_of_mask = 1
mask_use_deconv_layers = 0
# Kps HeatMap size
Rw_Kps = Roi_Width*(2**kps_use_deconv_layers)
Rh_Kps = Roi_Height*(2**kps_use_deconv_layers)
# Mask Binary-mask-size
Rw_Mask = Roi_Width*(2**mask_use_deconv_layers)
Rh_Mask = Roi_Height*(2**mask_use_deconv_layers)


# ssdparam
ssdparam = {
    'boxsizes': [[0.0645,0.1707,0.288],[0.3837,0.5096,0.7273,0.95]],
    'num_classes': 2,
    'use_bn': True,
    'prior_variance': [0.1,0.1,0.2,0.2],
    'normalizations': [],
    'aspect_ratios': [[[1, 0.25, 0.5],[1, 0.25, 0.5],[1, 0.25, 0.5]],[[1, 0.25, 0.5],[1, 0.25, 0.5],[1,0.5],[1]]],
    'flip': True,
    'clip': True,
    'inter_layer_channels': [[[256,1],[128,3]],[[256,1],[128,3]]],
    'kernel_size': 3,
    'pad': 1,
    'dropout': False,
}
partsparam = {
    'boxsizes':[[0.005,0.008],[0.01,0.015],[0.05,0.06],[0.1,0.15]],
    'num_classes':4,
    'prior_variance':[0.1,0.1,0.2,0.2],
    'normalizations':[],
    'aspect_ratios': [[[1],[1]],[[1],[1]],[[1],[1]]],
    'flip': True,
    'clip': True,
    'inter_layer_channels': [[[128,1],[128,3]],[[256,1],[128,3]],[[256,1],[128,3]]],
    'kernel_size': 3,
    'pad': 1,
    'dropout': False,
    'use_bn': True,
}
# bbox-loss
bbox_loss_param = {
	'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
	'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
	'loc_weight': 4,
	'conf_weight': 1,
	'num_classes': 2,
    'do_neg_mining':True,
    'encode_variance_in_target':False,
	'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'map_object_to_agnostic':False,
    'encode_variance_in_target':False,
	'overlap_threshold': 0.5,
	'use_difficult_gt': False,
	'neg_pos_ratio': 3,
	'neg_overlap': 0.5,
	'code_type': P.PriorBox.CENTER_SIZE,
    'size_threshold': 0.0001,
    'use_prior_for_matching':True,
    'share_location':True,
    'alias_id':0,
}
dense_bbox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': 4,
    'conf_weight': 1,
    'num_classes': 2,
    'do_neg_mining':False,
    'encode_variance_in_target':False,
    'encode_variance_in_target':False,
    'overlap_threshold': 0.5,
    'use_difficult_gt': False,
    'neg_pos_ratio': 9,
    'neg_overlap': 0.5,
    'code_type': P.PriorBox.CENTER_SIZE,
    'size_threshold': 0.001,
    'use_prior_for_matching':True,
    'alias_id':0,
    'using_focus_loss':True,
    'gama':2,

}
dense_parts_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type':P.MultiBoxLoss.LOGISTIC,
    'loc_weight': 4,
    'conf_weight': 1,
    'num_classes': 4,
    'do_neg_mining':False,
    'encode_variance_in_target':False,
    'encode_variance_in_target':False,
    'overlap_threshold': 0.5,
    'use_difficult_gt': False,
    'neg_pos_ratio': 9,
    'neg_overlap': 0.5,
    'code_type': P.PriorBox.CENTER_SIZE,
    'size_threshold': 0.0001,
    'use_prior_for_matching':True,
    'alias_id':1,
    'using_focus_loss':True,
    'gama':2,
}
parts_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type':P.MultiBoxLoss.LOGISTIC,
    'loc_weight': 4,
    'conf_weight': 1,
    'num_classes': 4,
    'do_neg_mining':True,
    'encode_variance_in_target':False,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'map_object_to_agnostic':False,
    'encode_variance_in_target':False,
    'overlap_threshold': 0.5,
    'use_difficult_gt': False,
    'neg_pos_ratio': 3,
    'neg_overlap': 0.5,
    'code_type': P.PriorBox.CENTER_SIZE,
    'size_threshold': 0.0001,
    'use_prior_for_matching':True,
    'share_location':True,
    'alias_id':1,
}
det_out_param = {
	'num_classes': 2,
	'share_location': True,
	'variance_encoded_in_target':False,
	'conf_threshold':0.5,
    'nms_threshold': 0.45,
    'size_threshold':0.0001,
    'code_type': P.PriorBox.CENTER_SIZE,
    'top_k': 200,
    'alias_id':0,
    'visual_param': {
                # enable or disable the visualize function
                'visualize': False,
                'conf_threshold':0.8,
                'size_threshold':0.001,
                # image display size
                'display_maxsize': 1000,
                # rectangle line width
                'line_width': 4,
                # rectangle color
                'color_param': {
                    'rgb': {
                        'val': [0,255,0],
                    }
                }
        }
}
parts_out_param = {
    'num_classes': 4,
    'share_location': True,
    'variance_encoded_in_target':False,
    'conf_threshold':0.5,
    'nms_threshold': 0.45,
    'size_threshold':0.0001,
    'code_type': P.PriorBox.CENTER_SIZE,
    'top_k': 200,
    'alias_id':1,
    'visual_param': {
                # enable or disable the visualize function
                'visualize': False,
                'conf_threshold':0.5,
                'size_threshold':0.001,
                # image display size
                'display_maxsize': 1000,
                # rectangle line width
                'line_width': 4,
                # rectangle color
                'color_param': {
                    'rgb': {
                        'val': [0,255,0],
                    }
                }
        }
}
vis_out_param = {
    'num_classes': 4,
    'share_location': True,
    'variance_encoded_in_target':False,
    'conf_threshold':0.5,
    'nms_threshold': 0.45,
    'size_threshold':0.0001,
    'code_type': P.PriorBox.CENTER_SIZE,
    'top_k': 200,
    'visual_param': {
                # enable or disable the visualize function
                'visualize': True,
                'conf_threshold':0.5,
                'size_threshold':0.001,
                # image display size
                'display_maxsize': 1000,
                # rectangle line width
                'line_width': 4,
                # rectangle color
                'color_param': {
                    'rgb': [
                        {
                        'val': [0,255,255]
                        },
                        {
                        'val': [0,255,255]
                        },
                        {
                        'val': [0,255,255]
                        },
                        {
                        'val': [0,255,255]
                        }
                       

                    ]                    
                }
        }
}
det_eval_param = {
	'num_classes':2,
	'evaluate_difficult_gt':False,
	'boxsize_threshold': [0,0.01,0.05,0.1,0.15,0.2,0.25],
	'iou_threshold': [0.9,0.75,0.5],
	'name_size_file': "",

}
parts_eval_param = {
    'num_classes':4,
    'evaluate_difficult_gt':False,
    'boxsize_threshold': [0.00,0.001,0.0025,0.005,0.01,0.025,0.05],
    'iou_threshold': [0.9,0.75,0.5],
    'name_size_file': "",

}
# ROIAlign
roi_align_param = {
	'roi_resized_width': Roi_Height,
	'roi_resized_height': Roi_Width,
	'inter_times': 4,
	'spatial_scale': 1./16.,
}
box_matching_param = {
    'overlap_threshold':0.5,
    'use_difficult_gt':False,
    'size_threshold':0.001,
    'top_k':100,
}
visual_mask_param={
    'kps_threshold':0.05,
    'mask_threshold':0.1,
    'write_frames':False,
    'output_directory':"/home/{}/Models/Results/MaskRcnn_vis".format(USERNAME),
    'show_mask':True,
    'show_kps':True,
    'print_score':True,
    'max_dis_size':1000,
}
