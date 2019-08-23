# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/zhangming/work/minihand/remodet_repository/python")
# import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
from DetRelease_General import *
sys.dont_write_bytecode = True
# ##############################################################################
# ------------------------------- USING SSD2 -----------------------------------
use_ssd2_for_detection = True
# ##############################################################################
# ----------------------- -------Conv6 Params-----------------------------------
Conv6_Param = {
    'conv6_output':[128,128,128,128,128],
    'conv6_kernal_size':[3,3,3,3,3],
}
# ##############################################################################
# -------------------------------Labels Definition------------------------------
# 0-person, 1-hand, 2-head, and 3-face
ssd_1_gt_labels = [0] # For Body
ssd_2_gt_labels = [1,3] # For Part: Hand and Face
ssd_3_gt_labels = [1] # For Minihand
eval_gt_labels  = [0,1,3] # Only Hand


def getTargetLabels(num):
    labels = []
    for i in range(num):
        labels.append(i+1)
    return labels
# ##############################################################################

def multi_key_dict_get(d, k):
    for keys, v in d.items():
        if k in keys:
            return v
    return None

def get_anchor_aspect_ratios(anchor_boxsizes,anchor_aspect_pixels,resized_width,resized_height):

    anchor_aspect_ratios = []
    for anchorsize_per_feat in anchor_boxsizes:
        as_per_feat = []
        for asize in anchorsize_per_feat:
            pixelratios = multi_key_dict_get(anchor_aspect_pixels,asize)
            as_per_scale = []
            for pr in pixelratios:
                normratio = float(resized_height) / float(resized_width) * pr
                as_per_scale.append(normratio)
            as_per_feat.append(as_per_scale)
        anchor_aspect_ratios.append(as_per_feat)
    return anchor_aspect_ratios
def show_anchor_pixelsize(anchor_boxsizes,anchor_aspect_ratios,resized_width,resized_height):
    for ifeat in xrange(len(anchor_boxsizes)):
        ws = []
        hs = []
        #w = bsize*sqrt(as)*img_w, h =  bsize/sqrt(as)*img_h
        for iscale  in xrange(len(anchor_boxsizes[ifeat])):
            asize = anchor_boxsizes[ifeat][iscale]
            as_per_scale = anchor_aspect_ratios[ifeat][iscale]
            for a in as_per_scale:
                w = min(int(asize * math.sqrt(a)*float(resized_width)),resized_width)
                h = min(int(asize / math.sqrt(a)*float(resized_height)),resized_height)
                ws.append(w)
                hs.append(h)
        print "width%d:"%ifeat,ws
        print "height%d:"%ifeat,hs
####USER DESIGN PART
############################For BODY Anchor
anchor_boxsizes_body = [[0.06,0.12],[0.18,0.24,0.32],[0.4,0.6,0.8,0.95]]
def get_anchor_aspect_ratios_body(flag_169):
    if flag_169:
        anchor_aspect_pixels_body = {
            (0.06,0.12,0.18,0.24,0.32,0.4,0.6):(1,2,0.5),
            (0.8,):(1.0,2.0),# w/h
            (0.95,):(2.0,)
        }
        resized_width = 512
        resized_height = 288
    else:
        anchor_aspect_pixels_body = {
            (0.06, 0.12, 0.18, 0.24, 0.32, 0.4, 0.6): (1, 2, 0.5),
            (0.8,): (1.0, 0.5),  #
            (0.95,): (0.5,)
        }
        resized_width = 288
        resized_height = 512
    anchor_aspect_ratios_body = get_anchor_aspect_ratios(anchor_boxsizes_body,anchor_aspect_pixels_body,resized_width,resized_height)
    print "printing anchor_aspect_ratios_body.... "
    show_anchor_pixelsize(anchor_boxsizes_body,anchor_aspect_ratios_body,resized_width,resized_height)
    return anchor_aspect_ratios_body
############################For PART Anchor
anchor_boxsizes_part = [[0.05,0.1],[0.15,0.25,0.35],[0.45,0.6,0.75,0.95]]
def get_anchor_aspect_ratios_part(flag_169):
    if flag_169:
        anchor_aspect_pixels_part = {
            (0.05,0.1,0.15,0.25,0.35,0.45,0.6):(1,),
            (0.75,):(1.0,),# w/h
            (0.95,):(1.0,)
        }
        resized_width = 512
        resized_height = 288
    else:
        anchor_aspect_pixels_part = {
            (0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.6): (1,),
            (0.75,): (1.0,),  #
            (0.95,): (1.0,)
        }
        resized_width = 288
        resized_height = 512
    anchor_aspect_ratios_part = get_anchor_aspect_ratios(anchor_boxsizes_part,anchor_aspect_pixels_part,resized_width,resized_height)
    print "printing aspect_ratios_part.... "
    show_anchor_pixelsize(anchor_boxsizes_part,anchor_aspect_ratios_part,resized_width,resized_height)
    return anchor_aspect_ratios_part
############################For Minihand Anchor
anchor_boxsizes_minihand = [[0.03,0.06,0.1],]
def get_anchor_aspect_ratios_minihand(flag_169):

    if flag_169:
        anchor_aspect_pixels_minihand = {
            (0.03,0.06,0.1):(1,),
        }
        resized_width = 512
        resized_height = 288
    else:
        anchor_aspect_pixels_minihand = {
            (0.03, 0.06, 0.1): (1,),
        }
        resized_width = 288
        resized_height = 512
    anchor_aspect_ratios_minihand = get_anchor_aspect_ratios(anchor_boxsizes_minihand,anchor_aspect_pixels_minihand,resized_width,resized_height)
    print "printing aspect_ratios_minihand.... "
    show_anchor_pixelsize(anchor_boxsizes_minihand,anchor_aspect_ratios_minihand,resized_width,resized_height)
    return anchor_aspect_ratios_minihand
############################For Minihand_And_Part Anchor
anchor_boxsizes_minihand_part = [[0.03,0.06],[0.1,0.13],[0.18,0.25,0.35],[0.45,0.6,0.75,0.95]]
def get_anchor_aspect_ratios_minihand_part(flag_169):

    if flag_169:
        anchor_aspect_pixels_minihand_part = {
            (0.03,0.06,0.1,0.13,0.18,0.25,0.35,0.45,0.6,0.75,0.95):(1,),
        }
        resized_width = 512
        resized_height = 288
    else:
        anchor_aspect_pixels_minihand_part = {
            (0.03, 0.06, 0.1, 0.13, 0.18, 0.25, 0.35, 0.45, 0.6, 0.75, 0.95): (1,),
        }
        resized_width = 288
        resized_height = 512
    anchor_aspect_ratios_minihand_part = get_anchor_aspect_ratios(anchor_boxsizes_minihand_part,anchor_aspect_pixels_minihand_part,resized_width,resized_height)
    print "printing aspect_ratios_minihand_part.... "
    show_anchor_pixelsize(anchor_boxsizes_minihand_part,anchor_aspect_ratios_minihand_part,resized_width,resized_height)
    return anchor_aspect_ratios_minihand_part
get_anchor_aspect_ratios_body(flag_169_global)
get_anchor_aspect_ratios_part(flag_169_global)
get_anchor_aspect_ratios_minihand(flag_169_global)
get_anchor_aspect_ratios_minihand_part(flag_169_global)
# ---------------------------------SSD1 Params----------------------------------
def get_ssd_Param_1(flag_169,bboxloss_loc_weight = 2.5,bboxloss_conf_weight=2.5):
    return {
        # FeatureLayers
        'feature_layers': ['featuremap1','featuremap2','featuremap3'],
        # num_classes
        'num_classes': len(ssd_1_gt_labels) + 1,
        'gt_labels': ssd_1_gt_labels,
        'target_labels': getTargetLabels(len(ssd_1_gt_labels)),
        'alias_id': ssd_1_gt_labels[0],
        # Anchors
        'anchor_boxsizes': anchor_boxsizes_body,
        'anchor_aspect_ratios': get_anchor_aspect_ratios_body(flag_169),
        'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
        'anchor_flip': True,
        'anchor_clip': True,
        # InterLayers
        'interlayers_normalizations': [],
        'interlayers_use_batchnorm': False,
        'interlayers_channels_kernels': [[[64,1],[128,3]],\
                                         [[64,1],[128,3]],
                                         [[64, 1], [128, 3]]],
        # @types
        'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,
        'bboxloss_use_dense_boxes': False,
        # @weights
        'bboxloss_loc_weight': bboxloss_loc_weight,
        'bboxloss_conf_weight': bboxloss_conf_weight,
        # @Overlaps & boxsize threshold
        'bboxloss_overlap_threshold': bboxloss_overlap_threshold_body,
        'bboxloss_neg_overlap': bboxloss_neg_overlap_body,
        'bboxloss_size_threshold': 0.001,
        # @OHEM & FocusLoss
        'bboxloss_do_neg_mining': True,
        'bboxloss_neg_pos_ratio': bboxloss_neg_pos_ratio_body,
        'bboxloss_using_focus_loss': False,
        'bboxloss_focus_gama': 2,
        # unchanged
        'bboxloss_use_difficult_gt': False,
        'bboxloss_code_type': P.PriorBox.CENTER_SIZE,
        'bboxloss_normalization': P.Loss.VALID,
        # detection out
        'detout_target_labels': ssd_1_gt_labels,
        'detout_conf_threshold': 0.5,
        'detout_nms_threshold': 0.45,
        'detout_size_threshold': 0.0001,
        'detout_top_k': 200,
        'flag_noperson':flag_noperson_body,
        'matchtype_anchorgt':matchtype_anchorgt,
        "margin_ratio":margin_ratio,
        'sigma_angtdist':sigma_angtdist,
	'only_w':only_w,
	'single_person_size':single_person_size,
	'merge_single_person_prob':merge_single_person_prob
    }
print get_ssd_Param_1(True)
# ##############################################################################
# ---------------------------------SSD2 Params----------------------------------
def get_ssd_Param_2(flag_169,bboxloss_loc_weight = 2.0,bboxloss_conf_weight=1.0):
    return {
        # FeatureLayers
        # 'feature_layers': ['pool1_recon'],
        # 'feature_layers': ['conv4_4/incep_deconv'],
        'feature_layers': ['featuremap1','featuremap2','featuremap3'],

        # num_classes
        'num_classes': len(ssd_2_gt_labels) + 1,
        'gt_labels': ssd_2_gt_labels,
        'target_labels': getTargetLabels(len(ssd_2_gt_labels)),
        'alias_id': ssd_2_gt_labels[0],
        # Anchors
        'anchor_boxsizes': anchor_boxsizes_part,
        # 'anchor_aspect_ratios': [[[0.5],[0.5],[0.5]]],
        'anchor_aspect_ratios': get_anchor_aspect_ratios_part(flag_169),
        'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
        'anchor_flip': True,
        'anchor_clip': True,
        # InterLayers
        'interlayers_normalizations': [],
        'interlayers_use_batchnorm': False,
        'interlayers_channels_kernels': [[[64,1],[128,3]],
                                         [[64, 1], [128, 3]],
                                         [[64, 1], [128, 3]]],
        # Loss layer
        # @types
        'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,
        'bboxloss_use_dense_boxes': True,
        # @weights
        'bboxloss_loc_weight': bboxloss_loc_weight,
        'bboxloss_conf_weight': bboxloss_conf_weight,
        # @Overlaps & boxsize threshold
        'bboxloss_overlap_threshold': bboxloss_overlap_threshold_part,
        'bboxloss_neg_overlap': bboxloss_neg_overlap_part,
        'bboxloss_size_threshold': 0.0003,
        # @OHEM & FocusLoss
        'bboxloss_do_neg_mining': True,
        'bboxloss_neg_pos_ratio': bboxloss_neg_pos_ratio_part,
        'bboxloss_using_focus_loss': False,
        'bboxloss_focus_gama': 2,
        # unchanged
        'bboxloss_use_difficult_gt': False,
        'bboxloss_code_type': P.PriorBox.CENTER_SIZE,
        'bboxloss_normalization': P.Loss.VALID,
        # detection out
        'detout_target_labels': ssd_2_gt_labels,
        'detout_conf_threshold': 0.5,
        'detout_nms_threshold': 0.4,
        'detout_size_threshold': 0.0001,
        'detout_top_k': 200,
        'flag_noperson': flag_noperson_part,
    }
def get_ssd_Param_3(flag_169,bboxloss_loc_weight = 2.0,bboxloss_conf_weight=1.0):
    return {
        'feature_layers': ['mini_multiscale'],

        # num_classes
        'num_classes': len(ssd_3_gt_labels) + 1,
        'gt_labels': ssd_3_gt_labels,
        'target_labels': getTargetLabels(len(ssd_3_gt_labels)),
        'alias_id': ssd_3_gt_labels[0],
        # Anchors
        'anchor_boxsizes': anchor_boxsizes_minihand,
        # 'anchor_aspect_ratios': ,
        'anchor_aspect_ratios': get_anchor_aspect_ratios_minihand(flag_169),
        'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
        'anchor_flip': True,
        'anchor_clip': True,
        # InterLayers
        'interlayers_normalizations': [],
        'interlayers_use_batchnorm': False,
        'interlayers_channels_kernels': [[[32,3],[32,3]]],
        # Loss layer
        # @types
        'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,
        'bboxloss_use_dense_boxes': False,
        # @weights
        'bboxloss_loc_weight': bboxloss_loc_weight,
        'bboxloss_conf_weight': bboxloss_conf_weight,
        # @Overlaps & boxsize threshold
        'bboxloss_overlap_threshold': bboxloss_overlap_threshold_mini,
        'bboxloss_neg_overlap': bboxloss_neg_overlap_mini,
        'bboxloss_size_threshold': 0.0003,
        # @OHEM & FocusLoss
        'bboxloss_do_neg_mining': True,
        'bboxloss_neg_pos_ratio': bboxloss_neg_pos_ratio_mini,
        'bboxloss_using_focus_loss': False,
        'bboxloss_focus_gama': 2,
        # unchanged
        'bboxloss_use_difficult_gt': False,
        'bboxloss_code_type': P.PriorBox.CENTER_SIZE,
        'bboxloss_normalization': P.Loss.VALID,
        # detection out
        'detout_target_labels': ssd_3_gt_labels,
        'detout_conf_threshold': 0.5,
        'detout_nms_threshold': 0.4,
        'detout_size_threshold': 0.0001,
        'detout_top_k': 200,
        'flag_noperson': flag_noperson_part,
    }
def get_ssd_Param_4(flag_169,bboxloss_loc_weight = 2.0,bboxloss_conf_weight=1.0):
    return {
        'feature_layers': ['mini_multiscale','featuremap1','featuremap2','featuremap3'],

        # num_classes
        'num_classes': len(ssd_2_gt_labels) + 1,
        'gt_labels': ssd_2_gt_labels,
        'target_labels': getTargetLabels(len(ssd_2_gt_labels)),
        'alias_id': ssd_2_gt_labels[0],
        # Anchors
        'anchor_boxsizes': anchor_boxsizes_minihand_part,
        # 'anchor_aspect_ratios': ,
        'anchor_aspect_ratios': get_anchor_aspect_ratios_minihand_part(flag_169),
        'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
        'anchor_flip': True,
        'anchor_clip': True,
        # InterLayers
        'interlayers_normalizations': [],
        'interlayers_use_batchnorm': False,
        'interlayers_channels_kernels': [[[32,3],[32,3]],
                                         [[64, 1], [128, 3]],
                                         [[64, 1], [128, 3]],
                                         [[64, 1], [128, 3]]
                                         ],
        # Loss layer
        # @types
        'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,
        'bboxloss_use_dense_boxes': True,
        # @weights
        'bboxloss_loc_weight': bboxloss_loc_weight,
        'bboxloss_conf_weight': bboxloss_conf_weight,
        # @Overlaps & boxsize threshold
        'bboxloss_overlap_threshold': bboxloss_overlap_threshold_part,
        'bboxloss_neg_overlap': bboxloss_neg_overlap_part,
        'bboxloss_size_threshold': 0.0003,
        # @OHEM & FocusLoss
        'bboxloss_do_neg_mining': True,
        'bboxloss_neg_pos_ratio': bboxloss_neg_pos_ratio_part,
        'bboxloss_using_focus_loss': False,
        'bboxloss_focus_gama': 2,
        # unchanged
        'bboxloss_use_difficult_gt': False,
        'bboxloss_code_type': P.PriorBox.CENTER_SIZE,
        'bboxloss_normalization': P.Loss.VALID,
        # detection out
        'detout_target_labels': ssd_2_gt_labels,
        'detout_conf_threshold': 0.5,
        'detout_nms_threshold': 0.4,
        'detout_size_threshold': 0.0001,
        'detout_top_k': 200,
        'flag_noperson': flag_noperson_part,
    }
# ##############################################################################
# ---------------------------------Eval Params----------------------------------
def get_eval_Param(eval_gt_labels=(0,1,3)):
    return {
        'eval_difficult_gt': False,
        'eval_boxsize_threshold': [0, 16e-4, 64e-4, 100e-4, 0.1, 0.15, 0.25],
        'eval_iou_threshold': [0.9, 0.75, 0.5],
        'eval_gt_labels': eval_gt_labels,
        'eval_num_classes': len(eval_gt_labels) + 1
    }
