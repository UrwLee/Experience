# -*- coding: utf-8 -*-
# import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
# ----------------------- -------SSD Detector----------------------------------
ssd_Param = {
    # Conv6
    'multilayers_conv6_output':[128,128,128,128,128,128],
    'multilayers_conv6_kernal_size':[3,3,3,3,3,3],
    # Headers
    'multilayers_boxsizes': [[0.05,0.1],[0.15,0.25,0.35],[0.45,0.6,0.75,0.95]],
    'multilayers_aspect_ratios': [[[0.5,0.25],[0.5,0.25]],[[1,0.5,0.25],[1,0.5,0.25],[1,0.5,0.25]],[[1,0.5,0.25],[1,0.5,0.25],[1,0.5,0.25],[1]]],
    'num_classes': 2,
    'multilayers_use_batchnorm': True,
    'multilayers_prior_variance': [0.1, 0.1, 0.2, 0.2],
    'multilayers_normalizations': [],
    'multilayers_flip': True,
    'multilayers_clip': True,
    'multilayers_inter_layer_channels': [[[64,1],[128,3]],[[64,1],[128,3]],[[64,1],[128,3]]],
    'multilayers_kernel_size': 3,
    'multilayers_pad': 1,
    # Loss layer
    'multiloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'multiloss_conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'multiloss_loc_weight': 1.0,
    'multiloss_conf_weight': 4.0,
    'multiloss_share_location': True,
    'multiloss_match_type': P.MultiBoxLoss.PER_PREDICTION,
    'multiloss_do_neg_mining': False,
    'multiloss_overlap_threshold': 0.5,
    'multiloss_neg_overlap': 0.5,
    'multiloss_neg_pos_ratio': 1000,
    'multiloss_use_difficult_gt': False,
    'multiloss_code_type': P.PriorBox.CENTER_SIZE,
    'multiloss_size_threshold': 0.0001,
    'multiloss_alias_id': 0,
    'multiloss_using_focus_loss': True,
    'multiloss_focus_gama': 2,
    'multiloss_normalization': P.Loss.VALID,
    # detection out
    'detectionout_conf_threshold': 0.5,
    'detectionout_nms_threshold': 0.45,
    'detectionout_size_threshold': 0.0000,
    'detectionout_top_k': 200,
    # detection eval
    'detectioneval_evaluate_difficult_gt': False,
    'detectioneval_boxsize_threshold': [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    'detectioneval_iou_threshold': [0.9,0.75,0.5],
}
