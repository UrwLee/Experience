# -*- coding: utf-8 -*-
# import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
# ----------------------- -------SSD Detector----------------------------------
yolo_ssd_Param = {
    'num_classes': 2,
    # Multi-Layer detector header
    'multilayers_boxsizes': [0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
    'multilayers_aspect_ratios': [1, 2, 0.5],
    'multilayers_use_batchnorm': True,
    'multilayers_inter_layer_channels': [],
    'multilayers_prior_variance': [0.1, 0.1, 0.2, 0.2],
    'multilayers_normalizations': [],
    'multilayers_flip': False,
    'multilayers_clip': False,
    'multilayers_kernel_size': 3,
    'multilayers_pad': 1,
    # Loss layer
    'multiloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'multiloss_conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'multiloss_loc_weight': 1.0,
    'multiloss_conf_weight': 1.0,
    'multiloss_overlap_threshold': 0.5,
    'multiloss_neg_overlap': 0.5,
    'multiloss_neg_pos_ratio': 3,
    'multiloss_share_location': True,
    'multiloss_match_type': P.MultiBoxLoss.PER_PREDICTION,
    'multiloss_use_prior_for_matching': True,
    'multiloss_background_label_id': 0,
    'multiloss_use_difficult_gt': True,
    'multiloss_do_neg_mining': True,
    'multiloss_code_type': P.PriorBox.CENTER_SIZE,
    'multiloss_encode_variance_in_target': False,
    'multiloss_map_object_to_agnostic': False,
    'multiloss_normalization': P.Loss.VALID,
    # detection out
    'detectionout_conf_threshold': 0.01,
    'detectionout_nms_threshold': 0.45,
    'detectionout_boxsize_threshold': 0.001,#no nus
    'detectionout_top_k': 200,
    'detectionout_visualize': False,
    'detectionout_visualize_conf_threshold': 0.5,
    'detectionout_visualize_size_threshold': 0.01,
    'detectionout_display_maxsize': 1000,
    'detectionout_line_width': 4,
    'detectionout_color': [[0,255,0],],
    # detection eval
    'detectioneval_evaluate_difficult_gt': False,
    'detectioneval_boxsize_threshold': [0,0.01,0.05,0.1,0.15,0.2,0.25],
    'detectioneval_iou_threshold': [0.9,0.75,0.5],
}
