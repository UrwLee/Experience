# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
# ----------------------- -------yolo Detector----------------------------------
code_type = P.McBoxLoss.YOLO

yolo_Param = {
    # Headers
    # Norm -> before conv, use_BN/inter_layer_channels -> interConvLayer
    # kernel_size/pad -> Conv-kernel for Loc/Conf Preds.
    # we use 512 for interlayers and 1/0-kernel for Loc/Conf prediction
    'mcheader_normalization': -1,
    'mcheader_use_batchnorm': True,
    'mcheader_inter_layer_channels': 512,
    'mcheader_kernel_size': 1,
    'mcheader_pad': 0,
    # MC-Loss
    'mcloss_num_classes': 1,
    'mcloss_boxsizes': [0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
    'mcloss_aspect_ratios': [1, 0.5, 2],
    'mcloss_pwidths': [],
    'mcloss_pheights': [],
    'mcloss_overlap_threshold': 0.5,
    'mcloss_use_prior_for_matching': True,
    'mcloss_use_prior_for_init': False,
    'mcloss_use_difficult_gt': True,
    'mcloss_rescore': True,
    'mcloss_code_type': code_type,
    'mcloss_clip': True,
    'mcloss_iters': 0,
    'mcloss_iter_using_bgboxes': 10000,
    'mcloss_background_box_loc_scale': 0.01,
    'mcloss_object_scale': 5,
    'mcloss_noobject_scale': 0.5,
    'mcloss_class_scale': 1,
    'mcloss_loc_scale': 1,
    'mcloss_background_label_id': 0,
    'mcloss_normalization': P.Loss.BATCH_SIZE,
    # Det & Eval
    'mcdetout_conf_threshold': 0.01,
    'mcdetout_nms_threshold': 0.45,
    'mcdetout_boxsize_threshold': 0.001,
    'mcdetout_top_k': 200,
    'mcdetout_visualize': False,
    'mcdetout_visualize_conf_threshold': 0.5,
    'mcdetout_visualize_size_threshold': 0.01,
    'mcdetout_display_maxsize': 1000,
    'mcdetout_line_width': 4,
    'mcdetout_code_type': code_type,
    'mcdetout_color': [[0,255,0]],
    'deteval_evaluate_difficult_gt': False,
    'deteval_boxsize_threshold': [0,0.01,0.05,0.1,0.15,0.2,0.25],
    'deteval_iou_threshold': [0.9,0.75,0.5],
}
