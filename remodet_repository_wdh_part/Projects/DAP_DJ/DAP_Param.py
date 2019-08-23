# -*- coding: utf-8 -*-
# import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
from DAPData import flag_noperson

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
ssd_1_gt_labels = [0]  # person
ssd_2_gt_labels = [1,3] # hand and face
eval_gt_labels  = [0,1,3] # eval labels
# eval_gt_labels = [0]
def getTargetLabels(num):
    labels = []
    for i in range(num):
        labels.append(i+1)
    return labels
# ##############################################################################
# ---------------------------------SSD1 Params----------------------------------
ssd_Param_1 = {
    # FeatureLayers
    'feature_layers': ['featuremap1','featuremap2','featuremap3'],
    # 'feature_layers': ['featur                  emap2', 'featuremap3'],
    # num_classes
    'num_classes': len(ssd_1_gt_labels) + 1,
    'gt_labels': ssd_1_gt_labels,
    'target_labels': getTargetLabels(len(ssd_1_gt_labels)),
    'alias_id': ssd_1_gt_labels[0],
    # Anchors
    ###Default anchors by concat featuremap
    # 'anchor_boxsizes': [[0.0645,0.1707,0.288],[0.3837,0.5096,0.7273,0.95]],
    # 'anchor_aspect_ratios': [[[1, 0.25, 0.5], \
    #                            [1, 0.25, 0.5], \
    #                            [1, 0.25, 0.5]],\
    #                           [[1, 0.25, 0.5], \
    #                            [1, 0.25, 0.5], \
    #                            [1,0.5],        \
    #                            [1]]],
    #new anchors  by single featuremap
    # 'anchor_boxsizes': [[0.08,0.12],[0.2,0.3,0.4],[0.5,0.65,0.8,0.95]],#DJ Design
    'anchor_boxsizes': [[0.06,0.12],[0.18,0.24,0.32],[0.4,0.6,0.8,0.95]],#ZM Design
    'anchor_aspect_ratios': [[[1, 0.25, 0.5], \
                               [1, 0.25, 0.5]],
                               [[1, 0.25, 0.5], \
                               [1, 0.25, 0.5], \
                               [1, 0.25, 0.5]],\
                              [[1, 0.25, 0.5], \
                               [1, 0.25, 0.5], \
                               [1,0.5],        \
                               [1]]],
    # 'anchor_aspect_ratios': [[[1, 2.0, 0.5], \
    #                            [1, 2.0, 0.5], \
    #                            [1, 2.0, 0.5]],\
    #                           [[1, 2.0, 0.5], \
    #                            [1, 2.0, 0.5], \
    #                            [1,2.0,0.5],        \
    #                            [1]]],
    #'anchor_boxsizes': [[0.05],[0.1,0.3],[0.5,0.75,0.95]],
    #'anchor_aspect_ratios': [[[0.5]], \
    #                         [[0.5],  \
    #                          [0.5]], \
    #                          [[0.5],  \
    #                          [1,0.5],\
    #                         [1,0.5]]],
    'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
    'anchor_flip': True,
    'anchor_clip': True,
    # InterLayers
    'interlayers_normalizations': [],
    'interlayers_use_batchnorm': False,
    'interlayers_channels_kernels': [[[64,1],[128,3]],\
                                     [[64,1],[128,3]],\
                                     [[64,1],[128,3]],\
                                     ],
    #'interlayers_channels_kernels': [[[64,1],[128,3]],\
    #                                 [[64,1],[128,3]],\
    #                                 [[64,1],[128,3]]],
    # Loss layer
    # @types
    'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,# if len(ssd_1_gt_labels) > 1 else P.MultiBoxLoss.SOFTMAX,
    'bboxloss_use_dense_boxes': True if len(ssd_1_gt_labels) > 1 else False,
    # @weights
    'bboxloss_loc_weight': 4.0,
    'bboxloss_conf_weight': 1.0,
    # @Overlaps & boxsize threshold
    'bboxloss_overlap_threshold': 0.5,
    'bboxloss_neg_overlap': 0.3,
    'bboxloss_size_threshold': 0.0001,
    # @OHEM & FocusLoss
    'bboxloss_do_neg_mining': True,
    'bboxloss_neg_pos_ratio': 3,
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
    'flag_noperson':flag_noperson,
}
# ##############################################################################
# ---------------------------------SSD2 Params----------------------------------
ssd_Param_2 = {
    # FeatureLayers
    'feature_layers': ['featuremap1','featuremap2','featuremap3'],
# 'feature_layers': ['conv2_1_recon_relu','featuremap1','featuremap2','featuremap3'],
    # 'feature_layers': ['pool1_recon'],
    # num_classes
    'num_classes': len(ssd_2_gt_labels) + 1,
    'gt_labels': ssd_2_gt_labels,
    'target_labels': getTargetLabels(len(ssd_2_gt_labels)),
    'alias_id': ssd_2_gt_labels[0],
    # Anchors
    # 'anchor_boxsizes': [[0.05,0.1,0.15,0.25,0.35,0.45,0.6,0.75,0.95]],
    # 'anchor_aspect_ratios': [[[1,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5]]],
    # 'anchor_boxsizes': [[0.03,0.06,0.1],[0.08,0.12],[0.15,0.25,0.35],[0.45,0.6,0.75,0.95]],
    # 'anchor_aspect_ratios': [[[0.5],[0.5],[0.5]], \
    #                          [[0.5],[1,0.5]], \
    #                          [[1,0.5,0.25],  \
    #                           [1,0.5,0.25], \
		# 	                  [1,0.5,0.25]], \
    #                          [[1,0.5,0.25],  \
    #                           [1,0.5,0.25],\
		# 	                  [1,0.5,0.25],\
    #                           [1,0.5]]],
    # 'anchor_boxsizes': [[0.02,0.05]],
    # 'anchor_aspect_ratios': [[[1],[1]],\
    #                          ],

    # default parameters from zhangming----Begin
    'anchor_boxsizes': [[0.05,0.1],[0.15,0.25,0.35],[0.45,0.6,0.75,0.95]],
   # 'anchor_boxsizes': [[0.06,0.12],[0.18,0.24,0.32],[0.4,0.6,0.8,0.95]],#ZM Design
   # 'anchor_aspect_ratios': [[[0.5],[0.5]], \
   #                           [[0.5,],  \
   #                            [0.5,], \
	# 		                  [0.5,]], \
   #                           [[0.5,],  \
   #                            [0.5,],\
	# 		                  [0.5,],\
   #                            [0.5]]],
   # default parameters from zhangming----End
'anchor_aspect_ratios': [[[0.5],[0.5]], \
                             [[1,0.5,0.25],  \
                              [1,0.5,0.25], \
			                  [1,0.5,0.25]], \
                             [[1,0.5,0.25],  \
                              [1,0.5,0.25],\
			                  [1,0.5,0.25],\
                              [1,0.5]]],
# 'anchor_aspect_ratios': [[[0.56],[0.56]], \
#                              [[0.42,0.7],  \
#                               [0.42,0.7], \
# 			                  [0.42,0.7]], \
#                              [[0.42,0.7],  \
#                               [0.42,0.7],\
# 			                  [0.42,0.7],\
#                               [0.56]]],
    'anchor_prior_variance': [0.1, 0.1, 0.2, 0.2],
    'anchor_flip': True,
    'anchor_clip': True,
    # InterLayers
    'interlayers_normalizations': [],
    'interlayers_use_batchnorm': False,
    'interlayers_channels_kernels': [[[64,1],[128,3]],\
                                     [[64,1],[128,3]],\
                                     [[64,1],[128,3]],\
                                     ],
    # Loss layer
    # @types
    'bboxloss_loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'bboxloss_conf_loss_type': P.MultiBoxLoss.LOGISTIC,
    'bboxloss_use_dense_boxes': True,
    # @weights
    'bboxloss_loc_weight': 2.0,
    'bboxloss_conf_weight': 1.0,
    # @Overlaps & boxsize threshold
    'bboxloss_overlap_threshold': 0.5,
    'bboxloss_neg_overlap': 0.3,
    'bboxloss_size_threshold': 0.0001,
    'bboxloss_size_threshold_max': 5.0,
    'flag_forcematchallgt':False,
    'flag_areamaxcheckinmatch':True,
    # @OHEM & FocusLoss
    'bboxloss_do_neg_mining': True,
    'bboxloss_neg_pos_ratio': 3,
    'bboxloss_using_focus_loss': False,
    'bboxloss_focus_gama': 2,
    # unchanged
    'bboxloss_use_difficult_gt': False,
    'bboxloss_code_type': P.PriorBox.CENTER_SIZE,
    'bboxloss_normalization': P.Loss.VALID,
    # detection out
    'detout_target_labels': ssd_2_gt_labels,
    'detout_conf_threshold': 0.5,
    'detout_nms_threshold': 0.45,
    'detout_size_threshold': 0.0001,
    'detout_top_k': 200,
    'flag_noperson':flag_noperson,
    'flag_showdebug':False,
  
}

# ##############################################################################
# ---------------------------------Eval Params----------------------------------
eval_Param = {
    'eval_difficult_gt': False,
    'eval_boxsize_threshold': [0, 16e-4, 64e-4, 100e-4, 0.1, 0.15, 0.25],
    'eval_iou_threshold': [0.9, 0.75, 0.5],
    'eval_gt_labels': eval_gt_labels,
    'eval_num_classes': len(eval_gt_labels) + 1
}
