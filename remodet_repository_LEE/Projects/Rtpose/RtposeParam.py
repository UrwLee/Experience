# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
# ----------------------- -------RtposeParam----------------------------------
pose_coco_kwargs = {
    # resize
    'resize_factor': 2,
    'resize_scale_gap': 0.3,
    'resize_start_scale': 1.0,
    # nms
    'nms_threshold': 0.05,
    'nms_max_peaks': 100,
    'nms_num_parts': 18,
    # connect
    'conn_is_type_coco': True,
    'conn_max_person': 10,
    'conn_max_peaks_use': 100,
    'conn_iters_pa_cal': 10,
    'conn_connect_inter_threshold': 0.05,
    'conn_connect_inter_min_nums': 8,
    'conn_connect_min_subset_cnt': 3,
    'conn_connect_min_subset_score': 0.4,
    # visual
    # POSE / HEATMAP_ID / HEATMAP_FROM / VECMAP_ID / VECMAP_FROM
    'visual_type': P.Visualizepose.POSE,
    'visual_visualize': False,
    'visual_draw_skeleton': False,
    'visual_print_score': False,
    'visual_part_id': 0,
    'visual_from_part': 0,
    'visual_vec_id': 0,
    'visual_from_vec': 0,
    'visual_pose_threshold': 0.01,
    'visual_write_frames': False,
    'visual_output_directory': "",
}
