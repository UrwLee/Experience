# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
# ----------------------- -------Param----------------------------------
def get_poseTestParam():
    return {
    # nms
    'nms_threshold': 0.05,
    'nms_max_peaks': 500,
    'nms_num_parts': 18,
    # connect
    'conn_is_type_coco': True,
    'conn_max_person': 10,
    'conn_max_peaks_use': 20,
    'conn_iters_pa_cal': 10,
    'conn_connect_inter_threshold': 0.05,
    'conn_connect_inter_min_nums': 8,
    'conn_connect_min_subset_cnt': 3,
    'conn_connect_min_subset_score': 0.4,
    # visual
    'eval_area_thre': 64*64,
    'eval_oks_thre': [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9],
    }
