# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True

sys.path.append('../')

from PyLib.LayerParam.ImageDataLayerParam import *

# -1: 自动搜索模型编号
# 0... -> 指定模型编号
model_idx = -1
# -1: 自动搜索最近的模型文件
# 0... -> 指定模型文件
caffemodel_index = -1
# CAM编号
webcam_index = 0
# 视频列表
video_list = ["your video"]
# 视频编号
video_file = video_list[0]

# 视频源： webcam / video
image_source = "webcam"

# 测试迭代
test_iter_only = 1000000

# 检测参数
detParam = {
    'conf_threshold': 0.01,
    'nms_threshold': 0.45,
    'boxsize_threshold': 0.001,
    'top_k': 200,
    'visualize': True,
    'visual_conf_threshold': 0.5,
    'visual_size_threshold': 0.05,
    'display_maxsize': 1000,
    'line_width': 4,
    'color': [[0,255,0]],
}

# 视频输入参数
def get_videoinput_transparam(resize_width,resize_height,mean_values):
    return {
        'mean_value': mean_values,
        'resize_param': get_resize_param(train=False, \
                                         resize_width=resize_width, \
                                         resize_height=resize_height),
    }
