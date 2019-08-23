# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True

sys.path.append('../')
from PyLib.LayerParam.ImageDataLayerParam import *

# 656 x 368 (41 x 23) -> (33x19)
# 528 x 304
net_width = 32*16
net_height = 32*9

# CAM编号
webcam_index = 0
# 视频列表
video_list = [
    '/home/zhangming/video/1.mkv',
    '/home/zhangming/video/FigureSkating.mp4',
    # '/home/zhangming/video/SpeedSkating.mp4',
    # '/home/zhangming/video/3.mp4',
    # '/home/zhangming/video/KongfuYoga.mp4',
    # '/home/zhangming/video/YogaBeauty.mp4',
    # '/home/zhangming/video/SpecialIdentity.mp4',
    # '/home/zhangming/video/FigureSkating1.mp4',
    # '/home/zhangming/video/FigureSkating2.mp4',
]
# 视频编号
video_file = video_list[1]

# 视频源： webcam / video
# image_source = "video"
image_source = "video"
# 测试迭代
test_iterations = 1000000

# loc -> CORNER
VideoframeParam = {
    'video_type': P.Videoframe.WEBCAM if image_source == "webcam" else P.Videoframe.VIDEO,
    'device_id': webcam_index,
    'video_file': video_file,
    'webcam_width': 1280,
    'webcam_height': 720,
    'initial_frame': 0,
    # NOTE: normalize is important for model, please check if it is right configured.
    'normalize': False,
    'mean_value': [104,117,123],
}

VideotransformParam = {
    'resize_param': get_resize_param(train=False, \
                                     resize_width=net_width, \
                                     resize_height=net_height),
}
