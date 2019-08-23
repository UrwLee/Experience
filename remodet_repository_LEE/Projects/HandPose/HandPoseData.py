# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
########################################################################################################


train_list = '/home/zhangming/Datasets/REMO_HandPose/Layout/handpose_train_V0.txt'
val_list = '/home/zhangming/Datasets/REMO_HandPose/Layout/handpose_test_V0.txt'
root_dir= "/home/zhangming/Datasets/REMO_HandPose/Images"
# batch_size
train_batchsize = 128
# resized dim
resized_height = 96
resized_width = 96
flag_clip = True
bbox_extend_min = 1.5
bbox_extend_max = 1.7
rotate_angle = 15
flag_augintrain = True

########################################################################################################
# 训练阶段增广参数

handpose_data_param_train = {
    'flip': True,
    'flip_prob': 0.5,
    'save': True,
    'save_path': '/home/ethan/DataSets/REMO_HandPose/ImagsVis',
    'resize_h': resized_height,
    'resize_w': resized_width,
    'source': train_list,
    'root_folder': root_dir,
    'bbox_extend_min': bbox_extend_min,
    'bbox_extend_max':bbox_extend_max,
    'shuffle': True,
    'rand_skip': 1000,
    'batch_size': train_batchsize,
    'flag_augintrain': flag_augintrain,
    'clip': flag_clip
}
# 测试阶段增广参数
handpose_data_param_val = {
    'flip': False,
    'flip_prob': 0.5,
    'save': False,
    'save_path': '/home/ethan/DataSets/REMO_HandPose/ImagsVis',
    'resize_h': resized_height,
    'resize_w': resized_width,
    'source': val_list,
    'root_folder': root_dir,
    'bbox_extend_min': bbox_extend_min,
    'bbox_extend_max':bbox_extend_max,
    'shuffle': False,
    'rand_skip': 1,
    'batch_size': 1,
    'clip': flag_clip
}
# 颜色修改参数　[Unused.] Default


transform_param = {
    'distort_param': {
                 'brightness_prob': 0.2,
                 'brightness_delta': 20,
                 'contrast_prob': 0.2,
                 'contrast_lower': 0.5,
                 'contrast_upper': 1.5,
                 'hue_prob': 0.2,
                 'hue_delta': 18,
                 'saturation_prob': 0.2,
                 'saturation_lower': 0.5,
                 'saturation_upper': 1.5,
                 'random_order_prob': 0,
             },
}
########################################################################################################
# 计算迭代次数
def get_lines(source):
    lines = open(source,"r").readlines()
    return len(lines)
def get_test_iter():
    return get_lines(val_list)
def get_train_iter_epoch():
    return get_lines(train_list) / train_batchsize
########################################################################################################
# 数据层以及前端
def getHandPoseDataLayer(net, train=True):
    # 数据读入层
    handpose_data_param = handpose_data_param_train if train else handpose_data_param_val
    if train:
        kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN'))}
    else:
        kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TEST'))}
    net.data, net.label = L.HandPoseData(name="data", handpose_data_param=handpose_data_param, transform_param=transform_param, ntop=2, **kwargs)
    return net
