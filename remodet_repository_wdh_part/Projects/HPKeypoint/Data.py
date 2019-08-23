# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import sys
sys.dont_write_bytecode = True
########################################################################################################
# 数据文件
train_list = ['/home/zhangming/Datasets/HandKeypoints/Layout/train.txt','/home/zhangming/Datasets/HandKeypoints/REMOCap_HandKeypoint_20180817/Layout/trainval.txt']
val_list = '/home/zhangming/Datasets/HandKeypoints/Layout/val.txt'
# 图像路径
root_dir_train = ['/home/zhangming/Datasets/HandKeypoints/pics','/home/zhangming/Datasets/HandKeypoints/REMOCap_HandKeypoint_20180817']
root_dir_val = '/media/ethan/RemovableDisk/Datasets/HandKeypoints'

train_list = '/media/ethan/RemovableDisk/Datasets/HandKeypoints/Layout/train.txt'
val_list = '/media/ethan/RemovableDisk/Datasets/HandKeypoints/Layout/val.txt'
root_dir_train = "/media/ethan/RemovableDisk/Datasets/HandKeypoints/pics"
root_dir_val = "/media/ethan/RemovableDisk/Datasets/HandKeypoints/pics"
# 图像路径

# batch_size
train_batchsize = 32
test_batchsize = 32
# resized dim
resized_height = 256
resized_width = 256
target_stride = 4
sigma = 5.0
num_keypoints = 20
num_limbs = 19
max_rotate_degree = 40
########################################################################################################
# 训练阶段增广参数
hpkeypoint_data_param_train = {
    'xml_list_multiple': train_list,
    'xml_root_multiple': root_dir_train,
    'batch_size': train_batchsize,
    'rand_skip': 1000,
    'shuffle': True,
    'mean_value': [104, 117, 123],
}
# 测试阶段增广参数
hpkeypoint_data_param_val = {
    'xml_list': val_list,
    'batch_size': test_batchsize,
    'rand_skip': 1,
    'shuffle': False,
    'xml_root': root_dir_val,
'mean_value': [104, 117, 123],
}
# Trans
trans_param = {
    'min_scale': 1.5,
    'max_scale': 2.0,
    'drift_scalar': 0.05,
    'max_rotate_degree':max_rotate_degree,
    'resized_width': resized_width,
    'resized_height': resized_height,
    # 'rotate_angle_max':15,
    "stride": target_stride,
    "sigma": sigma,
    'flip_prob': 0.5,
    'dis_param': {
        'brightness_prob': 0.5,
        'brightness_delta': 16,
        'contrast_prob': 0.5,
        'contrast_lower': 0.75,
        'contrast_upper': 1.25,
        'hue_prob': 0.5,
        'hue_delta': 10,
        'saturation_prob': 0.5,
        'saturation_lower': 0.75,
        'saturation_upper': 1.25,
        'random_order_prob': 0,
    }
}
########################################################################################################
# 计算迭代次数
def get_lines(source):
    lines = 0
    f = open(source,"r")
    for line in f:
        lines += 1
    f.close()
    return lines
def get_test_iter():
    return get_lines(val_list) / test_batchsize + 1
def get_train_iter_epoch():
    return get_lines(train_list) / train_batchsize + 1
########################################################################################################
# 数据层以及前端
def getHPDataLayer(net, train=True):
    # 数据读入层
    hpkeypoint_data_param = hpkeypoint_data_param_train if train else hpkeypoint_data_param_val
    if train:
        kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN'))}
    else:
        kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TEST'))}
    net.data, net.label = L.BBoxDataHandKeypoint(name="data", unified_data_param=hpkeypoint_data_param, unified_data_transform_param=trans_param, ntop=2, **kwargs)
    return net
