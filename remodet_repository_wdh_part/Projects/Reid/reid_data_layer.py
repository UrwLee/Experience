# -*- coding: utf-8 -*-
from username import USERNAME

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import sys
sys.dont_write_bytecode = True

net_input_width = 384
net_input_height = 384

def get_source_file():
    return '/home/%s/data/REID/dataset/Layout/train_list.txt' % USERNAME

def get_reidDataParam(batch_size=1):
    return {
        'xml_list': get_source_file(),
        'xml_root': '/home/%s/data/REID/dataset/' % USERNAME,
        'shuffle': True,
        'rand_skip': 1000,
        'batch_size': batch_size,
    }

def get_reidTransParam():
    return {
        'mirror': False,
        'mean_value': [104,117,123],
        'root_dir': '/home/%s/data/REID/dataset/' % USERNAME,
        'resized_width': net_input_width,
        'resized_height': net_input_height,
        'normalize': False,
        'visual': False,
        'save_dir': '/home/%s/data/REID/tmp/' % USERNAME
    }

def get_reidDataLayer(net, batch_size=1):
    kwargs = {
        # 'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        'reid_transform_param': get_reidTransParam(),
        }
    reid_data_kwargs = get_reidDataParam(batch_size=batch_size)
    net.data, net.label = L.ReidData(name="data", reid_data_param=reid_data_kwargs, \
                                     ntop=2, **kwargs)
    return net
