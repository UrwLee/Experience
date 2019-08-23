# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
#import inputParam
import os
import sys
import math
sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True
from DAPData import *
from DAPNet import *
from DAP_Param import *
# #################################################################################
caffe_root = "/home/{}/work/remodet_repository".format(USERNAME)
# --------------------------------Model Info--------------------------------------
Project = "DAPDet"
BaseNet = "ResNetPose"
Models = "RemoCoco"
Ver = "2SSD-MA2-OHEM-PLA-LLR-HFLR"
Specs = """
Base.
"""
Pretained_Model = "/home/zhangming/Models/Release/release1005/ResNet_Variant_WiMPI_8A_iter_300000.caffemodel"
Results_dir = "/home/{}/Models/Results".format(USERNAME)
# -----------------------------config for computation----------------------------
gpus = "2,3"
resume_training = True
test_initialization = False
# Not modified.
snapshot_after_train = True
remove_old_models = False
# --------------------------------solver Param-----------------------------------
batchsize_1050 = 1
batchsize_1080 = 24
batchsize_per_device = batchsize_1080
update_batchsize = 2 * batchsize_1080
test_batchsize = 1
train_max_itersize = 500000
base_lr =5e-4
weight_decay = 0.0005
lr_policy = "plateau" 
stepsize = 100000
stepvalue = [200000,350000,500000]
plateau_winsize = [25000, 25000, 25000]
gamma = 0.1
momentum = 0.9
snapshot = 20000
average_loss = 20
display = 20
solve_type = "SGD"
debug = False
test_interval = 10000
eval_type = "detection"
random_seed = 150
test_compute_loss = False
ap_version = "Integral"
usrname = "zhangming" if len(gpus.split(',')) > 1 else USERNAME
# ################################################################################
# --------------------------------DO NOT MODIFIED--------------------------------
# get gpus
def get_gpus():
    return gpus
# get num of gpus
def get_gpunums():
    return len(get_gpus().split(','))
# batchsize for diffrent gpu setting
def get_train_batchsize():
    return batchsize_per_device
# get update iters
def get_update_iter():
    return int(math.ceil(float(update_batchsize) / (batchsize_per_device * get_gpunums())))
# get solver mode of caffe
def get_solver_mode():
    if get_gpunums() > 0:
        return P.Solver.GPU
    else:
        return P.Solver.CPU
# get device id of root solve
def get_device_id():
    if get_gpunums() > 0:
        return int(get_gpus().split(',')[0])
    else:
        return -1
# get test batchsize
def get_test_batchsize():
    return test_batchsize

# get train max_iter_size
def get_train_max_iter():
    return train_max_itersize
def get_test_max_iter():
    return get_lines(val_list) / get_test_batchsize()
# get USERNAME
def get_username_real():
    if get_gpunums() == 2:
        return "zhangming"
    else:
        return USERNAME
# get num of samples of train/val
def get_lines(source):
    lines = 0
    f = open(source,"r")
    for line in f:
        lines += 1
    f.close()
    return lines
# get pretained Models
def get_pretained_model():
    return Pretained_Model
# --------------------------------Source Params-----------------------------------
# get solver param
def get_solver_param():
    return {
        # learning rate and update hyper-params
        'base_lr': base_lr,
        'weight_decay': weight_decay,
        'lr_policy': lr_policy,
        'stepsize': stepsize,
        'stepvalue':stepvalue,
        'gamma': gamma,
        'plateau_winsize': plateau_winsize,
        'momentum': momentum,
        'iter_size': get_update_iter(),
        'max_iter': get_train_max_iter(),
        'snapshot': snapshot,
        'display': display,
        'average_loss': average_loss,
        'type': solve_type,
        'solver_mode': get_solver_mode(),
        'device_id': get_device_id(),
        'debug_info': debug,
        'snapshot_after_train': snapshot_after_train,
        # Test parameters
        'test_interval': test_interval,
        'test_iter':[get_test_max_iter()],
        'test_net_type': [eval_type],
        'test_initialization': test_initialization,
        'test_compute_loss': test_compute_loss,
        'random_seed' : random_seed,
        'ap_version': ap_version,
    }
