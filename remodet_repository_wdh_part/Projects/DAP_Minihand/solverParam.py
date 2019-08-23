# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
# sys.path.insert(0, "/home/zhangming/work/minihand/remodet_repository/python")
sys.path.insert(0, '/home/xjx/work/remodet_repository_DJ/python')
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
#import inputParam
import os
import math
sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True
from DAPData import *
from DAPNet import *
from DAP_Param import *
# #################################################################################
caffe_root = "/home/zhangming/work/remodet_repository_DJ"
# --------------------------------Model Info--------------------------------------
Project = "DetNet_Minihand"
BaseNet = "DetMiniHand_R20180809_9_16_Conv4_5Deconv64EltSumConv2Hand_Two32InterNoBN"
Models = "B48"
Ver = "DAICREMO20180530REMO20180807_Dist0.5SmallValue_20180815"
Specs = """
Test Only.
"""
# Pretained_Model = "/home/zhangming/Models/FEMRelease/Release20180129/Base.caffemodel"
Pretained_Model = "/home/zhangming/Models/PretainedModels/DetPose_JtTr_DarkNet20180514MultiScaleNoBN_FTFromPose_TrunkBD_PDFaceHandLossWMult0.25_DataMinS0.75MaxS2.0NoExp9_16NoPer_OnlySmpBody_WD5e-3_20180804_iter_500000.caffemodel"
Results_dir = "/home/zhangming/Models/Results"
# -----------------------------config for computation----------------------------
gpus = "0,1"
fine_tuning = True   # NOTE
resume_training = False
test_initialization = False
# Not modified.
snapshot_after_train = True
remove_old_models = False
# --------------------------------solver Param-----------------------------------
batchsize = batch_size
batchsize_per_device = batchsize
update_batchsize = 1 * batchsize_per_device
test_batchsize = 1
train_max_itersize = 300000
base_lr = 1e-3
weight_decay = 0.0005
lr_policy = "step"
stepsize = 50000
stepvalue = [200000,350000,500000]
plateau_winsize = [20000, 20000, 20000]
gamma = 0.1
momentum = 0.9
snapshot = 20000
average_loss = 20
display = 20
solve_type = "Adam"
debug = False
test_interval = 5000
eval_type = "minihand"
random_seed = 150
test_compute_loss = True
ap_version = "Integral"
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
# get num of samples of train/val
def get_lines(source):
    lines = 0
    if isinstance(source, list):
        for si in source:
            f = open(si,"r")
            for line in f:
                lines += 1
            f.close()
    else:
        f = open(source, "r")
        for line in f:
            lines += 1
        f.close()
    return lines
# get pretained Models
def get_pretained_model():
    return Pretained_Model
# --------------------------------Source Params---------------------------------
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
