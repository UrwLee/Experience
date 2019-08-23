# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
import os
import sys
import math
from Data import *
from Net import *
sys.dont_write_bytecode = True
##################################################################################
caffe_root = "/home/ethan/work/remodet_repository"
# --------------------------------Model Info--------------------------------------
Project = "HPKeypoint"
BaseNet = "CNNAll64C"
Models = "Sigma5RT40"
Ver = "1A"
Specs = """
Test.
"""
Pretrained_Model = ''
Results_dir = "/home/ethan/Models/Results"
# -----------------------------config for computation----------------------------
gpus = "0"
resume_training = False
test_initialization = False
snapshot_after_train = True
remove_old_models = False
# --------------------------------solver Param-----------------------------------
update_batchsize = train_batchsize
train_max_itersize = 150000
base_lr = 1e-4
weight_decay = 0.0005
lr_policy = "step"
stepsize = 30000
stepvalue = [200000,350000,500000]
gamma = 0.1
momentum = 0.9
snapshot = 20000
average_loss = 20
display = 20
solve_type = "Adam"  # Adam / SGD
debug = False
test_interval = 5000
test_net_type = ["pose",]
random_seed = 150
test_compute_loss = True
# ################################################################################
# --------------------------------DO NOT MODIFIED--------------------------------
# get gpus
def get_gpus():
    return gpus
# get num of gpus
def get_gpunums():
    return len(get_gpus().split(','))
# get update iters
def get_update_iter():
    return int(math.ceil(float(update_batchsize) / (train_batchsize * get_gpunums())))
# get solver mode of caffe
def get_solver_mode():
    if get_gpunums() > 0:
        return P.Solver.GPU
    else:
        return P.Solver.CPU
# get device id of root solver
def get_device_id():
    if get_gpunums() > 0:
        return int(get_gpus().split(',')[0])
    else:
        return -1
def get_test_iter_for_project():
    return [get_test_iter()]
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
        'momentum': momentum,
        'iter_size': get_update_iter(),
        'max_iter': train_max_itersize,
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
        'test_iter':get_test_iter_for_project(),
        'test_net_type': test_net_type,
        'test_initialization': test_initialization,
        'test_compute_loss': test_compute_loss,
        'random_seed' : random_seed,
    }
