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
from DetRelease_Data import *
from DetNet_Param import *
from DetRelease_General import *
# #################################################################################
caffe_root = "/home/{}/work/remodet_repository_DJ".format(USERNAME)
# --------------------------------Model Info--------------------------------------
trunc_ratio = 1.0
Project = "DetNet"
BaseNet = "DetNet_DarkNet20180519FromPose"
BaseNet = "tmp_new"
# Models = "2SSD-MA2-OHEM-PLA-LLR"
# Ver = "ReTr_J"
Models = ""
Ver = ""
# Models = "TrunkBD_PDHeadHand"
# Ver = "DataMinS0.75MaxS2.0NoExp_HisiDataOnlySmpBody_WD5e-3_1A"
# Models = "AIC_Person"
# Ver = "1A"
Specs = """
Base.
"""
Pretained_Model = "/home/zhangming/Models/PretainedModels/Pose_DarkNet2018515_HisiDataWD5e-3_512x288_iter_240000_merge.caffemodel"
Pretained_Model = "/home/zhangming/Models/PretainedModels/Release20180906_AllFeaturemap_WithParamName_V0.caffemodel"
Results_dir = "/home/{}/Models/Results".format(USERNAME)
# -----------------------------config for computation----------------------------
gpus = "0,1"
fine_tuning = True
resume_training = False
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
base_lr =1e-3
weight_decay = 5e-2
lr_policy = "plateau"
stepsize = 100000
stepvalue = [200000,350000,500000]
plateau_winsize = [25000, 25000, 25000]
gamma = 0.1
momentum = 0.9
snapshot = 5000
average_loss = 20
display = 20
# solve_type = "SGD"
solve_type = "Adam"
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
    lines = get_lines(val_list_bdpd,False)
    if isinstance(lines,list):
        return [l/get_test_batchsize() for l in lines]
    else:
        return lines / get_test_batchsize()
# get USERNAME
def get_username_real():
    if get_gpunums() == 2:
        return "zhangming"
    else:
        return USERNAME
# get num of samples of train/val
def get_lines(source,train=True):
    if train:
        if isinstance(source, list):
            lines = 0
            for si in source:
                f = open(si,"r").readlines()
                lines+=len(f)
        else:
            f = open(source, "r").readlines()
            lines = len(f)
    else:
        if isinstance(source, list):
            lines = []
            for si in source:
                f = open(si,"r").readlines()
                lines.append(len(f))
        else:
            f = open(source, "r").readlines()
            lines = len(f)
    return lines
def fromlist2string(lista,delimiter):
    stri = ""
    for i in xrange(len(lista)):
        if i == len(lista)-1:
            stri+=lista[i]
        else:
            stri += lista[i] + delimiter
    return stri
# get pretained Models
def get_pretained_model():
    if type(Pretained_Model) is list:
        return fromlist2string(Pretained_Model,',')
    else:
        return Pretained_Model
# --------------------------------Source Params-----------------------------------
# get solver param
def get_solver_param():
    test_iter = get_test_max_iter()
    if train_net_id == 0:
        test_iter = [test_iter,test_iter]
        test_net_type = [eval_type,eval_type]
    else:
        test_iter = [test_iter,]
        test_net_type = [eval_type,]
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
        'test_iter':test_iter,
        'test_net_type': test_net_type,
        'test_initialization': test_initialization,
        'test_compute_loss': test_compute_loss,
        'random_seed' : random_seed,
        'ap_version': ap_version,
    }