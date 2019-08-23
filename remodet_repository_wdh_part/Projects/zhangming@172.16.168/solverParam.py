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
trunc_ratio = 0.999
Project = "DAPDet"
BaseNet = "DarkNet"
# Models = "2SSD-MA2-OHEM-PLA-LLR"
# Ver = "ReTr_J"
Models = "TrunkBD_PDHeadHand_NonCat"
Ver = "Exp2.5_1A_trunck{}".format(trunc_ratio)
# Models = "AIC_Person"
# Ver = "H"
Specs = """
Base.
"""
# Pretained_Model = ["/home/zhangming/Models/PretainedModels/ResNetPose_RemoCoco_2SSD-MA2-OHEM-PLA-LLR_iter_500000.caffemodel","/home/zhangming/Models/PretainedModels/ResNet_Variant_WiMPI_8A_FTFromDet_H_iter_80000.caffemodel"]
# Pretained_Model = "/home/zhangming/Models/Results/Det_JointTrain/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H/Models/ResNetPoseDet_JointTrain_I_L_WithFaceHeadHand_1H_iter_380000.caffemodel"
# Pretained_Model = "/home/zhangming/Models/PretainedModels/ResNetPoseDet_JointTrain_I_L_iter_500000_ChangeNameOneBaseNet.caffemodel"
Pretained_Model="/home/zhangming/Models/PretainedModels/DarkNet_TrunkBD_PDHeadHand_NonCat_1A_iter_220000_merge.caffemodel"
Results_dir = "/home/{}/Models/Results".format(USERNAME)
# -----------------------------config for computation----------------------------
gpus = "0,1"
fine_tuning = True
resume_training = False
test_initialization = False
# Not modified.
snapshot_after_train = True
remove_old_models = False


def get_TruncValues(caffemodel):
    net = caffe_pb2.NetParameter()
    f = open(caffemodel,'rb')
    net.ParseFromString(f.read())
    f.close()
    layers = net.layer
    truncvalues = {}
    for layer in layers:
        layer_name = layer.name
        layer_type = layer.type
        if layer_type != 'Convolution':
            continue
        weights = layer.blobs
        idx = 1
        for weight in weights:
            if idx == 1:
                w = weight.data
                w = np.abs(w).reshape((-1, ))
                w.sort()# in ascending order
                idx_start = int(len(w)*(trunc_ratio))
                truncvalues[layer_name] = w[idx_start]
            idx = idx + 1
    return truncvalues
if trunc_ratio<1.0:
    truncvalues = get_TruncValues(Pretained_Model)
else:
    truncvalues = {}
# --------------------------------solver Param-----------------------------------
batchsize_1050 = 1
batchsize_1080 = 24
batchsize_per_device = batchsize_1080
update_batchsize = 2 * batchsize_1080
test_batchsize = 1
train_max_itersize = 500000
base_lr =1e-3
weight_decay = 0.0005
lr_policy = "plateau" 
stepsize = 100000
stepvalue = [200000,350000,500000]
plateau_winsize = [25000, 25000, 25000]
gamma = 0.1
momentum = 0.9
snapshot = 5000
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
