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
from pose_data_layer import *
from pose_test_param import *
# #################################################################################
caffe_root = "/home/{}/work/repository_pose".format(USERNAME)
# --------------------------------Model Info--------------------------------------
Project = "DetPose_JointTrain "
BaseNet = "DarkNet"
# Models = "2SSD-MA2-OHEM-PLA-LLR"
# Ver = "ReTr_J"
Models = "Base2.0G"
Ver = "20180418"
Specs = """
Use MPII+COCO.
"""
Pretrained_Models_dir = "/home/{}/Models/PretainedModels".format(USERNAME)
Results_dir = "/home/{}/Models/Results".format(USERNAME)
# Pretrained_Models = "ForSmallFeatMap17A_SmallFeatMap17REA6e4AsBase_15F5e5AsStage.caffemodel"
# Pretrained_Models = "ForRes2REB_Res2REA14e4AsStudentBase_15F5e5AsTeacher.caffemodel"
Pretrained_Models = ""
# DarkNet_MultiStage_WiMPI_Tea8B_iter_420000
# Pretrained_Models = "DarkNet_MultiStage_WiMPI_Tea4A_iter_500000.caffemodel"
# Pretrained_Models = ["yolo2.caffemodel",]

# -----------------------------config for computation----------------------------
gpus = "0,1"
flag_TX2_global = True
snapshot_after_train = True
test_initialization = False
run_soon = True
resume_training = False
remove_old_models = False
fine_tuning = True
Finetune_layer = 1
Finetune_sublayer = 1
# --------------------------------solver Param-----------------------------------
batchsize_1050 = 1
batchsize_1080 = 20
batchsize_per_device = batchsize_1080
#update_batchsize = max(len(gpus.split(',')) * batchsize_per_device * 2, 20)
update_batchsize = 40 #batchsize_per_device * 4
test_batchsize = 1
train_max_itersize = 500000
base_lr = 5e-5
weight_decay = 5e-5
lr_policy = "step"
stepsize = 100000
stepvalue = [50000, 150000,300000]
# stepvalue = [150000,300000, 50000]
plateau_winsize = [30000, 50000, 50000]
gamma = 0.15
momentum = 0.9
snapshot = 20000
display = 20
average_loss = 20
solve_type = "Adam"
debug = False
test_interval = 10000
eval_type = "pose"
test_net_type = ["pose",]
random_seed = 150
test_compute_loss = True
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
# get device id of root solver
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
def fromlist2string(lista,delimiter):
    stri = ""
    for i in xrange(len(lista)):
        if i == len(lista)-1:
            stri+=lista[i]
        else:
            stri += lista[i] + delimiter
    return stri
def get_ModelsStrings(pretrainmodels):
    print(pretrainmodels)
    if type(pretrainmodels) is list:
        pretrainmodels = [os.path.join(Pretrained_Models_dir,model) for model in pretrainmodels]
        return fromlist2string(pretrainmodels,",")
    else:
        return os.path.join(Pretrained_Models_dir,pretrainmodels)
# get pretained Models
def get_pretained_model():
    if BaseNet == "VGG":
        if not Pretrained_Models:
            return "{}/VGG/VGG_ILSVRC_16_layers_fc_reduced.caffemodel".format(Pretrained_Models_dir)
        else:
            return "{}/{}".format(Pretrained_Models_dir,Pretrained_Models)
    elif BaseNet == "Res50":
        return "{}/ResNet/ResNet-50-model.caffemodel".format(Pretrained_Models_dir)
    elif BaseNet == "Res101":
        return "{}/ResNet/ResNet-101-model.caffemodel".format(Pretrained_Models_dir)
    elif BaseNet == "PVA":
        return "{}/PVA/pva9.1_pretrained.caffemodel".format(Pretrained_Models_dir)
    elif BaseNet == "DarkNet":
        if not Pretrained_Models:
            if not Finetune_layer == 0:
                return "{}/DarkNet/yolo2_conv{}_{}.caffemodel".format(Pretrained_Models_dir, Finetune_layer,
                                                                      Finetune_sublayer)
            else:
                return "{}/DarkNet/yolo2.caffemodel".format(Pretrained_Models_dir)
        else:
            model_strs = ""
            for id_model, modeli in enumerate(Pretrained_Models):

                if id_model != 0:
                    model_strs += ","
                model_strs += '%s/DarkNet/%s' % (Pretrained_Models_dir, modeli)
            return model_strs

    elif BaseNet == "VGG19":
        return "{}/VGG/VGG_ILSVRC_19_layers.caffemodel".format(Pretrained_Models_dir)
    elif Pretrained_Models:
        model_strs = get_ModelsStrings(Pretrained_Models)
        return model_strs
    else:
        raise ValueError("Only VGG(16/19)/ResNet(50/101/152)/PVA/DarkNet are supported.")
# --------------------------------Source Params-----------------------------------
# get test max_iter_size
def get_test_max_iter():
    return get_lines(get_source_file(train=False)) / get_test_batchsize()
# get solver param
def get_solver_param():
    if not use_trainval:
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
            'test_iter': [get_test_max_iter()],
            'test_interval': test_interval,
            'test_net_type': test_net_type,
            'eval_type': eval_type,
            'test_initialization': test_initialization,
            'test_compute_loss': test_compute_loss,
            'random_seed' : random_seed,
            # 'snapshot_format':"HDF5"
        }
    else:
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
            # 'snapshot_format': "HDF5"
        }
