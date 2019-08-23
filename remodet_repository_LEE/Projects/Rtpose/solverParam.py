# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe import params as P
from google.protobuf import text_format
#import inputParam
import os
import sys
import math
sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True
# #################################################################################
caffe_root = "/home/{}/work/repository".format(USERNAME)
# Projects name
Project = "RtPose"
ProjectName = "Rtpose_COCO"
Results_dir = "/home/{}/Models/Results".format(USERNAME)
# Pretrained_Model = "/home/{}/Models/PoseModels/pose_iter_440000.caffemodel".format(USERNAME)
# Pretrained_Model = "/home/{}/Models/PoseModels/VGG19_3S_0_iter_20000.caffemodel".format(USERNAME)
Pretrained_Model = "/home/{}/Models/PoseModels/DarkNet_3S_0_iter_450000_merge.caffemodel".format(USERNAME)
gpus = "0"
solver_mode = P.Solver.GPU
