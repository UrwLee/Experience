# -*- coding: utf-8 -*-
from __future__ import print_function
from username import USERNAME
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
import math
import os
import shutil
import stat
import subprocess
import sys
import time
import sys

sys.path.append('../')
sys.dont_write_bytecode = True

from solverParam import *
from mask_data_layer import *
from MaskNet import *
from PyLib.NetLib.PoseNet import *
from PyLib.NetLib.mPoseNet import *
from PyLib.NetLib.RemNet import *
from PyLib.Utils.path import *
from test import *
def unifiedNet():
	time_postfix = time.strftime("%m-%d_%H-%M-%S",time.localtime())
	################################################################################
	os.chdir(caffe_root)
	################################################################################
	# work dir
	ProjectName = "{}_{}_{}".format(BaseNet,Models,Ver)
	work_dir = "{}/{}/{}".format(Results_dir,Project,ProjectName)
	make_if_not_exist(work_dir)
	################################################################################
        # work and model dirs
	proto_dir = "{}/Proto".format(work_dir)
	log_dir = "{}/Logs".format(work_dir)
	model_dir = "{}/Models".format(work_dir)
	pic_dir = "{}/Pics".format(work_dir)
	job_dir = "{}/Job".format(work_dir)
	make_if_not_exist(proto_dir)
	make_if_not_exist(log_dir)
	make_if_not_exist(model_dir)
	make_if_not_exist(pic_dir)
	make_if_not_exist(job_dir)
    ################################################################################
    # work file
	log_file = "{}/{}.log".format(log_dir,time_postfix)
	train_net_file = "{}/train.prototxt".format(proto_dir)
	det_eval_file="{}/det_eval.prototxt".format(proto_dir)
	pose_eval_file="{}/pose_eval.prototxt".format(proto_dir)
	mask_eval_file="{}/mask_eval.prototxt".format(proto_dir)
	test_net_file = "{}/test.prototxt".format(proto_dir)
	solver_file = "{}/solver.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,ProjectName)
	job_file = "{}/train.sh".format(job_dir)
	################################################################################
	# TRAIN
	# net = caffe.NetSpec()
	# net = get_UnifiedDataLayer(net,train=True,batchsize=batchsize_per_device)
	# net = MaskNet_Train(net)
	# with open(train_net_file, 'w') as f:
	#         print('name: "{}_train"'.format(ProjectName), file=f)
	#         print(net.to_proto(), file=f)
	net = caffe.NetSpec()
	net = get_UnifiedDataLayer(net,train=True,batchsize=batchsize_per_device)
	net = MTD_Train(net)
	with open(train_net_file, 'w') as f:
	        print('name: "{}_train"'.format(ProjectName), file=f)
	        print(net.to_proto(), file=f)
	################################################################################
	#eval
	net = caffe.NetSpec()
	net = get_UnifiedDataLayer(net,train=False,batchsize=1)
	net = MaskNet_Val_MTD(net)
	
	with open(det_eval_file,'w') as f:
		print('name: "{}_Val_Det"'.format(ProjectName), file=f)
		print(net.to_proto(),file = f)
	# net = caffe.NetSpec()
	# net = get_UnifiedDataLayer(net,train=False,batchsize=1)
	# net = MaskNet_Val_Det(net)
	
	# with open(det_eval_file,'w') as f:
	# 	print('name: "{}_Val_Det"'.format(ProjectName), file=f)
	# 	print(net.to_proto(),file = f)

	# net = caffe.NetSpec()
	# net = get_UnifiedDataLayer(net,train=False,batchsize=1)
	# net = MaskNet_Val_Pose(net)
	
	# with open(pose_eval_file,'w') as f:
	# 	print('name: "{}_Val_Pose"'.format(ProjectName), file=f)
	# 	print(net.to_proto(),file = f)

	# net = caffe.NetSpec()
	# net = get_UnifiedDataLayer(net,train=False,batchsize=1)
	# net = MaskNet_Val_Mask(net)
	
	# with open(mask_eval_file,'w') as f:
	# 	print('name: "{}_Val_Mask"'.format(ProjectName), file=f)
	# 	print(net.to_proto(),file = f)
	################################################################################
	#test
	test()
	################################################################################
	# Solver
	solver_param = get_solver_param()
	solver = caffe_pb2.SolverParameter(train_net=train_net_file, \
		  			test_net=[det_eval_file],snapshot_prefix=snapshot_prefix,**solver_param)
	with open(solver_file, 'w') as f:
	        print(solver, file=f)
	################################################################################
	# CaffeModel & Snapshot
	max_iter = 0
	for file in os.listdir(model_dir):
		if file.endswith(".solverstate"):
			basename = os.path.splitext(file)[0]
			iter = int(basename.split("{}_iter_".format(ProjectName))[1])
			if iter > max_iter:
				max_iter = iter
	train_param = '--weights="{}" \\\n'.format(get_pretained_model())
	if resume_training:
		if max_iter > 0:
			train_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
	################################################################################
	# scripts
	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe train \\\n')
		f.write('--solver="{}" \\\n'.format(solver_file))
		f.write(train_param)
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(get_gpus(), log_file))
		else:
			f.write('2>&1 | tee {}.log\n'.format(log_file))
	os.chmod(job_file, stat.S_IRWXU)
        # ==========================================================================
        # Training
        subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    unifiedNet()
