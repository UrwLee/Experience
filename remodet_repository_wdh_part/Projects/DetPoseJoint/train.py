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
# from solverParam_pose import *
from solverParam import *
from DAPData import *
from DAPNet import *
from DAP_Param import *
from PyLib.Utils.path import *
from mPoseNet_ResidualNet import *
# from pose_data_layer import *
# from pose_test_param import *


def DAPNet_Train():
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
	# pic_dir = "{}/Pics".format(work_dir)
	job_dir = "{}/Job".format(work_dir)
	make_if_not_exist(proto_dir)
	make_if_not_exist(log_dir)
	make_if_not_exist(model_dir)
	# make_if_not_exist(pic_dir)
	make_if_not_exist(job_dir)
    ################################################################################
    # work file
	log_file = "{}/{}.log".format(log_dir,time_postfix)
	train_net_file = "{}/train_DetNOPARTSAndPose_ShareParam".format(proto_dir)
	test_net_file = "{}/test_DetNOPARTSAndPose_ShareParam".format(proto_dir)
	solver_file = "{}/solver.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,ProjectName)
	job_file = "{}/train.sh".format(job_dir)
	################################################################################
	# TRAIN
	net = caffe.NetSpec()
	net=get_DAPDataLayer(net,train=True,batchsize_det=batchsize_det,batchsize_pose=batchsize_pose)
	#det
	# net = get_detDAPDataLayer(net,train=True,batchsize=batchsize_det)
	net = DAPPoseNet(net, train=True, data_layer="data", gt_label="label",\
				net_width=resized_width, net_height=resized_height)
	#pose

	# net = get_poseDataLayer(net,train=True,batch_size=batchsize_pose)
	# net_base = None
	# net_stage = None
	# pose_test_kwargs = get_poseTestParam()

	# net= mPoseNet_COCO_ShuffleVariant_PoseFromReconBase_Train(net, data_layer="data_pose", label_layer="label_pose", train=True, **pose_test_kwargs)
	
	with open(train_net_file, 'w') as f:
	        print('name: "{}_train"'.format(ProjectName), file=f)
	        print(net.to_proto(), file=f)
	################################################################################
	# # TEST
	net = caffe.NetSpec()
	# #det
	net=get_DAPDataLayer(net,train=False,batchsize_det=1,batchsize_pose=1)

	# net = get_detDAPDataLayer(net,train=False,batchsize=1)
	net = DAPPoseNet(net, train=False, data_layer="data", gt_label="label", \
				net_width=resized_width, net_height=resized_height)
	#pose
	# net = get_poseDataLayer(net,train=False)
	# pose_test_kwargs = get_poseTestParam()
	# net = mPoseNet_COCO_ShuffleVariant_PoseFromReconBase_Train(net, data_layer="data_pose", label_layer="label_pose", train=False, **pose_test_kwargs)
	with open(test_net_file,'w') as f:
		print('name: "{}_test"'.format(ProjectName), file=f)
		print(net.to_proto(),file = f)
	################################################################################
	# Solver
	solver_param = get_solver_param()
	solver = caffe_pb2.SolverParameter(train_net=train_net_file.replace(USERNAME,'zhangming'), \
		  			test_net=[test_net_file.replace(USERNAME,'zhangming')],snapshot_prefix=snapshot_prefix.replace(USERNAME,'zhangming'),**solver_param)
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
	train_param = None
	if len(get_pretained_model())>0:
		train_param = '--weights="{}" \\\n'.format(get_pretained_model())


	if resume_training:
		if max_iter > 0:
			train_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
	################################################################################
	# job scripts
	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root).replace(USERNAME,'zhangming'))
		f.write('./build/tools/caffe train \\\n')
		f.write('--solver="{}" \\\n'.format(solver_file).replace(USERNAME,'zhangming'))
		if not train_param is None:
			f.write(train_param)
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(get_gpus(), log_file).replace(USERNAME,'zhangming'))
		else:
			f.write('2>&1 | tee {}.log\n'.format(log_file).replace(USERNAME,'zhangming'))
	os.chmod(job_file, stat.S_IRWXU)
        # ==========================================================================
        # Training
        subprocess.call(job_file, shell=True)


if __name__ == "__main__":
    DAPNet_Train()
