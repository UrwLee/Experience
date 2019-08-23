# -*- coding: utf-8 -*-
from __future__ import print_function
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
import time
from solverParam import *
from HandPoseData import *
from HandPoseNet import *
sys.path.append('../')
from PyLib.Utils.path import *
sys.dont_write_bytecode = True
def fromlist2string(lista,delimiter):
    stri = ""
    for i in xrange(len(lista)):
        if i == len(lista)-1:
            stri+=lista[i]
        else:
            stri += lista[i] + delimiter
    return stri
def get_ModelsStrings(pretrainmodels):

    if type(pretrainmodels) is list:
        return fromlist2string(pretrainmodels,",")
    else:
        return pretrainmodels
# 训练
def HandPoseNet_Train():
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
	test_net_file = "{}/test.prototxt".format(proto_dir)
	solver_file = "{}/solver.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,ProjectName)
	job_file = "{}/train.sh".format(job_dir)
	################################################################################
	# TRAIN
	net = caffe.NetSpec()
	net = getHandPoseDataLayer(net, train=True)
	net = HandPoseResNet(net, train=True, lr=1, decay=1)
	# net = ReidReconNet(net, train=True,data_layer="data",label="label", use_bn=True,flag_finetune = False)
	with open(train_net_file, 'w') as f:
	        print('name: "{}_train"'.format(ProjectName), file=f)
	        print(net.to_proto(), file=f)
	################################################################################
	# TEST
	net = caffe.NetSpec()
	net = getHandPoseDataLayer(net, train=False)
	net = HandPoseResNet(net, train=False)
	# net = ReidReconNet(net, train=False, data_layer="data", label="label", use_bn=True, flag_finetune=False)
	with open(test_net_file,'w') as f:
		print('name: "{}_test"'.format(ProjectName), file=f)
		print(net.to_proto(),file = f)
	################################################################################
	# Solver
	test_net_files = [test_net_file]
	solver_param = get_solver_param()
	solver = caffe_pb2.SolverParameter( \
		train_net=train_net_file, test_net=test_net_files, \
		snapshot_prefix=snapshot_prefix, **solver_param)
	with open(solver_file, 'w') as f:
	        print(solver, file=f)
	################################################################################
	# CaffeModel & Snapshot
	max_iter = 0
	use_pretrain_model = False
	for f in os.listdir(model_dir):
		if f.endswith(".solverstate"):
			basename = os.path.splitext(f)[0]
			iter = int(basename.split("{}_iter_".format(ProjectName))[1])
			if iter > max_iter:
				max_iter = iter
	if Pretrained_Model != '':
		train_param = '--weights="{}" \\\n'.format(get_ModelsStrings(Pretrained_Model))
		use_pretrain_model = True
	if resume_training:
		if max_iter > 0:
			train_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
			use_pretrain_model = True
	################################################################################
	# Running Script
	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe train \\\n')
		f.write('--solver="{}" \\\n'.format(solver_file))
		if use_pretrain_model:
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
	HandPoseNet_Train()
