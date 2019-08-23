# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from google.protobuf import text_format
import filecmp
import math
import os
import shutil
import stat
import subprocess
import sys
import time
import shutil
import difflib
import json

sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True

import solverParam
from solverParam import get_image_data_param, get_detector_param, get_detector_type
from solverParam import BaseNet, get_pretained_model, Results_dir
from solverParam import DataSets, DetMethod, InputScale, ModelLevel, Specs
from solverParam import caffe_root

from PyLib.LayerParam.ImageDataLayerParam import *
from PyLib.NetLib.SsdDetector import *
from PyLib.NetLib.YoloDetector import *
from PyLib.NetLib.ImageDataLayer import *
from PyLib.Utils.dict import *
from PyLib.Utils.data import *
from PyLib.Utils.path import *

from scoreParam import model_idx, caffemodel_index, ap_version, merge

def score(Base_network=BaseNet,Datasets=DataSets,Methods=DetMethod,Input_scale=InputScale,Model_level=ModelLevel):
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}/{}/{}/{}".format(Results_dir,Base_network,Datasets,Methods,Input_scale,Model_level)
	work_dir = "{}/choosen".format(Results_dir)
	################################################################################
	# get model index
	model_index = model_idx
	if model_index == -1:
		# search the max id
		for filedir in os.listdir(work_dir):
			if filedir.isdigit():
				idx = int(filedir)
				if idx > model_index:
					model_index = idx
	if model_index < 0:
		raise ValueError("No model is found.")
	################################################################################
	# work and model dirs
	if merge:
		work_model_dir = "{}/{}/Merge".format(work_dir,model_index)
	else:
		work_model_dir = "{}/{}".format(work_dir,model_index)
	proto_dir = "{}/Proto".format(work_model_dir)
	model_dir = "{}/Models".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
	make_if_not_exist(job_dir)
	################################################################################
	# work file
	# score_train_file = "{}/score_train.prototxt".format(proto_dir)
	# score_test_file = "{}/score_test.prototxt".format(proto_dir)
	# solver_file = "{}/score_solver.prototxt".format(proto_dir)
	score_model_file = "{}/test.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,model_index)
	job_file = "{}/{}_score.sh".format(job_dir, model_index)
	################################################################################
	# 定义模型名称
	project_name = "{}-{}-{}-{}-{}-{}".format(Base_network,Datasets,Methods,Input_scale,Model_level,model_index)
	################################################################################
	# 训练模型
	max_iter = 0
	for file in os.listdir(model_dir):
		if file.endswith(".caffemodel"):
			basename = os.path.splitext(file)[0]
			iter = int(basename.split("{}_iter_".format(model_index))[1])
			if iter > max_iter:
				max_iter = iter
	if max_iter == 0:
		raise ValueError("not found .caffemodel in directory: {}".format(model_dir))
	if caffemodel_index > 0 and caffemodel_index < max_iter:
		max_iter = caffemodel_index
	out_file = "{}/{}_{}_score_results.txt".format(work_model_dir,max_iter,ap_version)
	################################################################################
	train_param = '--weights="{}_iter_{}.caffemodel" \\\n'.format(snapshot_prefix, max_iter)
	solver_param = solverParam.get_solver_param()
	# create running file
	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe score \\\n')
		f.write('--model="{}" \\\n'.format(score_model_file))
		f.write('--iterations="{}" \\\n'.format(solverParam.get_test_max_iter()))
		f.write('--ap_version="{}" \\\n'.format(ap_version))
		f.write(train_param)
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), out_file))
		else:
			f.write('2>&1 | tee {}.log\n'.format(out_file))
	os.chmod(job_file, stat.S_IRWXU)
	subprocess.call(job_file, shell=True)

if __name__ == "__main__":
	score()
