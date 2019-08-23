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

from PyLib.NetLib.SsdDetector import *
from PyLib.NetLib.YoloDetector import *
from PyLib.NetLib.Yolo_SsdDetector import Yolo_SsdDetector
from PyLib.NetLib.ImageDataLayer import *
from PyLib.Utils.path import *
from PyLib.Utils.par import *
from PyLib.Utils.dict import *
from PyLib.Utils.data import *

def train(Base_network=BaseNet,Datasets=DataSets,Methods=DetMethod,Input_scale=InputScale,Model_level=ModelLevel):
	################################################################################
	# time postfix
	time_postfix = time.strftime("%m-%d_%H-%M-%S",time.localtime())
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# creat work dir
	work_dir = "{}/{}/{}/{}/{}/{}".format(Results_dir,Base_network,Datasets,Methods,Input_scale,Model_level)
	################################################################################
        make_if_not_exist(work_dir)
	# get the model params data
	model_params_dir = "{}/Models_Params".format(work_dir)
	make_if_not_exist(model_params_dir)
	model_params_file = "{}/Model_Params_dict.txt".format(model_params_dir)

	if check_if_exist(model_params_file):
		with open(model_params_file, 'r') as f:
			for line in f:
				model_params_data = json.loads(line)
				break
	else:
		model_params_data = {"state":{"index":-1,"cover_keys":[]}}

	current_params = solverParam.get_all_params()
	current_params = json.dumps(current_params)
	current_params = json.loads(current_params)

	flag_new_model = True
	for index,itemvalue in model_params_data.items():
		compare_params = itemvalue
		if index == "state":
			continue
		if cmp_dict(current_params,compare_params):
			model_index = index
			flag_new_model = False
			break
        if flag_new_model:
            print("A new model with updated params is created.")
        else:
            print("Use existing model.")
	operation_file = "{}/Operation.txt".format(work_dir)
	f_operation = open(operation_file,"a+")

	# change the model name from type str to type int
	model_params_data_change = {}
	for index,itemvalue in model_params_data.items():
		if index == "state":
			model_params_data_change[index] = itemvalue
		else:
			model_params_data_change[int(index)] = itemvalue
	model_params_data = model_params_data_change

	if flag_new_model:
		model_index = model_params_data["state"]["index"] + 1
		model_params_data["state"]["index"] = model_index
		model_params_data[model_index] = current_params
		current_keys = current_params.keys()
		cover_keys = list(set(current_keys + model_params_data["state"]["cover_keys"]))
		model_params_data["state"]["cover_keys"] = cover_keys
		with open(model_params_file, 'w') as f:
			f.write(json.dumps(model_params_data))
		cmp_index = model_index - 1
		if cmp_index in model_params_data:
			cmp_params = model_params_data[cmp_index]
			diff_keys,same_keys = get_diff_keys(cover_keys,cmp_params,current_params)
			key_list = diff_keys+same_keys
		else:
			key_list = current_keys
		model_params_csv_file = "{}/Model_Params_table.csv".format(model_params_dir)

		get_model_params_csv(key_list,model_params_data,model_params_csv_file)
		f_operation.write("{}  Create a new model: {}\n".format(time_postfix,model_index))
		print ("{}  Create a new model: {}\n".format(time_postfix,model_index))
	else:
		f_operation.write("{}  Rerun an old model: {}\n".format(time_postfix,model_index))
		print ("{}  Rerun an old model: {}\n".format(time_postfix,model_index))
	f_operation.close()

	#return 0

	work_model_dir = "{}/{}".format(work_dir,model_index)
	proto_dir = "{}/Proto".format(work_model_dir)
	log_dir = "{}/Logs".format(work_model_dir)
	model_dir = "{}/Models".format(work_model_dir)
	pic_dir = "{}/Pics".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
	param_dir = "{}/Params".format(work_model_dir)

	make_if_not_exist(proto_dir)
	make_if_not_exist(model_dir)
	make_if_not_exist(job_dir)
	make_if_not_exist(log_dir)
	make_if_not_exist(pic_dir)
	make_if_not_exist(param_dir)

	################################################################################
	if flag_new_model:
		params_file = "{}/params.txt".format(param_dir)
		with open(params_file,"w") as f:
			f.write(json.dumps(current_params))
	################################################################################
	# 创建完后直接开始运行
	run_soon = solverParam.run_soon
	# True：从快照目录中获取最新的快照恢复训练 / False：从预训练模型重新开始训练
	resume_training = solverParam.resume_training
	# 删除旧的模型文件
	remove_old_models = solverParam.remove_old_models

	log_file =  "{}/{}.log".format(log_dir,time_postfix)
	train_net_file = "{}/train.prototxt".format(proto_dir)
	test_net_file = "{}/test.prototxt".format(proto_dir)
	deploy_net_file = "{}/deploy.prototxt".format(proto_dir)
	solver_file = "{}/solver.prototxt".format(proto_dir)
	# 快照前缀
	snapshot_prefix = "{}/{}".format(model_dir,model_index)
	# 运行脚本
	job_file = "{}/{}_train.sh".format(job_dir, model_index)
	################################################################################
	# 定义模型名称
	project_name = "{}-{}-{}-{}-{}-{}".format(Base_network,Datasets,Methods,Input_scale,Model_level,model_index)
	################################################################################
	# 创建训练网络
	net = caffe.NetSpec()
	# 为其创建标注数据层，返回数据和标注
	image_data_param = get_image_data_param()
	net.data, net.label = ImageDataLayer(train=True, output_label=True, \
                                          resize_width=solverParam.net_width, \
                                          resize_height=solverParam.net_height, \
                                          **image_data_param)
        # create detector
        detector_type = get_detector_type()
        detector_param = get_detector_param()
        if detector_type == "YOLO":
    	    net = YoloDetector(net, train=True, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo, \
                        extra_top_layers=solverParam.extra_top_layers_forBaseNet, \
                        extra_top_depth=solverParam.extra_top_depth_forBaseNet, \
                        **detector_param)
        elif detector_type == "SSD":
            net = SsdDetector(net, train=True, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, **detector_param)
        elif detector_type == "YOLO_SSD":
            net = Yolo_SsdDetector(net, train=True, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo,**detector_param)
	# create train net file
	with open(train_net_file, 'w') as f:
		print('name: "{}_train"'.format(project_name), file=f)
		print(net.to_proto(), file=f)
	#shutil.copy(train_net_file,"{}/train.prototxt".format(this_result_dir))
	################################################################################
	# 创建测试网络
	net = caffe.NetSpec()
	net.data, net.label = ImageDataLayer(train=False, output_label=True, \
                                          resize_width=solverParam.net_width, \
                                          resize_height=solverParam.net_height, \
                                          **image_data_param)
        if detector_type == "YOLO":
    	    net = YoloDetector(net, train=False, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo, \
                        extra_top_layers=solverParam.extra_top_layers_forBaseNet, \
                        extra_top_depth=solverParam.extra_top_depth_forBaseNet, \
                        **detector_param)
        elif detector_type == "SSD":
            net = SsdDetector(net, train=False, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, **detector_param)
        elif detector_type == "YOLO_SSD":
            net = Yolo_SsdDetector(net, train=False, data_layer="data", \
                        gt_label="label", \
                        net_width=solverParam.net_width, \
                        net_height=solverParam.net_height, \
                        basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo,**detector_param)
	# create test net file
	with open(test_net_file, 'w') as f:
		print('name: "{}_test"'.format(project_name), file=f)
		print(net.to_proto(), file=f)
	#shutil.copy(test_net_file,"{}/test.prototxt".format(this_result_dir))
	################################################################################
	# create depoly net file
	deploy_net = net
	with open(deploy_net_file, 'w') as f:
		net_param = deploy_net.to_proto()
		del net_param.layer[0]
		del net_param.layer[-1]
		net_param.name = '{}_deploy'.format(project_name)
		net_param.input.extend(['data'])
		net_param.input_shape.extend([
		  caffe_pb2.BlobShape(dim=[1, 3, solverParam.net_height, solverParam.net_width])])
		print(net_param, file=f)
	#shutil.copy(deploy_net_file,"{}/deploy.prototxt".format(this_result_dir))
	################################################################################
	# create Solver
	solver_param = solverParam.get_solver_param()
	solver = caffe_pb2.SolverParameter(
		  train_net=train_net_file,
		  test_net=[test_net_file],
		  snapshot_prefix=snapshot_prefix,
		  **solver_param)
	with open(solver_file, 'w') as f:
		print(solver, file=f)
	################################################################################
	# 定义快照和权值文件
	max_iter = 0
	# 查找最近快照文件
	for file in os.listdir(model_dir):
		if file.endswith(".solverstate"):
			basename = os.path.splitext(file)[0]
			iter = int(basename.split("{}_iter_".format(model_index))[1])
			if iter > max_iter:
				max_iter = iter
	train_param = '--weights="{}" \\\n'.format(get_pretained_model())
	if resume_training:
		if max_iter > 0:
			train_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
	# 如果删除旧的快照模型
	if remove_old_models:
		for file in os.listdir(snapshot_dir):
			if file.endswith(".solverstate"):
				basename = os.path.splitext(file)[0]
				iter = int(basename.split("{}_iter_".format(network))[1])
				if max_iter > iter:
					os.remove("{}/{}".format(snapshot_dir, file))
			if file.endswith(".caffemodel"):
				basename = os.path.splitext(file)[0]
				iter = int(basename.split("{}_iter_".format(network))[1])
				if max_iter > iter:
					os.remove("{}/{}".format(snapshot_dir, file))
	# create running file
	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe train \\\n')
		f.write('--solver="{}" \\\n'.format(solver_file))
		f.write(train_param)
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), log_file))
		else:
			f.write('2>&1 | tee {}.log\n'.format(log_file))
	os.chmod(job_file, stat.S_IRWXU)
	if run_soon:
		subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    train()
