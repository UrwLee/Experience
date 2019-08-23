# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from google.protobuf import text_format
import math
import os
import stat
import subprocess
import sys

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

from speedParam import test_iterations,layer_test,model_idx,test_already_created, merge


def speed(Base_network=BaseNet,Datasets=DataSets,Methods=DetMethod,Input_scale=InputScale,Model_level=ModelLevel):
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}/{}/{}/{}".format(Results_dir,Base_network,Datasets,Methods,Input_scale,Model_level)
	work_dir = "{}/choosen".format(Results_dir)
	################################################################################
	project_name = "{}-{}-{}-{}-{}".format(Base_network,Datasets,Methods,Input_scale,Model_level)
	################################################################################
	solver_param = solverParam.get_solver_param()
	################################################################################
	if test_already_created:
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
		# work_model_dir = "{}/Speed".format(work_dir)
		if merge:
			work_model_dir = "{}/{}/Merge".format(work_dir,model_index)
		else:
			work_model_dir = "{}/{}".format(work_dir,model_index)
		proto_dir = "{}/Proto".format(work_model_dir)
		model_dir = "{}/Models".format(work_model_dir)
		job_dir = "{}/Job".format(work_model_dir)
		################################################################################
		# work file
		speed_model_file = "{}/test.prototxt".format(proto_dir)
		job_file = "{}/{}_speed.sh".format(job_dir,model_index)
		out_file = "{}/speed_results.txt".format(work_model_dir)
		################################################################################
	else:
		work_model_dir = "{}/SpeedTemp".format(work_dir)
		proto_dir = "{}/Proto".format(work_model_dir)
		job_dir = "{}/Job".format(work_model_dir)
		make_if_not_exist(work_model_dir)
		make_if_not_exist(proto_dir)
		make_if_not_exist(job_dir)
		################################################################################
		speed_model_file = "{}/test.prototxt".format(proto_dir)
		job_file = "{}/temp_speed.sh".format(job_dir)
		out_file = "{}/speed_results.txt".format(work_model_dir)
		# 创建网络
		net = caffe.NetSpec()
		image_data_param = get_image_data_param()
		# image_data_param['test_batchsize'] = 1
		detector_type = get_detector_type()
		detector_param = get_detector_param()
		# test
		net.data, net.label = ImageDataLayer(train=False, output_label=True, \
							  resize_width=solverParam.net_width, \
							  resize_height=solverParam.net_height, \
							  **image_data_param)
		if detector_type == "YOLO":
			net = YoloDetector(net, train=False, data_layer="data", \
					gt_label="label", eval_enable=False, \
					net_width=solverParam.net_width, \
					net_height=solverParam.net_height, \
					basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo, \
					extra_top_layers=solverParam.extra_top_layers_forBaseNet, \
					extra_top_depth=solverParam.extra_top_depth_forBaseNet, \
					**detector_param)
		elif detector_type == "SSD":
			net = SsdDetector(net, train=False, data_layer="data", \
					gt_label="label", eval_enable=False, \
					net_width=solverParam.net_width, \
					net_height=solverParam.net_height, \
					basenet=BaseNet, **detector_param)
		else:
			raise ValueError("Unknown detector type.")

		with open(speed_model_file, 'w') as f:
			print('name: "{}_speed"'.format(project_name), file=f)
			print(net.to_proto(), file=f)

	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe speed \\\n')
		f.write('--model="{}" \\\n'.format(speed_model_file))
		f.write('--iterations="{}" \\\n'.format(test_iterations))
		f.write('--layertest="{}" \\\n'.format(layer_test))
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), out_file))
		else:
			f.write('2>&1 | tee {}\n'.format(out_file))
	os.chmod(job_file, stat.S_IRWXU)
	subprocess.call(job_file, shell=True)


if __name__ == "__main__":
	speed()
