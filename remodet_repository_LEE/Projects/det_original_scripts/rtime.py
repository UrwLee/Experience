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

test_iterations = 100

def rtime(Base_network=BaseNet,Datasets=DataSets,Methods=DetMethod,Input_scale=InputScale,Model_level=ModelLevel):
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}/{}/{}/{}".format(Results_dir,Base_network,Datasets,Methods,Input_scale,Model_level)
	################################################################################
        # work and model dirs
	work_model_dir = "{}/Time".format(work_dir)
        make_if_not_exist(work_model_dir)
	proto_dir = "{}/Proto".format(work_model_dir)
        make_if_not_exist(proto_dir)
	job_dir = "{}/Job".format(work_model_dir)
        make_if_not_exist(job_dir)
        ################################################################################
        # work file
        time_net_file = "{}/time_net.prototxt".format(proto_dir)
	job_file = "{}/time.sh".format(job_dir)
        out_file = "{}/time_results.log".format(proto_dir)
	################################################################################
	# 定义模型名称
	project_name = "{}-{}-{}-{}-{}".format(Base_network,Datasets,Methods,Input_scale,Model_level)
	################################################################################
	# 创建网络
	net = caffe.NetSpec()
        image_data_param = get_image_data_param()
        detector_type = get_detector_type()
        detector_param = get_detector_param()
        # train
	net.data, net.label = ImageDataLayer(train=True, output_label=True, \
                                          resize_width=solverParam.net_width, \
                                          resize_height=solverParam.net_height, \
                                          **image_data_param)
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
        else:
            raise ValueError("Unknown detector type.")

	with open(time_net_file, 'w') as f:
		print('name: "{}_train"'.format(project_name), file=f)
		print(net.to_proto(), file=f)
	solver_param = solverParam.get_solver_param()
	################################################################################
        with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe time \\\n')
		f.write('--model="{}" \\\n'.format(time_net_file))
		f.write('--iterations="{}" \\\n'.format(test_iterations))
		if solver_param['solver_mode'] == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), out_file))
		else:
			f.write('2>&1 | tee {}\n'.format(out_file))
	os.chmod(job_file, stat.S_IRWXU)
	subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    rtime()
