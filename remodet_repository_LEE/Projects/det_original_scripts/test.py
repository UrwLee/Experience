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

from testParam import image_source, webcam_index, video_file, test_iter_only, model_idx, caffemodel_index
from testParam import detParam, get_videoinput_transparam

def test(Base_network=BaseNet,Datasets=DataSets,Methods=DetMethod,Input_scale=InputScale,Model_level=ModelLevel):
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}/{}/{}/{}".format(Results_dir,Base_network,Datasets,Methods,Input_scale,Model_level)
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
	work_model_dir = "{}/{}".format(work_dir,model_index)
	proto_dir = "{}/Proto".format(work_model_dir)
	model_dir = "{}/Models".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
        ################################################################################
        # work file
	test_net_file = "{}/test_only.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,model_index)
	job_file = "{}/test_only.sh".format(job_dir)
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
        pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)
        if caffemodel_index > 0 and caffemodel_index < max_iter:
            pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix, caffemodel_index)
	################################################################################
	# 创建网络
	net = caffe.NetSpec()
        image_data_param = get_image_data_param()
        video_input_transparam = get_videoinput_transparam(solverParam.net_width, solverParam.net_height, \
                                              image_data_param['mean_values'])
        if image_source is "webcam":
            (net.data, net.orig_data) = L.VideoData2(ntop=2, \
                                    video_data_param=dict(video_type=P.VideoData.WEBCAM, device_id=webcam_index), \
                                    data_param=dict(batch_size=1), \
                                    transform_param=video_input_transparam)
        else:
            (net.data, net.orig_data) = L.VideoData2(ntop=2, \
                                    video_data_param=dict(video_type=P.VideoData.VIDEO, video_file=video_file), \
                                    data_param=dict(batch_size=1), \
                                    transform_param=video_input_transparam)
        # create detector
        detector_type = get_detector_type()
        detector_param = get_detector_param()
        if detector_type == "YOLO":
	    detector_param['mcdetout_conf_threshold'] = detParam['conf_threshold']
	    detector_param['mcdetout_nms_threshold'] = detParam['nms_threshold']
	    detector_param['mcdetout_boxsize_threshold'] = detParam['boxsize_threshold']
	    detector_param['mcdetout_top_k'] = detParam['top_k']
	    detector_param['mcdetout_visualize'] = detParam['visualize']
	    detector_param['mcdetout_visualize_conf_threshold'] = detParam['visual_conf_threshold']
	    detector_param['mcdetout_visualize_size_threshold'] = detParam['visual_size_threshold']
	    detector_param['mcdetout_display_maxsize'] = detParam['display_maxsize']
	    detector_param['mcdetout_line_width'] = detParam['line_width']
	    detector_param['mcdetout_color'] = detParam['color']
	    net = YoloDetector(net, train=False, data_layer="data", \
                    visualize=True, extra_data="orig_data", eval_enable=False, \
                    gt_label="label", \
                    net_width=solverParam.net_width, \
                    net_height=solverParam.net_height, \
                    basenet=BaseNet, use_layers=solverParam.use_feature_layers_for_yolo, \
                    extra_top_layers=solverParam.extra_top_layers_forBaseNet, \
                    extra_top_depth=solverParam.extra_top_depth_forBaseNet, \
                    **detector_param)
        elif detector_type == "SSD":
	    detector_param['detectionout_conf_threshold'] = detParam['conf_threshold']
	    detector_param['detectionout_nms_threshold'] = detParam['nms_threshold']
	    detector_param['detectionout_boxsize_threshold'] = detParam['boxsize_threshold']
	    detector_param['detectionout_top_k'] = detParam['top_k']
	    detector_param['detectionout_visualize'] = detParam['visualize']
	    detector_param['detectionout_visualize_conf_threshold'] = detParam['visual_conf_threshold']
	    detector_param['detectionout_visualize_size_threshold'] = detParam['visual_size_threshold']
	    detector_param['detectionout_display_maxsize'] = detParam['display_maxsize']
	    detector_param['detectionout_line_width'] = detParam['line_width']
	    detector_param['detectionout_color'] = detParam['color']
            net = SsdDetector(net, train=False, data_layer="data", \
                    visualize=True, extra_data="orig_data", eval_enable=False, \
                    gt_label="label", \
                    net_width=solverParam.net_width, \
                    net_height=solverParam.net_height, \
                    basenet=BaseNet, **detector_param)
        else:
            raise ValueError("unknown detector type.")
	# create test net file
	with open(test_net_file, 'w') as f:
		print('name: "{}_test_only"'.format(project_name), file=f)
		print(net.to_proto(), file=f)
	################################################################################
        with open(job_file, 'w') as f:
            f.write('cd {}\n'.format(caffe_root))
            f.write('./build/tools/caffe test \\\n')
            f.write('--model="{}" \\\n'.format(test_net_file))
            f.write('--weights="{}" \\\n'.format(pretrain_model))
            f.write('--iterations="{}" \\\n'.format(test_iter_only))
            if solverParam.get_solver_mode() == P.Solver.GPU:
                f.write('--gpu {}\n'.format(solverParam.get_gpus()))
	os.chmod(job_file, stat.S_IRWXU)
        # ==========================================================================
        # enjoy now
        subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    test()
