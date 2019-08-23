# -*- coding: utf-8 -*-
from __future__ import print_function
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

sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True

from solverParam import Project, ProjectName, Results_dir, Pretrained_Model, gpus, solver_mode
from solverParam import caffe_root

from RtposeParam import *
from VideoframeParam import *

from PyLib.NetLib.PoseNet import *
from PyLib.NetLib.mPoseNet import *
from PyLib.Utils.path import *

def rtpose():
	time_postfix = time.strftime("%m-%d_%H-%M-%S",time.localtime())
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}".format(Results_dir,Project,ProjectName)
	make_if_not_exist(work_dir)
	################################################################################
        # work and model dirs
	proto_dir = "{}/Proto".format(work_dir)
	job_dir = "{}/Job".format(work_dir)
	log_dir = "{}/Log".format(work_dir)
	make_if_not_exist(proto_dir)
	make_if_not_exist(job_dir)
	make_if_not_exist(log_dir)
        ################################################################################
        # work file
	test_file = "{}/rtpose.prototxt".format(proto_dir)
	job_file = "{}/rtpose.sh".format(job_dir)
	log_file = "{}/{}.log".format(log_dir,time_postfix)
	################################################################################
	# 创建网络
	net = caffe.NetSpec()
	# 创建输入层
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': VideotransformParam,
            }
        net.data, net.orig_data = L.Videoframe(name="data", \
                              video_frame_param=VideoframeParam, \
                              ntop=2, **kwargs)
	# 创建网络
	# net = VGG19_PoseNet_COCO_Test(net, from_layer="data", frame_layer="orig_data", **pose_coco_kwargs)
	# net = VGG19_PoseNet_Stage3_COCO_Test(net, from_layer="data", frame_layer="orig_data", **pose_coco_kwargs)
	net = mPoseNet_COCO_3S_Test(net, from_layer="data", frame_layer="orig_data", use_bn=False, **pose_coco_kwargs)
	with open(test_file, 'w') as f:
		print('name: "{}"'.format(ProjectName), file=f)
		print(net.to_proto(), file=f)
        with open(job_file, 'w') as f:
            f.write('cd {}\n'.format(caffe_root))
            f.write('./build/tools/caffe test \\\n')
            f.write('--model="{}" \\\n'.format(test_file))
            f.write('--weights="{}" \\\n'.format(Pretrained_Model))
            f.write('--iterations="{}" \\\n'.format(test_iterations))
            if solver_mode == P.Solver.GPU:
                f.write('--gpu {} 2>&1 | tee {}\n\n'.format(gpus,log_file))
	os.chmod(job_file, stat.S_IRWXU)
        # ==========================================================================
        subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    rtpose()
