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

from solverParam import *
from RtposeParam import *
from VideoframeParam import *

from PyLib.NetLib.PoseNet import *
from PyLib.NetLib.mPoseNet import *
from PyLib.Utils.path import *

from speedParam import test_iterations,layer_test


def speed():
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	work_dir = "{}/{}/{}".format(Results_dir,Project,ProjectName)
	make_if_not_exist(work_dir)
	################################################################################
	work_model_dir = "{}/Speed".format(work_dir)
	proto_dir = "{}/Proto".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
	make_if_not_exist(work_model_dir)
	make_if_not_exist(proto_dir)
	make_if_not_exist(job_dir)
	################################################################################
	speed_model_file = "{}/test.prototxt".format(proto_dir)
	job_file = "{}/speed.sh".format(job_dir)
	out_file = "{}/speed_results.txt".format(work_model_dir)
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
	# net = mPoseNet(net, from_layer="data", frame_layer="orig_data", use_stages=3, **pose_coco_kwargs)
	net = mPoseNet_COCO_3S_Test(net, from_layer="data", frame_layer="orig_data", use_bn=False, **pose_coco_kwargs)
	with open(speed_model_file, 'w') as f:
		print('name: "{}_speed"'.format(Project), file=f)
		print(net.to_proto(), file=f)

	with open(job_file, 'w') as f:
		f.write('cd {}\n'.format(caffe_root))
		f.write('./build/tools/caffe speed \\\n')
		f.write('--model="{}" \\\n'.format(speed_model_file))
		# f.write('--weights="{}" \\\n'.format(Pretrained_Model))
		f.write('--iterations="{}" \\\n'.format(test_iterations))
		f.write('--layertest="{}" \\\n'.format(layer_test))
		if solver_mode == P.Solver.GPU:
			f.write('--gpu {} 2>&1 | tee {}\n'.format(gpus, out_file))
		else:
			f.write('2>&1 | tee {}\n'.format(out_file))
	os.chmod(job_file, stat.S_IRWXU)
	subprocess.call(job_file, shell=True)


if __name__ == "__main__":
	speed()
