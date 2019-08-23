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
import solverParam
from VideoframeParam import *
from mask_data_layer import *
from MaskNet import MaskNet_Test
from MaskNet import MTD_TEST
from username import USERNAME
sys.dont_write_bytecode = True
from solverParam import Results_dir,Project,BaseNet,Models,Ver,caffe_root
from testparam import test_iter,caffemodel_index
def test(Results_dir=Results_dir,Project=Project,BaseNet=BaseNet,Models=Models,Ver=Ver):
	os.chdir(caffe_root)

	work_dir = "{}/{}".format(Results_dir,Project)
	
	model_index = BaseNet+'_'+Models+'_'+Ver
	work_model_dir = "{}/{}".format(work_dir,model_index)

	proto_dir = "{}/Proto".format(work_model_dir)
	model_dir = "{}/Models".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
	test_net_file = "{}/test.prototxt".format(proto_dir)
	snapshot_prefix = "{}/{}".format(model_dir,model_index)
	job_file = "{}/test.sh".format(job_dir)

	Project_name = "{}-{}-{}-{}".format(Results_dir,Project,BaseNet,model_index)

	max_iter = 0
	for file in os.listdir(model_dir):
		if file.endswith(".caffemodel"):
			basename = os.path.splitext(file)[0]
			iter = int(basename.split("_iter_")[1])
			if iter >max_iter:
				max_iter=iter
	pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix,max_iter)
	if caffemodel_index > 0 and caffemodel_index<max_iter:
		pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix,caffemodel_index)

	net = caffe.NetSpec()
	kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': VideotransformParam,
            }
	net.data, net.orig_data = L.Videoframe(name="data", \
		video_frame_param=VideoframeParam, \
		ntop=2, **kwargs)
	# net = MaskNet_Test(net)
	net = MTD_TEST(net)
	with open(test_net_file,'w') as f:
		print("name: '{}_test'".format(Project_name),f)
		print(net.to_proto(),file=f)

		with open(job_file,'w') as f:
			f.write('cd {}\n'.format(caffe_root))
			f.write('./build/tools/caffe test \\\n')
			f.write('--model="{}" \\\n'.format(test_net_file))
			f.write('--weights="{}" \\\n'.format(pretrain_model))
			f.write('--iterations="{}" \\\n'.format(test_iter))
			if solverParam.get_solver_mode() == P.Solver.GPU:
				f.write('--gpu {}\n'.format(solverParam.get_gpus()))
	os.chmod(job_file, stat.S_IRWXU)		
if __name__ == "__main__":
    test()