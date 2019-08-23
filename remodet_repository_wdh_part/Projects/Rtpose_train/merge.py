#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.dont_write_bytecode = True
import os

import os.path as osp
import google.protobuf as pb
from argparse import ArgumentParser
import sys
import caffe
#from __future__ import print_function
from google.protobuf import text_format
import filecmp
import math
import shutil
import stat
import subprocess
import time
import shutil
import difflib
import json
sys.path.append('../')
from username import USERNAME
sys.dont_write_bytecode = True

from solverParam import *

from PyLib.LayerParam.ImageDataLayerParam import *
from PyLib.NetLib.SsdDetector import *
from PyLib.NetLib.YoloDetector import *
from PyLib.NetLib.ImageDataLayer import *
from PyLib.Utils.dict import *
from PyLib.Utils.data import *
from PyLib.Utils.path import *

from mergeParam import model_idx, caffemodel_index
#from solverParam import caffe_root

def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights):

	with open(src_model) as f:
		model = caffe.proto.caffe_pb2.NetParameter()
		pb.text_format.Merge(f.read(), model)

	for i, layer in enumerate(model.layer):
		if layer.type == 'Convolution': # or layer.type == 'Scale':
			# Add bias layer if needed
			if layer.convolution_param.bias_term == False:
				layer.convolution_param.bias_term = True
				layer.convolution_param.bias_filler.type = 'constant'
				layer.convolution_param.bias_filler.value = 0.0

	with open(dst_model, 'w') as f:
		f.write(pb.text_format.MessageToString(model))

	caffe.set_device(0)  # if we have multiple GPUs, pick the first one
	caffe.set_mode_gpu()
	net_src = caffe.Net(src_model, src_weights, caffe.TEST)
	#net_src = caffe.Net(src_model, caffe.TEST)
	net_dst = caffe.Net(dst_model, caffe.TEST)
	for key in net_src.params.keys():
		#print "key",key,len(net_src.params[key])
		for i in range(len(net_src.params[key])):
			net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]
	return net_dst


def merge_conv_and_bn(net, i_conv, i_bn, i_scale):
	# This is based on Kyeheyon's work
	assert(i_conv != None)
	assert(i_bn != None)

	def copy_double(data):
		return np.array(data, copy=True, dtype=np.double)

	key_conv = net._layer_names[i_conv]
	key_bn = net._layer_names[i_bn]
	key_scale = net._layer_names[i_scale] if i_scale else None

	# Copy
	bn_mean = copy_double(net.params[key_bn][0].data)
	bn_variance = copy_double(net.params[key_bn][1].data)
	num_bn_samples = copy_double(net.params[key_bn][2].data) #scale

	# print bn_mean
	# print bn_variance
	#print num_bn_samples
	# and Invalidate the BN layer
	net.params[key_bn][0].data[:] = 0
	net.params[key_bn][1].data[:] = 1
	net.params[key_bn][2].data[:] = 1
	if num_bn_samples[0] == 0:
		num_bn_samples[0] = 1

	if net.params.has_key(key_scale):
		print ('Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale))
		scale_weight = copy_double(net.params[key_scale][0].data)
		scale_bias = copy_double(net.params[key_scale][1].data)
		# print scale_weight
		# print scale_bias
		net.params[key_scale][0].data[:] = 1
		net.params[key_scale][1].data[:] = 0
	else:
		print ('Combine {:s} + {:s}'.format(key_conv, key_bn))
		scale_weight = 1
		scale_bias = 0
	# return 0
	weight = copy_double(net.params[key_conv][0].data)
	bias = copy_double(net.params[key_conv][1].data)
	alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
	net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
	for i in range(len(alpha)):
		net.params[key_conv][0].data[i] = weight[i] * alpha[i]

def merge_batchnorms_in_net(net):
	# for each BN
	for i, layer in enumerate(net.layers):
		if layer.type != 'BatchNorm':
			continue

		l_name = net._layer_names[i]
		# print (l_name)
		l_bottom = net.bottom_names[l_name]
		assert(len(l_bottom) == 1)
		l_bottom = l_bottom[0]
		l_top = net.top_names[l_name]
		assert(len(l_top) == 1)
		l_top = l_top[0]

		can_be_absorbed = True

		# Search all (bottom) layers
		for j in xrange(i - 1, -1, -1):
			tops_of_j = net.top_names[net._layer_names[j]]
			if l_bottom in tops_of_j:
				if net.layers[j].type not in ['Convolution', 'InnerProduct']:
					can_be_absorbed = False
				else:
					# There must be only one layer
					conv_ind = j
					break

		if not can_be_absorbed:
			continue

		# find the following Scale
		scale_ind = None
		for j in xrange(i + 1, len(net.layers)):
			bottoms_of_j = net.bottom_names[net._layer_names[j]]
			if l_top in bottoms_of_j:
				if scale_ind:
					# Followed by two or more layers
					scale_ind = None
					break

				if net.layers[j].type in ['Scale']:
					scale_ind = j

					top_of_j = net.top_names[net._layer_names[j]][0]
					if top_of_j == bottoms_of_j[0]:
						# On-the-fly => Can be merged
						break

				else:
					# Followed by a layer which is not 'Scale'
					scale_ind = None
					break
		#print "scale_ind",scale_ind
		# if scale_ind != None:
		#     print (conv_ind, i, scale_ind)
		#     print (net._layer_names[conv_ind],net._layer_names[i],net._layer_names[scale_ind])
		#     merge_conv_and_bn(net, conv_ind, i, scale_ind)
		#     break
		merge_conv_and_bn(net, conv_ind, i, scale_ind)
	return net


def process_model(net, src_model, dst_model, func_loop, func_finally):
	with open(src_model) as f:
		model = caffe.proto.caffe_pb2.NetParameter()
		pb.text_format.Merge(f.read(), model)


	for i, layer in enumerate(model.layer):
		map(lambda x: x(layer, net, model, i), func_loop)

	map(lambda x: x(net, model), func_finally)

	with open(dst_model, 'w') as f:
		f.write(pb.text_format.MessageToString(model))

# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
	if layer.type not in ['BatchNorm', 'Scale']:
		return

	bottom = layer.bottom[0]
	top = layer.top[0]

	if (bottom != top):
		# Not supperted yet
		return

	if layer.type == 'BatchNorm':
		zero_mean = np.all(net.params[layer.name][0].data == 0)
		one_var = np.all(net.params[layer.name][1].data == 1)
		#length_is_1 = (net.params['conv1_1/bn'][2].data == 1) or (net.params[layer.name][2].data == 0)
		length_is_1 = (net.params[layer.name][2].data == 1)

		if zero_mean and one_var and length_is_1:
			print ('Delete layer: {}'.format(layer.name))
			to_delete_empty.append(layer)

	if layer.type == 'Scale':
		no_scaling = np.all(net.params[layer.name][0].data == 1)
		zero_bias = np.all(net.params[layer.name][1].data == 0)

		if no_scaling and zero_bias:
			print ('Delete layer: {}'.format(layer.name))
			to_delete_empty.append(layer)

def remove_empty_layers(net, model):
	map(model.layer.remove, to_delete_empty)


# A function to add 'engine: CAFFE' param into 1x1 convolutions
def set_engine_caffe(layer, net, model, i):
	if layer.type == 'Convolution':
		if layer.convolution_param.kernel_size == 1\
			or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
			layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']


def main(model,weights,output_model,output_weights):

	net = load_and_fill_biases(model, weights, model + '.temp.pt', None)
	print ("load biases done")
	net = merge_batchnorms_in_net(net)
	print ("merge net done")
	process_model(net, model + '.temp.pt', output_model,
				  [pick_empty_layers, set_engine_caffe],
				  [remove_empty_layers])
	print ("process done")
	net.save(output_weights)
	caffe.set_device(0)  # if we have multiple GPUs, pick the first one
	caffe.set_mode_gpu()
	net_final = caffe.Net(output_model, output_weights, caffe.TEST)
	net_final.save(output_weights)
	# for layer_name, param in net.params.iteritems():
	# 	print (layer_name + '\t' + str(len(param)))
	# for layer_name, param in net_final.params.iteritems():
	# 	print (layer_name + '\t' + str(len(param)))

	print ("save done")

def deploy_to_test(test,deploy,dst):
	test_content = []
	layer_index = []
	with open(test, 'r') as f:
		save_flag = False
		for line in f:
			if not save_flag:
				if line.startswith("name: "):
					line = line.replace("test","merge_test")
					test_content.append(line)
				elif line == "  name: \"data\"\n" or line == "  name: \"detection_eval\"\n":
					layer_index.append(len(test_content))
					test_content.append("layer {\n")
					#layer_index.append(len(test_content))
					test_content.append(line)
					save_flag = True
			else:
				if line == "}\n":
					save_flag = False
				test_content.append(line)
	#print test_content
	#print layer_index

	with open(dst, 'w') as f:
		index = layer_index[1]
		for i in range(index):
			line = test_content[i].replace("zhangming",USERNAME)
			f.write(line)
		save_flag = False
		with open(deploy,"r") as f_src:
			for line in f_src:
				if not save_flag:
					if line == "layer {\n":
						f.write(line)
						save_flag = True
				else:
					if line == "\n":
						continue
					f.write(line)
		for item in test_content[index:]:
			if item == "\n":
				continue
			f.write(item)
def deploy_to_release(deploy,dst):
	head="""layer {
  name: "data"
  type: "VideoData2"
  top: "data"
  top: "orig_data"
  transform_param {
    mean_value: 104
    mean_value: 117
    mean_value: 123
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 416
      width: 416
      interp_mode: LINEAR
    }
  }
  data_param {
    batch_size: 1
  }
  video_data_param {
    video_type: VIDEO
    video_file: "/home/zhangming/video/FigureSkating2.mp4"
  }
}
"""
	tail = """layer {
  name: "slience"
  type: "Silence"
  bottom: "detection_out"
  include {
    phase: TEST
  }
}"""
	flag = False
	with open(dst,"w") as f:
		with open(deploy,"r") as f_deploy:
			for line in f_deploy:
				if line.startswith("name: "):
					f.write(line.replace("deploy","release"))
					f.write(head)
				if "layer {" in line:
					flag = True
				if flag:
					if "visualize" in line:
						line = line.replace("false","true")
					if line == "\n":
						continue
					if "top:" in line and  "detection_out" in line:
						f.write("  bottom: \"orig_data\"\n")
					f.write(line)
		f.write(tail)



def merge():
	################################################################################
	# caffe根目录
	os.chdir(caffe_root)
	################################################################################
	# work dir
	ProjectName = "{}_{}_{}".format(BaseNet,Models,Ver)
	work_model_dir = "{}/{}/{}".format(Results_dir,Project,ProjectName)
	################################################################################
	model_dir = "{}/Models".format(work_model_dir)
	snapshot_prefix = "{}/{}".format(model_dir,ProjectName)
	proto_dir = "{}/Proto".format(work_model_dir)
	log_dir = "{}/Log".format(work_model_dir)
	job_dir = "{}/Job".format(work_model_dir)
	# new file
	job_file = "{}/release.sh".format(job_dir)
	log_file = "{}/release.log".format(log_dir)
	################################################################################
	# 训练模型
	max_iter = 0
	for file in os.listdir(model_dir):
		if file.endswith(".caffemodel"):
			basename = os.path.splitext(file)[0]
			iter = int(basename.split("{}_iter_".format(ProjectName))[1])
			if iter > max_iter:
				max_iter = iter
	if max_iter == 0:
		raise ValueError("not found .caffemodel in directory: {}".format(model_dir))
	if caffemodel_index > 0 and caffemodel_index < max_iter:
		max_iter = caffemodel_index
	################################################################################
	weights = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)
	model = "{}/deploy.prototxt".format(proto_dir)
	release_file = "{}/release.prototxt".format(proto_dir)
	# test_file = "{}/test.prototxt".format(proto_dir)
	################################################################################
	save_work_dir = "{}/Merge".format(work_model_dir)
	save_model_dir = "{}/Models".format(save_work_dir)
	save_snapshot_prefix = "{}/{}".format(save_model_dir,ProjectName)
	save_proto_dir = "{}/Proto".format(save_work_dir)
	save_job_dir =  "{}/Job".format(save_work_dir)
	save_log_dir =  "{}/Log".format(save_work_dir)
	save_job_file = "{}/release.sh".format(save_job_dir)
	make_if_not_exist(save_work_dir)
	make_if_not_exist(save_model_dir)
	make_if_not_exist(save_proto_dir)
	make_if_not_exist(save_job_dir)
	make_if_not_exist(save_log_dir)
	output_weights = "{}_iter_{}.caffemodel".format(save_snapshot_prefix, max_iter)
	output_model = "{}/deploy.prototxt".format(save_proto_dir)
	main(model,weights,output_model,output_weights)
	save_release_file = "{}/release.prototxt".format(save_proto_dir)
	save_log_file = "{}/release.log".format(save_log_dir)
	# save_test_file = "{}/test.prototxt".format(save_proto_dir)
	################################################################################
	# deploy_to_test(test_file,output_model,save_test_file)
	# deploy_to_release(output_model,save_release_file)
	# deploy_to_release(model,release_file)
	# ################################################################################
	# solver_param = solverParam.get_solver_param()
	# create running file
	# with open(job_file, 'w') as f:
	# 	f.write('cd {}\n'.format(caffe_root))
	# 	f.write('./build/tools/caffe test \\\n')
	# 	f.write('--model="{}" \\\n'.format(release_file))
	# 	f.write('--weights="{}" \\\n'.format(weights))
	# 	f.write('--iterations="{}" \\\n'.format(1000000))
	# 	if solver_param['solver_mode'] == P.Solver.GPU:
	# 		f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), log_file))
	# 	else:
	# 		f.write('2>&1 | tee {}\n'.format(save_log_file))
	#
	# with open(save_job_file, 'w') as f:
	# 	f.write('cd {}\n'.format(caffe_root))
	# 	f.write('./build/tools/caffe test \\\n')
	# 	f.write('--model="{}" \\\n'.format(save_release_file))
	# 	f.write('--weights="{}" \\\n'.format(output_weights))
	# 	f.write('--iterations="{}" \\\n'.format(1000000))
	# 	if solver_param['solver_mode'] == P.Solver.GPU:
	# 		f.write('--gpu {} 2>&1 | tee {}\n'.format(solverParam.get_gpus(), save_log_file))
	# 	else:
	# 		f.write('2>&1 | tee {}\n'.format(save_log_file))

	# os.chmod(job_file, stat.S_IRWXU)
	# subprocess.call(job_file, shell=True)


if __name__ == '__main__':
	merge()
