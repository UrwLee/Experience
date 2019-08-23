# -*- coding: utf-8 -*-
from __future__ import print_function
from username import USERNAME
import sys
import os
import caffe
from caffe import layers as L
from caffe import params as P
from google.protobuf import text_format
import math
import shutil
import stat
import subprocess
import sys
import time
import sys
sys.path.append('../')

sys.dont_write_bytecode = True
from solverParam import *
from DetRelease_Net import *
from DetNet_Param import *
from PyLib.Utils.path import *
from DetRelease_General import *

def DAPNet_Train():
    time_postfix = time.strftime("%m-%d_%H-%M-%S",time.localtime())
    ################################################################################
    os.chdir(caffe_root) # func :check dir 切换到的路径
    ################################################################################
    # work dir
    ProjectName = "{}_{}_{}".format(BaseNet,Models,Ver)
    work_dir = "{}/{}/{}".format(Results_dir,Project,ProjectName)
    make_if_not_exist(work_dir) # func: not exist , mkdir it
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
    if train_net_id == 0:
        net = DetRelease_FirstBodyPartPoseNet(train=True)
    else:
        net = DetRelease_SecondPartAllNet(train=True)

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(ProjectName), file=f)
        print(net.to_proto(), file=f)

    ################################################################################
    # TEST
    if train_net_id == 0:
        net = DetRelease_FirstBodyPartPoseNet(train=False)
    else:
        net = DetRelease_SecondPartAllNet(train=False)

    if isinstance(net, list):
        test_net_files = []
        for idx in xrange(len(net)):

            test_net_file = "{}/test{}.prototxt".format(proto_dir, idx)
            with open(test_net_file, 'w') as f:
                print('name: "{}_test{}"'.format(ProjectName, idx), file=f)
                print(net[idx].to_proto(), file=f)
            test_net_files.append(test_net_file)
        test_net_file = test_net_files
    else:
        with open(test_net_file, 'w') as f:
            print('name: "{}_test"'.format(ProjectName), file=f)
            print(net.to_proto(), file=f)
        test_net_file = [test_net_file,]

    ################################################################################
    # Solver
    solver_param = get_solver_param()
    solver = caffe_pb2.SolverParameter(train_net=train_net_file, \
                    test_net=test_net_file,snapshot_prefix=snapshot_prefix,**solver_param)
    with open(solver_file, 'w') as f:
            print(solver, file=f)
    ################################################################################
    # CaffeModel & Snapshot
    max_iter = 0
    for file in os.listdir(model_dir):
        if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(ProjectName))[1])
            if iter > max_iter:
                max_iter = iter
    if fine_tuning:
        train_param = '--weights="{}" \\\n'.format(get_pretained_model())
    else:
        train_param = ''

    if resume_training:
        if max_iter > 0:
            train_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
    ################################################################################
    # job scripts
    with open(job_file, 'w') as f:
        f.write('cd {}\n'.format(caffe_root))
        f.write('./build/tools/caffe train \\\n')
        f.write('--solver="{}" \\\n'.format(solver_file))
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
    DAPNet_Train()
