# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
