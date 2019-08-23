import math




########train_net_id##############################
"""
train_net_id = 0: training first bd and pd
train_net_id = 1: train second pd and minihand
"""
train_net_id = 1
flag_169_global = False#if train_net_id==0: it does not matter where flag_169 is True or False
flag_noperson = True
###Batchsize###

###Dist_Param
dist_prob = 0.3
flag_eqhist = False
brightness_delta = 20#20
contrast_lower = 0.5#0.5
contrast_upper = 1.5#1.5
hue_delta = 18
saturation_lower = 0.5
saturation_upper = 1.5
####Information for Body and Part
if train_net_id == 0:
    min_scale = 0.75
    max_scale = 2.0
    for_body = True
    batch_size = 12
else:
    min_scale = 0.05
    max_scale = 2.0
    for_body = False
    batch_size = 24




