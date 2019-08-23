# -*- coding: utf-8 -*-

# ap_version: '11point' / 'MaxIntegral' / 'Integral'
test_iterations = 500
layer_test = "yes"
# if Ture, test the already existed model;
# if false, the model is not trained, and the result will be put in the Base/TempSpeed folder
# and the model_idx will have no use.
test_already_created = True
# -1: 自动搜索模型编号
# 0... -> 指定模型编号
model_idx = 3
# find the model and caffemodel_index in merge folder
merge = True