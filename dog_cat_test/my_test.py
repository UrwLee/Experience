#-*- coding=utf-8 -*-
import sys
import os
print(sys.path)
sys.path.insert(0, "/home/remo/anaconda2/")
import cv2
import random
import glob
sys.path.insert(0, '/home/remo/Desktop/remodet_repository_llp/python')
# print sys.path
import caffe
import numpy as np

img1 = cv2.imread('/home/remo/from_wdh/CatDog_Videos/猫/2_特殊动作/2_tsdz_473538641_a856b692f9_c.jpg')
img_w_det = img1.shape[1]
img_h_det = img1.shape[0]
im = cv2.resize(img1,(512,288))
im = im.astype(np.float).transpose((2, 0, 1))
net = caffe.Net('/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt','/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models_data/DarkNet_fpn_ig_iter_500000.caffemodel',caffe.TEST)
net.blobs["data"].data[...] = im
net.forward()
det_out = net.blobs["det_out"].data
for i in range(det_out.shape[2]):
    xmin = int(det_out[0][0][i][3] * img_w_det)
    ymin = int(det_out[0][0][i][4] * img_h_det)
    xmax = int(det_out[0][0][i][5] * img_w_det)
    ymax = int(det_out[0][0][i][6] * img_h_det)
    cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), 0, 2)
print(det_out)
print(det_out.shape)
print(xrange(det_out.shape[2]))
cv2.imshow('det_out',img1)
cv2.waitKey(0)
