#-*- coding=utf-8 -*-
# 生成计算mAP的检测文件
import cv2
import sys
import os
import random
sys.path.insert(0, '/home/remo/from_wdh/remodet_repository_llp/python')
# print sys.path
import caffe
import numpy as np
sys.path.append("../")
import img_func as func
import math
sys.path.append("/home/remo/from_wdh")
from detFunc import SsdDet
caffe.set_mode_gpu()
caffe.set_device(0)

# 获取模型文件地址,以及测试的图像/视频的地址

def save_txt(dirs,txtx):
    for ind in xrange(len(dirs)):

        dirr = dirs[ind]
        if os.path.exists(os.path.dirname(dirr))==False:
            os.makedirs(os.path.dirname(dirr))
        txt = txtx[ind]
        f=open(dirr,'w')
        for obj in txt:
            st=str(obj[0])
            for ind in xrange(len(obj)):
                if ind !=0:
                    st = st + ' ' + str(obj[ind])
            f.writelines(st+'\n')
        # f.write(txtx)
        f.close()

def det_models():
    net_info = {
        # "cat_dog_A_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A/test.prototxt",
        #                   "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A/Models/DarkNet_cat_dog_A_iter_500000.caffemodel",
        #                   0],
        "cat_dog_fpn_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
                           "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models2/DarkNet_fpn_iter_500000.caffemodel",
                           0],
        "cat_dog_ssd_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_ssd/test.prototxt",
                            "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_ssd/Models/DarkNet_ssd_iter_500000.caffemodel",
                            0],
        "cat_dog_fpn_dis0.5_35w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_dis0.5/test.prototxt",
                                   "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_dis0.5/Models/DarkNet_fpn_iter_350000.caffemodel",
                                   0]
        # "cat_dog_A_dis0.5_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A_dis0.5/test.prototxt",
		# 							"/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A_dis0.5/Models/DarkNet_cat_dog_A_dis0.5_iter_80000.caffemodel",
		# 							0],

    }


    img_root = ["/home/remo/from_wdh/data/val_XML"]

    return net_info, img_root



def gen_det_txt(net_dic,img_roots,det=SsdDet()):
    for img_root in img_roots:
        img_lists = os.listdir(img_root)
        img_lists.sort()

        for num, img_name in enumerate(img_lists):

            if img_name.split(".")[-1] == 'xml':
                img = cv2.imread('/home/remo/from_wdh/data/val2017/' + img_name[:-4] + '.jpg')

                res = det.det_txt(img)
                txt_dirs = res.keys()
                txt_dirs = ['/home/remo/from_wdh/data/predict/' + dirr +'/'+img_name[:-4] + '.txt' for dirr in txt_dirs]
                txts = res.values()
                save_txt(txt_dirs,txts)





if __name__ == '__main__':
    net_dict_info, img_roots = det_models()
    ssd_det = SsdDet()
    ssd_det.det_init(net_dict_info)
    gen_det_txt(net_dict_info,img_roots,ssd_det)

