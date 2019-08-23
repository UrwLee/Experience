#-*- coding=utf-8 -*-
# 统计数据检出情况
import cv2
import sys
import os
import glob
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
		"cat_dog_fpn_ig_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
							   "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models_ig/DarkNet_fpn_ig_iter_200000.caffemodel",
							   0],
    }


    img_root = ["/home/remo/from_wdh/CatDog_Videos/猫","/home/remo/from_wdh/CatDog_Videos/狗"]

    return net_info, img_root



def gen_det_txt(net_dic,img_roots,det=SsdDet()):
    for img_root in img_roots:# cat or dog
        print(img_root)
        img_lists0 = os.listdir(img_root)
        img_lists0 = [img_root +'/'+ pa for pa in img_lists0]
        for  img_listss in img_lists0: # zishi
            img_lists = glob.glob(img_listss +'/*')
            print(img_listss.split('/')[-1])
            total = 0
            detout=0
            for num, img_name in enumerate(img_lists):
                total+=1
                if img_name.split(".")[-1] == 'jpg':
                    img = cv2.imread(img_name)
                    # 2. -------------------检测---------------------------------
                    resss=det.det_txt(img)
                    # print(resss)

                    if resss!=0:
                        detout+=1
            print('nodet, %d  total %d'%(total-detout,total))




if __name__ == '__main__':
    net_dict_info, img_roots = det_models()
    ssd_det = SsdDet()
    ssd_det.det_init(net_dict_info)
    gen_det_txt(net_dict_info,img_roots,ssd_det)




