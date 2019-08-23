#-*- coding=utf-8 -*-
import cv2
import sys
import os
import math
import numpy as np
import inspect
import re


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)


def iou(box1,box2):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1)*(y2_1 - y1_1)
    area2 = (x2_2 - x1_2)*(y2_2 - y1_2)
    return float(area_inter)/(area1 + area2 - area_inter)


def putImgsToOne(imgs,strs,ncols,txt_org=(20,10),fonts = 1,color=(0,0,255)):
    #imgs: list of images
    #strs: strings put on images to identify different models,len(imgs)==len(strs)
    #ncols: columns of images, the code will computed rows according to len(imgs) and ncols automatically
    #txt_org: (xmin,ymin) origin point to put strings
    #fonts: fonts to put strings
    #color: color to put stings
    w_max_win = 1400
    h_max_win = 980
    img_h_max = -1
    img_w_max = -1
    for i in xrange(len(imgs)):
        h,w = imgs[i].shape[:2]
        if h >img_h_max:
            img_h_max = h
        if w >img_w_max:
            img_w_max = w

    if len(imgs)<ncols:
        ncols = len(imgs)
        n_rows = 1
    else:
        n_rows = int(math.ceil(float(len(imgs))/float(ncols)))
    x_space = 5
    y_space = 5

    img_one = np.zeros((img_h_max*n_rows + (n_rows - 1)*y_space,img_w_max*ncols + (ncols - 1)*x_space,3)).astype(np.uint8)
    img_one[:,:,0] = 255
    cnt = 0
    for i_r in xrange(n_rows):
        for i_c in xrange(ncols):
            if cnt>len(imgs) - 1:
                break
            xmin = i_c * (img_w_max + x_space)
            ymin = i_r * (img_h_max + y_space)
            img = imgs[cnt]
            cv2.putText(img, "%d_"%(cnt + 1)+strs[cnt], txt_org, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fonts, color=color,thickness=1)
            img_h_cur, img_w_cur,_ = img.shape
            # print i_r,i_c,"img_h_cur:", img_h_cur, "img_w_cur:", img_w_cur, "img_w_max:", img_w_max, "img_h_max:", img_h_max
            xmax = xmin + img_w_cur
            ymax = ymin + img_h_cur
            # print i_r, i_c,xmin,xmax,ymin,ymax
            img_one[ymin:ymax,xmin:xmax,:] = img
            cnt += 1
    scale_x = float(w_max_win)/float(img_w_max*ncols)
    scale_y = float(h_max_win) / float(img_h_max*n_rows)
    scale = min(scale_x,scale_y)
    # img_one = cv2.resize(img_one,dsize=None,fx = scale,fy=scale)
    return img_one

# 将输入为169的
def get_pad169size(image_h,image_w):
    img_width_new = int(max(float(image_h)*16.0/9.0,image_w))
    img_height_new = int(max(float(image_w)*9.0/16.0, image_h))
    return img_height_new,img_width_new


def get_pad916size(image_h,image_w):
    img_width_new = int(max(float(image_h)*9.0/16.0,image_w))
    img_height_new = int(max(float(image_w)*16.0/9.0, image_h))
    return img_height_new,img_width_new


def compute_iou(box1,box2):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1)*(y2_1 - y1_1)
    area2 = (x2_2 - x1_2)*(y2_2 - y1_2)
    # print area_inter,area1,area2,float(area_inter)/float(area1)/float(area2)
    return float(area_inter)/float(area1+ area2 - area_inter)


def compute_center_dist(box1,box2):
    xc_1 = (box1[0] +  box1[2])/2.0
    yc_1 = (box1[1] + box1[3]) / 2.0
    xc_2 = (box2[0] + box2[2]) / 2.0
    yc_2 = (box2[1] + box2[3]) / 2.0
    dist = math.sqrt((xc_1 - xc_2)**2 + (yc_1 - yc_2)**2)
    return dist


def check_boxes_diff(boxes):
    #boxes = [boxelist_of_model1,boxelist_of_model2,boxelist_of_model3... ]
    flag_diff = False
    for i in xrange(len(boxes)-1):
        if len(boxes[i])!=len(boxes[i+1]):
            flag_diff = True
            break
    if not flag_diff:
        flag_break = False
        for i in xrange(len(boxes)-1):
            for j in xrange(len(boxes[i])):

                dists = []
                for k in xrange(len(boxes[i+1])):
                    d = compute_iou(boxes[i][j],boxes[i + 1][k])
                    dists.append(d)
                flags = [d>0.5 for d in dists]
                flag_match = any(flags)
                if not flag_match:
                    flag_diff = True
                    flag_break = True
                    break
            if flag_break:
                break

    return flag_diff


def compare_inter(frame_org, img_w_det, img_h_det, ):
    cv2.namedWindow("222", cv2.NORM_HAMMING)
    cv2.resizeWindow("222", 1920, 1080)
    frame_resize_inter = cv2.resize(frame_org, (img_w_det, img_h_det))[150:209, 220:269]
    img_nearest = cv2.resize(frame_org, (img_w_det, img_h_det), cv2.INTER_NEAREST)[150:209, 220:269]
    img_linear = cv2.resize(frame_org, (img_w_det, img_h_det), cv2.INTER_LINEAR)[150:209, 220:269]
    img_cubic = cv2.resize(frame_org, (img_w_det, img_h_det), cv2.INTER_CUBIC)[150:209, 220:269]
    img_area = cv2.resize(frame_org, (img_w_det, img_h_det), cv2.INTER_AREA)[150:209, 220:269]
    img_lanczos4 = cv2.resize(frame_org, (img_w_det, img_h_det), cv2.INTER_LANCZOS4)[150:209, 220:269]
    imgs_show_inter = [frame_resize_inter, img_nearest, img_linear, img_cubic, img_area, img_lanczos4]
    # name_imgs_show_inter = ["frame_resize_inter", "img_nearest", "img_linear", "img_cubic", "img_area", "img_lanczos4"]
    name_imgs_show_inter = ["1", "img_nearest", "2", "3", "4", "5"]

    # print list_name
    img_one_inter = putImgsToOne(imgs_show_inter, name_imgs_show_inter, ncols=3, txt_org=(0, 5), fonts=1,
                                      color=(0, 0, 255))
    cv2.imshow("222", img_one_inter)
