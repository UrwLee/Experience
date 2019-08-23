#coding=utf-8
import numpy as np
import cv2
import time
import sys
import h5py
import os
import glob
import math
import copy
import xml.dom.minidom
from xml.dom.minidom import Document
from collections import Counter
sys.path.append("/home/remo/from_wdh/remodet_repository_DJ/python")
import caffe


def get_pad169size(image_h,image_w):
    img_width_new = int(max(float(image_h)*16.0/9.0,image_w))
    img_height_new = int(max(float(image_w)*9.0/16.0, image_h))
    return img_height_new,img_width_new


def get_pad916size(image_h,image_w):
    img_width_new = int(max(float(image_h)*9.0/16.0,image_w))
    img_height_new = int(max(float(image_w)*16.0/9.0, image_h))
    return img_height_new,img_width_new


def show_img(frame_resize, wait_time):
    cv2.namedWindow("img",cv2.NORM_HAMMING)
    cv2.imshow("img",cv2.resize(frame_resize,dsize=None, fx=2.0, fy=2.0))
    return cv2.waitKey(wait_time)


def show_det_out(det_out, img_raw):
    img = copy.deepcopy(img_raw)
    num_person = det_out.shape[2]
    rois = []
    score_det = []
    for i_person in xrange(num_person):
        xmin = int(det_out[0][0][i_person][3] * img_w)
        ymin = int(det_out[0][0][i_person][4] * img_h)
        xmax = int(det_out[0][0][i_person][5] * img_w)
        ymax = int(det_out[0][0][i_person][6] * img_h)
        score_det.append(det_out[0][0][i_person][2])
        rois.append([0,xmin,ymin,xmax,ymax])

        cv2.rectangle(img,(xmin,ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(img, '%d' % i_person, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(img,'%.2f' % score_det[i_person], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
    return img


def show_box_prior( det_out2,prior_out, img_raw):
    img = copy.deepcopy(img_raw)
    det_list = [ det_out2,prior_out]
    for cnt, det_out in enumerate(det_list):
        num_person = det_out.shape[0]
        rois = []
        score_det = []
        for i_person in xrange(num_person):
            xmin = int(det_out[i_person][0] * img_w)
            ymin = int(det_out[i_person][1] * img_h)
            xmax = int(det_out[i_person][2] * img_w)
            ymax = int(det_out[i_person][3] * img_h)
            rois.append([0,xmin,ymin,xmax,ymax])

            if cnt == 0:
                cv2.rectangle(img,(xmin,ymin), (xmax, ymax), (255, 0, 0), 1)
                cv2.putText(img, '%d' % i_person, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            elif cnt == 1:
                cv2.rectangle(img,(xmin,ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.putText(img,'%d' % i_person, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
#                 cv2.putText(img,'%.2f' % score_det[i_person], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    return img


def show_box( det_out2, img_raw, img_w, img_h):
    img = copy.deepcopy(img_raw)
    det_list = [ det_out2]
    for cnt, det_out in enumerate(det_list):
        num_person = det_out.shape[0]
        rois = []
        score_det = []
        for i_person in xrange(num_person):
            xmin = int(det_out[i_person][0] * img_w)
            ymin = int(det_out[i_person][1] * img_h)
            xmax = int(det_out[i_person][2] * img_w)
            ymax = int(det_out[i_person][3] * img_h)
            rois.append([0,xmin,ymin,xmax,ymax])

            if cnt == 0:
                cv2.rectangle(img,(xmin,ymin), (xmax, ymax), (255, 0, 0), 1)
                cv2.putText(img, '%d' % i_person, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)


    return img


def read_remotrain_xml(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    meta = {}
    meta['boxes'] = []
    # --------------------------------------------------------------------------
    # ImagePath
    image_path_node = root.getElementsByTagName('ImagePath')[0]
    image_path = image_path_node.childNodes[0].data
    meta['image_path'] = str(image_path.split('.')[0])
    # width & height
    width_node = root.getElementsByTagName('ImageWidth')[0]
    height_node = root.getElementsByTagName('ImageHeight')[0]
    num_person_node = root.getElementsByTagName('NumPerson')[0]
    width = int(width_node.childNodes[0].data)
    height = int(height_node.childNodes[0].data)
    num_person = int(num_person_node.childNodes[0].data)
    meta['width'] = width
    meta['height'] = height
    for i in xrange(num_person):
        person_node = root.getElementsByTagName('Object_%d'%(i + 1))[0]
        box = {}
        cid = int(person_node.getElementsByTagName('cid')[0].childNodes[0].data)
        xmin = int(person_node.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(person_node.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(person_node.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(person_node.getElementsByTagName('ymax')[0].childNodes[0].data)
        box['cid'] = cid
        box['xmin'] = xmin
        box['ymin'] = ymin
        box['xmax'] = xmax
        box['ymax'] = ymax
        meta['boxes'].append(box)
    return meta


def compute_iou(box1, box2,flag_dist,sigma):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = float(area_inter) / float(area1 + area2 - area_inter + 1e-5)
    if flag_dist:
        x1_center = (x1_1 + x2_1)/2.0
        y1_center = (y1_1 + y2_1)/2.0
        x2_center = (x1_2 + x2_2) / 2.0
        y2_center = (y1_2 + y2_2) / 2.0
        dist = math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        w  = math.exp(-dist**2/2/sigma**2)
        iou *= w
        if w>1:
            print w
    return iou


def compute_coverage(box1, box2):
    #box2 is gt
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = float(area_inter) / float(area2 + 1e-5)

    return iou


def GetPriorBox(net, featuremap):
    if featuremap == 1:
        prior_data = net.blobs["featuremap1_1_mbox_priorbox"].data
    elif featuremap == 2:
        prior_data = net.blobs["featuremap2_1_mbox_priorbox"].data
    elif featuremap == 3:
        prior_data = net.blobs["featuremap3_1_mbox_priorbox"].data
    elif featuremap == 'sum':
        prior_data = net.blobs["mbox_2_priorbox"].data
        
    prior_box_data = prior_data[0,0]
    prior_var_data = prior_data[0,1]
    num_priors = prior_box_data.shape[0]/4
    # print "num_priors: ", num_priors
    box = np.zeros((num_priors,4))
    var = np.zeros((num_priors,4))
    for i in xrange(num_priors):
        start_idx = i * 4
        for j in [0,1,2,3]:
            box[i][j] = (prior_box_data[start_idx+j])
            var[i][j] = (prior_var_data[start_idx+j])
    return (box,var)


def GetLocPreds(net, featuremap, numclass):
    if featuremap == 1:
        prior_data = net.blobs["featuremap1_1_mbox_priorbox"].data
        loc_data = net.blobs["featuremap1_1_mbox_loc_flat"].data 
    elif featuremap == 2:
        prior_data = net.blobs["featuremap2_1_mbox_priorbox"].data
        loc_data = net.blobs["featuremap2_1_mbox_loc_flat"].data
    elif featuremap == 3:
        prior_data = net.blobs["featuremap3_1_mbox_priorbox"].data
        loc_data = net.blobs["featuremap3_1_mbox_loc_flat"].data
    
    elif featuremap == 'sum':
        prior_data = net.blobs["mbox_2_priorbox"].data
        loc_data = net.blobs["mbox_2_loc"].data
        
    num_priors = prior_data.shape[2]/4 * (numclass - 1)
    num = loc_data.shape[0]
    
    box = np.zeros((num, num_priors, 4))
    for i in xrange(num):
        for p in xrange(num_priors):
            start_idx = p * 4
            for j in [0, 1, 2, 3]:
                box[i, p ,j] = loc_data[i, start_idx + j]
        loc_data = loc_data[i, start_idx:] # 从下一段 loc开始计数
    return box # num, num_priors, 4)


def GetConfScores(net, featuremap,use_sigmoid=True,num_class=2):
    if featuremap == 1:
        prior_data = net.blobs["featuremap1_1_mbox_priorbox"].data
        conf_data = net.blobs["featuremap1_1_mbox_conf_flat"].data
    elif featuremap == 2:
        prior_data = net.blobs["featuremap2_1_mbox_priorbox"].data
        conf_data = net.blobs["featuremap2_1_mbox_conf_flat"].data
    elif featuremap == 3:
        prior_data = net.blobs["featuremap3_1_mbox_priorbox"].data
        conf_data = net.blobs["featuremap3_1_mbox_conf_flat"].data
    
    elif featuremap == 'sum':
        prior_data = net.blobs["mbox_2_priorbox"].data
        if use_sigmoid:
            conf_data = net.blobs["mbox_2_conf_sigmoid"].data
        else:
            conf_data = net.blobs["mbox_2_conf"].data
        
    num_priors = prior_data.shape[2]/4
    num = conf_data.shape[0]
    
    conf_pred = np.zeros((num, num_class, num_priors))

    for i in xrange(num):
        for j in xrange(num_class):
            for p in xrange(num_priors):
                conf_pred[i, j ,p] = conf_data[i, p * num_class + j]
        conf_data = conf_data[i, num_class * num_priors:]
    return conf_pred


def DecodeBBoxes(net, featuremap):
    all_loc_preds = GetLocPreds(net, featuremap, 2)
    all_conf_scores = GetConfScores(net, featuremap,2)
    (prior_box, var)=GetPriorBox(net, featuremap)

    # 解码过程
    # print "all_loc_preds shape: ",all_loc_preds.shape
    # print "prior_box shape: ",prior_box.shape

    num_bboxes = prior_box.shape[0]
    decode_bboxes = np.zeros((num_bboxes, 4)) 
    pr_w = prior_box[:, 2] - prior_box[:, 0]
    pr_h = prior_box[:, 3] - prior_box[:, 1]
    pr_ctr_x = (prior_box[:, 0] + prior_box[:, 2]) / 2
    pr_ctr_y = (prior_box[:, 3] + prior_box[:, 1]) / 2

    decode_box_x = var[:, 0] * all_loc_preds[:,:, 0] * pr_w + pr_ctr_x
    decode_box_y = var[:, 1] * all_loc_preds[:,:, 1] * pr_h + pr_ctr_y
    decode_box_w = np.exp(var[:, 2] * all_loc_preds[:,:, 2]) * pr_w
    decode_box_h = np.exp(var[:, 3] * all_loc_preds[:,:, 3]) * pr_h

    decode_bboxes[:, 0] = (decode_box_x - decode_box_w /2)
    decode_bboxes[:, 1] = (decode_box_y - decode_box_h /2) 
    decode_bboxes[:, 2] = (decode_box_x + decode_box_w /2) 
    decode_bboxes[:, 3] = (decode_box_y + decode_box_h /2) 
    
    # print "decode_bboxes shape: " , decode_bboxes.shape
    
    decode_bboxes = np.maximum(decode_bboxes, 0)  # [0, 1] 之间
    decode_bboxes = np.minimum(decode_bboxes, 1)
    return decode_bboxes # decode_bboxes  形式为 [num_priors] 和 prior box 的个数相同


def DecodeDenseBBoxes(net, featuremap, num_class=3):
    all_loc_preds = GetLocPreds(net, featuremap, numclass=num_class) # all_loc_preds 形式为 (num, num_priors*(num_class-1), 4)
    all_conf_scores = GetConfScores(net, featuremap, num_class=num_class)
    (prior_box, var) = GetPriorBox(net, featuremap)

    # 解码过程
    # print "all_loc_preds shape: ",all_loc_preds.shape
    # print "prior_box shape: ",prior_box.shape

    num_bboxes = prior_box.shape[0]
    decode_bboxes = np.zeros((num_class-1,num_bboxes, 4))
    pr_w = prior_box[:, 2] - prior_box[:, 0]
    pr_h = prior_box[:, 3] - prior_box[:, 1]
    pr_ctr_x = (prior_box[:, 0] + prior_box[:, 2]) / 2
    pr_ctr_y = (prior_box[:, 3] + prior_box[:, 1]) / 2

    for j in xrange(num_class-1):
        decode_box_x = var[:, 0] * all_loc_preds[:, j*num_bboxes: (j*num_bboxes + num_bboxes ), 0] * pr_w + pr_ctr_x
        decode_box_y = var[:, 1] * all_loc_preds[:, j*num_bboxes: (j*num_bboxes + num_bboxes ), 1] * pr_h + pr_ctr_y
        decode_box_w = np.exp(var[:, 2] * all_loc_preds[:, j*num_bboxes : (j*num_bboxes + num_bboxes), 2]) * pr_w
        decode_box_h = np.exp(var[:, 3] * all_loc_preds[:, j*num_bboxes : (j*num_bboxes + num_bboxes), 3]) * pr_h
        decode_bboxes[j][:, 0] = (decode_box_x - decode_box_w / 2) # 等式右 (1,25344)
        decode_bboxes[j][:, 1] = (decode_box_y - decode_box_h / 2)
        decode_bboxes[j][:, 2] = (decode_box_x + decode_box_w / 2)
        decode_bboxes[j][:, 3] = (decode_box_y + decode_box_h / 2)


    # print "decode_bboxes shape: " , decode_bboxes.shape

    decode_bboxes = np.maximum(decode_bboxes, 0)  # [0, 1] 之间
    decode_bboxes = np.minimum(decode_bboxes, 1)
    return decode_bboxes # shape:[num_class - 1] [num_priors]  num_priors = prior_box.size()


def nms(dets,score,thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         print ovr
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=2 ):


    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i][4]
        maxpos = i

        tx1 = boxes[i][0]
        ty1 = boxes[i][1]
        tx2 = boxes[i][2]
        ty2 = boxes[i][3]
        ts = boxes[i][4]

        pos = 1 + i

        while pos < N:  # 找到分数最高的 pos
            if maxscore < boxes[pos][4]:
                maxscore = boxes[pos][ 4]
                maxpos = pos
            pos = pos + 1

            # add max box as a detection
        boxes[i][0] = boxes[maxpos][0]
        boxes[i][1] = boxes[maxpos][1]
        boxes[i][2] = boxes[maxpos][2]
        boxes[i][3] = boxes[maxpos][3]
        boxes[i][4] = boxes[maxpos][4]

        # swap ith box with position of max box
        boxes[maxpos][0] = tx1
        boxes[maxpos][1] = ty1
        boxes[maxpos][2] = tx2
        boxes[maxpos][3] = ty2
        boxes[maxpos][4] = ts

        tx1 = boxes[i][0]
        ty1 = boxes[i][1]
        tx2 = boxes[i][2]
        ty2 = boxes[i][3]
        ts =  boxes[i][4]
        pos = i + 1

        while pos < N:
            x1 = boxes[pos][0]
            y1 = boxes[pos][1]
            x2 = boxes[pos][2]
            y2 = boxes[pos][3]
            s  = boxes[pos][4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box
                    # print ov
                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                            # print "weight : ", weight
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)

                    else:  # original NMS
                        if ov > Nt:
                            # print "==== ", Nt
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight* boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if  boxes[pos][ 4] < threshold:
                        boxes[pos][ 0] = boxes[N - 1][ 0]
                        boxes[pos][ 1] = boxes[N - 1][ 1]
                        boxes[pos][ 2] = boxes[N - 1][ 2]
                        boxes[pos][ 3] = boxes[N - 1][ 3]
                        boxes[pos][ 4] = boxes[N - 1][ 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

        keep = [i for i in range(N)]
        # print keep
        return keep


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 坐标从大到小排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # score最大的框与按照score顺序与剩余框计算交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 交集大小
        inter = w * h
        # IOU列表
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 将IOU大于预知的从order中排除出去
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


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
    img_one = cv2.resize(img_one,dsize=None,fx = scale,fy=scale)
    return img_one


def compute_iou_simple(box1,box2):
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

#####################AnchorMatch#####################################################


def match_gt_anchor_new(gt_boxes,anchor_boxes,flag_iou,flag_dist,sigma,iou_pos_thre):
    # please set flag_iou=True and flag_dist=True if you want to use gaussian weight
    overlaps = np.zeros((len(anchor_boxes),len(gt_boxes)))

    for ianchor in xrange(len(anchor_boxes)):
        for igt in xrange(len(gt_boxes)):
            if flag_iou:
                iou = compute_iou(anchor_boxes[ianchor], gt_boxes[igt],flag_dist,sigma)
            else:
                iou = compute_coverage(anchor_boxes[ianchor], gt_boxes[igt])
            overlaps[ianchor][igt] = iou
            # overlaps :KxA
    argmax_overlaps = overlaps.argmax(axis=1)  # (K,)
    max_overlaps = overlaps[np.arange(len(anchor_boxes)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # (A,)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]  # (A,)
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    matched = {}

    for i in xrange(len(gt_argmax_overlaps)):
        # print gt_argmax_overlaps[i]
        igt = int(argmax_overlaps[gt_argmax_overlaps[i]])
        try:
            matched[igt].append(gt_argmax_overlaps[i])
        except:
            matched[igt] = [gt_argmax_overlaps[i],]
    pos_ids = np.where(max_overlaps>iou_pos_thre)[0]
    for ip in xrange(len(pos_ids)):
        ianchor = pos_ids[ip]
        igt = argmax_overlaps[ianchor]
        matched[igt].append(ianchor)
    # for ip in pos_ids:

    for key in matched.keys():
        tmp = matched[key]
        tmp = sorted(list(set(tmp)))
        matched[key] = tmp

    return matched


def remove_large_anchor(gt_boxes,anchor_boxes,matched,margin_ratio=0.25,flag_onlycheckW = False):
    #flag_onlycheckW = True: only check width of priorbox; else check both height and width
    matched_new = {}
    for i_gt in matched.keys():
        gt_b = gt_boxes[i_gt]
        gt_w = gt_b[2] - gt_b[0]
        gt_h = gt_b[3] - gt_b[1]
        maring_w = gt_w*margin_ratio
        maring_h = gt_h * margin_ratio
        matched_new[i_gt] = []
        for ianchor in matched[i_gt]:
            anchor_b = anchor_boxes[ianchor]
            margin_left = gt_b[0] - anchor_b[0]
            margin_right = anchor_b[2] - gt_b[2]
            margin_top = gt_b[1] - anchor_b[1]
            margin_bottom = anchor_b[3] - gt_b[3]
            if flag_onlycheckW:
                if margin_left<maring_w and margin_right<maring_w:
                    matched_new[i_gt].append(ianchor)
            else:
                if margin_left<maring_w and margin_right<maring_w and margin_bottom<maring_h and margin_top<maring_h:
                    matched_new[i_gt].append(ianchor)
    return matched_new


def ShowCppDets(net_det, colors, frame_resize, img_w_det, img_h_det):
    frame_resize_dets = frame_resize.copy()
    det_out = net_det.blobs["det_out"].data
    for i_person in xrange(det_out.shape[2]):
        cid = int(det_out[0][0][i_person][1])
        xmin = int(det_out[0][0][i_person][3] * img_w_det)
        ymin = int(det_out[0][0][i_person][4] * img_h_det)
        xmax = int(det_out[0][0][i_person][5] * img_w_det)
        ymax = int(det_out[0][0][i_person][6] * img_h_det)
        score = det_out[0][0][i_person][2]
        # if cid == 0:
        cv2.rectangle(frame_resize_dets, (xmin, ymin), (xmax, ymax), colors[cid], 2)
        cv2.putText(frame_resize_dets, '%d_%0.2f' % (i_person,score), (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[cid], 1)

    return frame_resize_dets


def ShowDets(net_det, colors, frame_resize, img_w_det, img_h_det):
    frame_resize_dets = frame_resize.copy()
    det_out = net_det.blobs["det_out"].data
    for i_person in xrange(det_out.shape[2]):
      cid = int(det_out[0][0][i_person][1])
      xmin = int(det_out[0][0][i_person][3] * img_w_det)
      ymin = int(det_out[0][0][i_person][4] * img_h_det)
      xmax = int(det_out[0][0][i_person][5] * img_w_det)
      ymax = int(det_out[0][0][i_person][6] * img_h_det)
      score = det_out[0][0][i_person][2]
      if cid == 0:
          cv2.rectangle(frame_resize_dets, (xmin, ymin), (xmax, ymax), colors[cid], 2)
          cv2.putText(frame_resize_dets, '%d_%0.2f' % (i_person,score), (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[cid], 1)
    return frame_resize_dets

#####################ShowBboxAndAnchor#####################################################

def ShowBboxAndAnchor(frame_org, net_det, width, height, colors,
                      featuremap='sum', conf_thre=0.8, nms_thre=0.35, pre_nms_topN=200,
                      flag_16_9=True, img_w_det=512, img_h_det=288, mean_data = [104.0,117.0,123.0],
                      showcppdets=False, num_class=2):
    if flag_16_9:
        height_new, width_new = get_pad169size(height, width)
    else:
        height_new, width_new = get_pad916size(height, width)
    frame_org = cv2.copyMakeBorder(frame_org, 0, height_new-height, 0, width_new - width, cv2.BORDER_CONSTANT,
                                  value=(104,117,123))

    frame_resize = cv2.resize(frame_org, (img_w_det, img_h_det))
    frame_input = frame_resize.astype(np.float).transpose((2, 0, 1))
    for i in xrange(3):
        frame_input[i] -= mean_data[i]     
    net_det.blobs["data"].reshape(1,3,img_h_det,img_w_det)
    net_det.blobs["data"].data[...] = frame_input
    net_det.forward()
    if showcppdets:
        frame_resize = ShowCppDets(net_det, colors, frame_resize, img_w_det, img_h_det)
        # print (" print detout without prior box")
        return frame_resize
    else:
        frame_resize = ShowCppDets(net_det, colors, frame_resize, img_w_det, img_h_det)
        (prior_box,var) = GetPriorBox(net_det, featuremap)
        prior_box[:,0] *= img_w_det
        prior_box[:,1] *= img_h_det
        prior_box[:,2] *= img_w_det
        prior_box[:,3] *= img_h_det
        if num_class <= 2:
            TrueBbox = DecodeBBoxes(net_det, featuremap) # 获得检测的所有box
            TrueBbox[:,0] *= img_w_det
            TrueBbox[:,1] *= img_h_det
            TrueBbox[:,2] *= img_w_det
            TrueBbox[:,3] *= img_h_det
        else :
            TrueBbox = DecodeDenseBBoxes(net_det, featuremap,  num_class=3)  # 获得检测的所有box
            # shape: (num_class-1, num_prior, 4)
            TrueBbox[:,:, 0] *= img_w_det
            TrueBbox[:,:, 1] *= img_h_det
            TrueBbox[:,:, 2] *= img_w_det
            TrueBbox[:,:, 3] *= img_h_det

        scores = GetConfScores(net_det, featuremap,num_class=num_class) # shape:[num][num_class-1]][num_prior]
        # for class_id in xrange(num_class-1): # 使用多分类
        class_id = 0 # 使用 2分类
        fg_scores = scores[0,class_id+1,:] # 获得前景的分数 0 是背景
        if num_class > 2:
            TrueBbox = TrueBbox[class_id]
        stay_inds = np.where(fg_scores > conf_thre)[0]
        fg_scores = fg_scores[stay_inds]
        TrueBbox  =  TrueBbox[stay_inds] # shape: (num_class-1, num_prior, 4)
        prior_box = prior_box[stay_inds]
        if pre_nms_topN > 0 and fg_scores.shape[0]>pre_nms_topN:
            order = fg_scores.ravel().argsort()[::-1] #按分数排序
            order = order[:pre_nms_topN]
            TrueBbox = TrueBbox[order, :]
            fg_scores = fg_scores[order]
            prior_box = prior_box[order, :]
        # TrueBbox shape:  [num_prior][4]
        # fg_scores shape: [num_prior][4]
        #

        dets = np.hstack((TrueBbox,fg_scores[:, np.newaxis],prior_box)).astype(np.float32)
        keep = py_cpu_nms(dets, nms_thre)
        # keep = soft_nms(dets, threshold=1, method=1, Nt=1)

        dets = dets[keep, :]
        if dets.shape[1] ==0 or keep == 0:
            return frame_resize
        for i in xrange(dets.shape[0]):
            Truexmin=dets[i,0]
            Trueymin=dets[i,1]
            Truexmax=dets[i,2]
            Trueymax=dets[i,3]
            Priorxmin=dets[i,5]
            Priorymin=dets[i,6]
            Priorxmax=dets[i,7]
            Priorymax=dets[i,8]
            cv2.rectangle(frame_resize, (Truexmin, Trueymin), (Truexmax, Trueymax), colors[0], 2)
            cv2.rectangle(frame_resize, (Priorxmin, Priorymin), (Priorxmax, Priorymax), colors[1], 2)
            cv2.putText(frame_resize, '%d_%0.2f' % (i,dets[i,4]), (Truexmin, Trueymax), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[0], 1)
            cv2.putText(frame_resize, '%d' % (i), (Priorxmin, Priorymax), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[1], 1)
        return frame_resize


def label_to_box(label):
    # label shape: (1,1,num_gt,ndim_label_) ndim_label = 9
    # 0 bindex + base_bindex_;
    # 1 cid;
    # 2 pid;
    # 3 is_diff;
    # 4 iscrowd;
    # 5 bx1_;
    # 6 by1_;
    # 7 bx2_;
    # 8 by2_;
    num_gt = label.shape[2]
    box = []
    for i in xrange(num_gt):
        xmin = label[0,0,i,5]
        ymin = label[0,0,i,6]
        xmax = label[0,0,i,7]
        ymax = label[0,0,i,8] # (0~1)
        cid  = label[0,0,i,1]
        box.append([cid, xmin, ymin, xmax, ymax])
    return box


def gt_stastic(data,label):
    num = data.shape[0]
    is_border=[]
    area=[]
    for i in xrange(num):
        boxes=label_to_box(label)

        for box in boxes:
            cid  = (int)(box[0])
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            area.append((ymax-ymin)*(xmax-xmin))
            if abs(xmin*ymin)<1e-6 or abs((1-xmax)*(1-ymax))<1e-6:
                is_border.append(1)
            else:
                is_border.append(0)
    return area, is_border





def blob_to_img(data, label, colors, mode=False,pic_num=None,pkl=None,save_path=None,flag_save_pic=False,anno=None):
    num = data.shape[0]
    h, w = data.shape[2:4]
    mean_data = [104, 117, 123,104,117,123]  # 均值
    for i in xrange(num):
        imgs = data[i] # shape: (c,h,w)
        for i in xrange(3):
            imgs[i] += mean_data[i]
        imgs = imgs.astype(np.float).transpose((1,2,0)) # shape:(h, w, c)
        img = imgs[..., :3].copy().astype(np.uint8)
        if flag_save_pic:
            cv2.imwrite(save_path+'/images/'+str(pic_num)+'.jpg',img)
            anno[str(pic_num)+'.jpg']={'person':[],'head':[],'hand':[]}
        # raw_img = imgs[..., 3:].copy().astype(np.uint8)
        boxes = label_to_box(label) # boxes shape:(n,5),
        num_box=0
        for box in boxes:
            num_box+=1
            cid  = (int)(box[0])
            if cid == 100:
                print 'aaaaaa'
                continue
            xmin = (int)(box[1] * w)
            ymin = (int)(box[2] * h)
            xmax = (int)(box[3] * w)
            ymax = (int)(box[4] * h)
            if cid == 0:
                if flag_save_pic:
                    anno[str(pic_num)+'.jpg']["person"].append([str(h),str(w),str(xmin),str(ymin),str(xmax),str(ymax)])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cid], 1)
                if mode == "show_iou":
                    gt_area = (xmax-xmin)*(ymax-ymin)*1.0
                    img_area = h*w*1.0
                    # cv2.putText(img,'%.4f'% (gt_area/img_area), (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
                    # cv2.putText(img, '%d*%d' % (xmax - xmin, ymax - ymin), (xmin, ymax + 30),
                    #             cv2.FONT_HERSHEY_COMPLEX, 0.3, [0, 255, 0], 1)
            if cid == 1:
                if flag_save_pic:
                    anno[str(pic_num)+'.jpg']["hand"].append([str(h),str(w),str(xmin),str(ymin),str(xmax),str(ymax)])
            if cid == 2:
                if flag_save_pic:
                    anno[str(pic_num)+'.jpg']["head"].append([str(h),str(w),str(xmin),str(ymin),str(xmax),str(ymax)])
    if flag_save_pic:
        cv2.imwrite(save_path+'/box_images/' +str(pic_num)+'.jpg',img)

    return img,anno

def blob_to_seg_gt(data):
    num = data.shape[0]
    h, w = data.shape[2:4] # 72, 128


    for i in xrange(num):
        imgs = data[i]
        imgs = imgs.astype(np.float).transpose((1,2,0))
        img = imgs.copy().astype(np.uint8)
        img[np.where(img==1)] = 255
        img[np.where(img==3)] = 128
    return img
