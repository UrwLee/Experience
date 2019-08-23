#-*- coding=utf-8 -*-
import cv2
import sys
import os
import random
sys.path.insert(0, '/home/remo/Desktop/remodet_repository_DJ/python')
# print sys.path
import caffe
import numpy as np
sys.path.append("../")
import img_func as func
sys.path.append("/home/remo/from_wdh")
caffe.set_mode_gpu()
caffe.set_device(0)


class SsdDet:

    def det_init(self, net_dict_info):
        self.net_dets = []
        self.mean_data = [104.0, 117.0, 123.0]
        self.colors = [[0, 0, 255], [0, 255, 0], [0, 255, 255], [0, 0, 0]]
        self.root_folders = ["","/home/xjx/Models/Results/DetPose_JointTrain","/home/xjx/Models/Results/DetNet"]
        self.net_keys = net_dict_info.keys()
        self.net_keys = sorted(self.net_keys)
        self.flag_169 = True
        self.flag_916 = not self.flag_169
        self.res = {}
        self.res2 = []

        for i in xrange(len(self.net_keys)):
            path_id_in_root_folders, path_mode, proto_path, weights_path_or_extra_string, itersize = net_dict_info[self.net_keys[i]]

            if path_mode == 0:
                det_proto = os.path.join(self.root_folders[path_id_in_root_folders], proto_path)
                det_weights = os.path.join(self.root_folders[path_id_in_root_folders], weights_path_or_extra_string)
            elif path_mode == 1:
                det_proto = os.path.join(self.root_folders[path_id_in_root_folders], proto_path)
                det_weights = os.path.join(self.root_folders[path_id_in_root_folders], "%s/%s%s_iter_%d.caffemodel" %
                                           (proto_path.split("/")[0],proto_path.split("/")[0], weights_path_or_extra_string, itersize))

            self.net_dets.append(caffe.Net(det_proto,det_weights,caffe.TEST))

        for i in xrange(len(self.net_keys)):
            self.net_keys[i] += "_%s"%str(net_dict_info[self.net_keys[i]][-1])

    # def det_config(self, ):
    def det_mode(self, frame_org):

        if self.flag_169:
            self.img_h_det = 288
            self.img_w_det = 512
        elif self.flag_916 :
            self.img_h_det = 512
            self.img_w_det = 288

        self.scale = 1
        self.imgs_show_all = []
        self.strs_show_all = []
        self.blob_name_detout = "det_out"
        self.ncols_show = 3

        # frame_resize = np.zeros((self.img_h_det,self.img_w_det,3)).astype(np.uint8)
        width = frame_org.shape[1]
        height = frame_org.shape[0]
        if self.flag_169:
            height_new, width_new = func.get_pad169size(height, width)  # func: 加入边框设置为16:9

        elif self.flag_916:
            height_new, width_new = func.get_pad916size(height, width)  # func: 加入边框设置为16:9

        frame_org = cv2.copyMakeBorder(frame_org, 0, height_new - height, 0, width_new - width, cv2.BORDER_CONSTANT,
                                           value=(104, 117, 123))

        frame_resize = cv2.resize(frame_org, (self.img_w_det, self.img_h_det), )

        frame_org_resize = cv2.resize(frame_org, (self.img_w_det, self.img_h_det), cv2.INTER_NEAREST)
        # frame_org_resize = cv2.resize(frame_org_resize,dsize=None,fx=self.scale,fy=self.scale) # 放大
        # 使用双线性插值
        frame_resize[0:frame_org_resize.shape[0],0:frame_org_resize.shape[1],:] = frame_org_resize

        # 移动一个像素
        # M = np.float32([[1, 0, 200], [0, 1, 120]])  # 10
        # frame_resize = cv2.warpAffine(frame_resize, M, (frame_resize.shape[1], frame_resize.shape[0]))  # 11
        # 转化到(C,H,W)
        frame_input = frame_resize.astype(np.float).transpose((2, 0, 1))

        for i in xrange(3):
            frame_input[i] -= self.mean_data[i]

        self.person_boxes_all = []
        for i in xrange(len(self.net_keys)):
            self.net_dets[i].blobs["data"].reshape(1,3,self.img_h_det,self.img_w_det)
            self.net_dets[i].blobs["data"].data[...] = frame_input
            self.net_dets[i].forward()
            det_out = self.net_dets[i].blobs[self.blob_name_detout].data
            #print(det_out.shape,'----------------')
            # des  的mask 输出结果
            # des_mask_out = self.net_dets[i].blobs[].data

            instance_boxes = []
            person_boxes = []
            num_person = det_out.shape[2]
            for i_person in xrange(num_person):

                cid = det_out[0][0][i_person][1]
                xmin = int(det_out[0][0][i_person][3] * self.img_w_det)
                ymin = int(det_out[0][0][i_person][4] * self.img_h_det)
                xmax = int(det_out[0][0][i_person][5] * self.img_w_det)
                ymax = int(det_out[0][0][i_person][6] * self.img_h_det)
                score = det_out[0][0][i_person][2]
                instance_boxes.append([xmin,ymin,xmax,ymax,score,int(cid)])
                if cid == 0:
                    person_boxes.append([xmin, ymin, xmax, ymax, score, int(cid)])
            self.person_boxes_all.append(person_boxes)
            self.img_all_box = instance_boxes #
            frame_show = frame_resize.copy()
            #print instance_boxes
            for i_person in xrange(len(instance_boxes)):
                xmin = instance_boxes[i_person][0]
                ymin = instance_boxes[i_person][1]
                xmax = instance_boxes[i_person][2]
                ymax = instance_boxes[i_person][3]
                score =instance_boxes[i_person][4]
                cid =  instance_boxes[i_person][5]
                cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), self.colors[cid], 2)
                cv2.putText(frame_show, '%d_%0.2f' % (i_person,score), (xmin, ymax-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, self.colors[cid], 1)
                # iou = func.compute_iou([xmin,ymin,xmax,ymax], [0,0,512,288])
                # cv2.putText(frame_show, '%0.5f' % (iou), (xmax+20, ymin-20), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[cid], 1)
                if cid != 0:
                    cv2.putText(frame_show, '%d*%d' % (xmax-xmin, ymax-ymin), (xmin, ymax+30), cv2.FONT_HERSHEY_COMPLEX, 0.8,[0,0,255], 1)
                else:
                    cv2.putText(frame_show, '%d*%d' % (xmax-xmin, ymax-ymin), (xmin, ymax-50), cv2.FONT_HERSHEY_COMPLEX, 0.8,[0,0,255], 1)
            # cv2.putText(frame_show, '%d' % cnt_frame, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            # cv2.imwrite(  "/home/xjx/Documents/det/"+ str(cnt_img)+".jpg", frame_show)
            self.imgs_show_all.append(frame_show)
            self.img_one = func.putImgsToOne(self.imgs_show_all, self.net_keys, ncols=self.ncols_show, txt_org=(0, 20), fonts=1,
                                        color=(0, 0, 255))


    # def det_config(self, ):
    def det_mode_and_save(self, frame_org,img_name):
        f = open("/home/remo/Desktop/remo_cat_dog/Data_CatDog/OtherBackGround_Images/wrong_pic.txt",'a+')
        if self.flag_169:
            self.img_h_det = 288
            self.img_w_det = 512
        elif self.flag_916 :
            self.img_h_det = 512
            self.img_w_det = 288

        self.scale = 1
        self.imgs_show_all = []
        self.strs_show_all = []
        self.blob_name_detout = "det_out"
        self.ncols_show = 3

        # frame_resize = np.zeros((self.img_h_det,self.img_w_det,3)).astype(np.uint8)
        width = frame_org.shape[1]
        height = frame_org.shape[0]
        if self.flag_169:
            height_new, width_new = func.get_pad169size(height, width)  # func: 加入边框设置为16:9

        elif self.flag_916:
            height_new, width_new = func.get_pad916size(height, width)  # func: 加入边框设置为16:9

        frame_org = cv2.copyMakeBorder(frame_org, 0, height_new - height, 0, width_new - width, cv2.BORDER_CONSTANT,
                                           value=(104, 117, 123))

        frame_resize = cv2.resize(frame_org, (self.img_w_det, self.img_h_det), )

        frame_org_resize = cv2.resize(frame_org, (self.img_w_det, self.img_h_det), cv2.INTER_NEAREST)
        # frame_org_resize = cv2.resize(frame_org_resize,dsize=None,fx=self.scale,fy=self.scale) # 放大
        # 使用双线性插值
        frame_resize[0:frame_org_resize.shape[0],0:frame_org_resize.shape[1],:] = frame_org_resize

        # 移动一个像素
        # M = np.float32([[1, 0, 200], [0, 1, 120]])  # 10
        # frame_resize = cv2.warpAffine(frame_resize, M, (frame_resize.shape[1], frame_resize.shape[0]))  # 11
        # 转化到(C,H,W)
        frame_input = frame_resize.astype(np.float).transpose((2, 0, 1))

        for i in xrange(3):
            frame_input[i] -= self.mean_data[i]

        self.person_boxes_all = []
        for i in xrange(len(self.net_keys)):
            self.net_dets[i].blobs["data"].reshape(1,3,self.img_h_det,self.img_w_det)
            self.net_dets[i].blobs["data"].data[...] = frame_input
            self.net_dets[i].forward()
            det_out = self.net_dets[i].blobs[self.blob_name_detout].data
            #print(det_out.shape,'----------------')
            # des  的mask 输出结果
            # des_mask_out = self.net_dets[i].blobs[].data

            instance_boxes = []
            person_boxes = []
            num_person = det_out.shape[2]
            for i_person in xrange(num_person):
                cid = det_out[0][0][i_person][1]
                xmin = int(det_out[0][0][i_person][3] * self.img_w_det)
                ymin = int(det_out[0][0][i_person][4] * self.img_h_det)
                xmax = int(det_out[0][0][i_person][5] * self.img_w_det)
                ymax = int(det_out[0][0][i_person][6] * self.img_h_det)
                score = det_out[0][0][i_person][2]
                instance_boxes.append([xmin,ymin,xmax,ymax,score,int(cid)])
                if cid == 0:
                    person_boxes.append([xmin, ymin, xmax, ymax, score, int(cid)])
            self.person_boxes_all.append(person_boxes)
            self.img_all_box = instance_boxes #
            frame_show = frame_resize.copy()

            for i_person in xrange(len(instance_boxes)):
                xmin = instance_boxes[i_person][0]
                ymin = instance_boxes[i_person][1]
                xmax = instance_boxes[i_person][2]
                ymax = instance_boxes[i_person][3]
                score =instance_boxes[i_person][4]
                cid =  instance_boxes[i_person][5]
                cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), self.colors[cid], 2)
            if(instance_boxes[0][0]>0):
                cv2.imwrite("/home/remo/Desktop/remo_cat_dog/Data_CatDog/OtherBackGround_Images/wrong_pic/"+img_name.split('/')[-2]+'_'+img_name.split('/')[-1],frame_show)
                f.write(img_name+'\n')
                f.close()

    def det_txt(self,frame):
        self.img_h_det = 288
        self.img_w_det = 512
        self.blob_name_detout = "det_out"
        # 记录输入图像的大小
        width = frame.shape[1]
        height = frame.shape[0]
        # 计算变为169所需加的边大小,并向右下方扩充,并resize成169(512,288)
        height_new, width_new = func.get_pad169size(height, width)  # func: 加入边框设置为16:9
        frame_org = cv2.copyMakeBorder(frame, 0, height_new - height, 0, width_new - width, cv2.BORDER_CONSTANT,
                                       value=(104, 117, 123))
        frame_org_resize = cv2.resize(frame_org, (self.img_w_det, self.img_h_det), cv2.INTER_NEAREST)
        frame_input = frame_org_resize.astype(np.float).transpose((2, 0, 1))
        for i in xrange(3):
            frame_input[i] -= self.mean_data[i]

        # print(xrange(len(self.net_keys)))
        for i in xrange(len(self.net_keys)):

            self.net_dets[i].blobs["data"].reshape(1,3,self.img_h_det,self.img_w_det)
            self.net_dets[i].blobs["data"].data[...] = frame_input
            self.net_dets[i].forward()
            det_out = self.net_dets[i].blobs[self.blob_name_detout].data

            tmp_res=[]
            num_object = det_out.shape[2]
            for j in xrange(num_object):
                cid = int(det_out[0][0][j][1])
                xmin = int(np.clip(det_out[0][0][j][3] * width_new,0,width))
                ymin = int(np.clip(det_out[0][0][j][4] * height_new,0,height))
                xmax = int(np.clip(det_out[0][0][j][5] * width_new,0,width))
                ymax = int(np.clip(det_out[0][0][j][6] * height_new,0,height))
                score = det_out[0][0][j][2]
                if cid != -1:
                    tmp_res.append([cid, score, xmin, ymin, xmax, ymax])
            # print(tmp_res)
            # self.res2.append(tmp_res)
            # print self.res2
        return len(tmp_res)
