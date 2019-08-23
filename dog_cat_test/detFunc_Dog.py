#-*- coding=utf-8 -*-
import sys
import os
print(sys.path)
sys.path.insert(0, "/home/remo/anaconda2/")
import cv2
import random
import glob
sys.path.insert(0, '/home/remo/Desktop/remodet_repository_llp/python')
import caffe
import numpy as np
sys.path.append("../")
import img_func as func
import math
sys.path.append("/home/remo/from_wdh")
from detFunc import SsdDet
caffe.set_mode_gpu()
caffe.set_device(0)

false_dir = './'
# mean_data = [104.0,117.0,123.0] # 均值
# mean_data_color = [104,117,123]
# flag_169 = True
# colors = [[0,0,255],[0,255,0],[0,0,0],[0,255,255]]
# root_folders = ["/home/xjx","/home/xjx/Models/Results/DetPose_JointTrain","/home/xjx/Models/Results/DetNet"]


def show_img(img, name, wait_time):
    cv2.namedWindow("img", cv2.NORM_HAMMING2)
    # cv2.resizeWindow("img", 1920, 1080)
    cv2.imshow("img", img)
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        cv2.destroyAllWindows()
        return True
    if key == ord('s'):
        cv2.imwrite(false_dir+name, img)




def move_pix(img, xmove=0, ymove=0):
    M = np.float32([[1, 0, xmove], [0, 1, ymove]])  #  10
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))  # 11
    return img


def rand_rotate_im(im, prng=np.random, max_angle=0, set_angle=None,reg_loc=None):
    img_h, img_w = im.shape[:2]
    if set_angle is not None:
        angle = set_angle
    else:
        angle = prng.randint(0, max_angle + 1)
    if prng.uniform() > 0.5: # func: 正转, 反转
        angle = -angle

    rotate_M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)  # default anti-clock
    print rotate_M
    cos_M = np.abs(rotate_M[0, 0])  # abs value
    sin_M = np.abs(rotate_M[0, 1])
    # whole box
    r_w = int((img_h * sin_M) + (img_w * cos_M))
    r_h = int((img_h * cos_M) + (img_w * sin_M))
    # position bias
    rotate_M[0, 2] += r_w / 2 - img_w / 2
    rotate_M[1, 2] += r_h / 2 - img_h / 2

    img_R = cv2.warpAffine(im, rotate_M, (r_w, r_h), borderValue=(104.0, 117.0, 123.0))  # (104.0, 117.0, 123.0))
    # Rotate reg_loc
    if reg_loc is not None:
        # 4 cordinates
        tmp_cor_ = np.ones((3, 4))
        for i, cor_ in enumerate(reg_loc):
            if cor_ == []:
                continue
            ncor_ = np.array(cor_).reshape(2, 2).transpose((1, 0))
            tmp_cor_[0:2, 0:2] = ncor_
            tmp_cor_[0, 2] = ncor_[0, 1]
            tmp_cor_[1, 2] = ncor_[1, 0]
            tmp_cor_[0, 3] = ncor_[0, 0]
            tmp_cor_[1, 3] = ncor_[1, 1]

            r_cor_ = np.dot(rotate_M, tmp_cor_)
            # cv2.line(img_R, (int(r_cor_[0, 0]),int(r_cor_[1, 0])) , (int(r_cor_[0, 2]), \
            #                                             int(r_cor_[1, 2])),(0, 255, 0))
            # cv2.line(img_R, (int(r_cor_[0, 2]),int(r_cor_[1, 2])) , (int(r_cor_[0, 1]), \
            #                                             int(r_cor_[1, 1])),(0, 255, 0))
            # cv2.line(img_R, (int(r_cor_[0, 1]),int(r_cor_[1, 1])) , (int(r_cor_[0, 3]), \
            #                                             int(r_cor_[1, 3])),(0, 255, 0))
            # cv2.line(img_R, (int(r_cor_[0, 3]),int(r_cor_[1, 3])) , (int(r_cor_[0, 0]), \
            #                                             int(r_cor_[1, 0])),(0, 255, 0))
            reg_loc[i][0] = int(min(r_cor_[0, :]))
            reg_loc[i][1] = int(min(r_cor_[1, :]))
            reg_loc[i][2] = int(max(r_cor_[0, :]))
            reg_loc[i][3] = int(max(r_cor_[1, :]))

    return img_R, angle, reg_loc


def copy_img(img, scale=1, type_border=cv2.BORDER_CONSTANT):
    '''
    /*
     Various border types, image boundaries are denoted with '|'
     * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
     * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
     * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba     2
     * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg   1
     * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
     */
     '''
    img = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    height_new, width_new = func.get_pad169size(height, width)  # func: 加入边框设置为16:9
    # img = cv2.copyMakeBorder(img, 0, height_new - height, 0, width_new - width, cv2.BORDER_CONSTANT, value=(104, 117, 123))
    img = cv2.copyMakeBorder(img, 0, height_new - height, 0, width_new - width, type_border)
    img = cv2.resize(img, (512, 288))
    # scale = random.uniform(0.1, 0.5)
    if scale >= 1:
        img_w, img_h = int(scale *img.shape[1]), int(scale *img.shape[0]) # func: 缩小crop box框
        resize_img = cv2.resize(img, (img_w, img_h))
        crop_y1 = random.randint(0, img_h - 288)
        crop_x1 = random.randint(0, img_w - 512)
        new_img = resize_img[crop_y1:crop_y1+288, crop_x1:crop_x1+512].copy() # h, w
    else:
        img_w, img_h = int(scale*img.shape[1]), int(scale*img.shape[0])
        img = cv2.resize(img, (img_w, img_h)) # h, w
        new_img = cv2.copyMakeBorder(img, 288-img_h, 0, 512-img_w, 0, type_border )
    return new_img


def nothing(emp):
    pass


def show_videos_with_trackbar(video_name, det=SsdDet()):
    video = video_name[0]
    cv2.namedWindow("video", cv2.NORM_HAMMING2)
    cv2.resizeWindow("video", 1820, 980)
    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获得总帧数
    # print cap.get(cv2.CAP_PROP_FPS) # 获得FPS
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', 'video', 0, frames, nothing)#设置滑动条
    cur_video = "start"
    while 1:
        if loop_flag == pos: # 视频起始位置
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'video', loop_flag)
        else:
            # 设置视频播放位置
            pos = cv2.getTrackbarPos('time', 'video')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos) # 设置当前帧所在位置
        ret, img = cap.read()

        #img = cv2.imread("/home/remo/Desktop/d381eb013ace326d408bda46c6fc36f.jpg")
        if ret == False :
            print("read error ")
            break
        cv2.imshow("raw", img)
        #------------------makeboreder操作------------------------------
        #img = copy_img(img,1)
        # 1. ------------------旋转操作------------------------------
        #img = cv2.transpose(img)
        #img = cv2.flip(img,1)
        # 1. ------------------增广操作------------------------------
        img = aug_img(img)
        #img = img[:,326:639,:]
        # 2. -------------------检测---------------------------------
        det.det_mode(img)
        # 3. --------------- 检测输出 拼接图片-------------------------
        # print(det.imgs_show_all[0].shape)
        # print(det.img_one.shape)
        cv2.imshow("video", det.img_one)
        key = cv2.waitKey(0)
        if key == ord('q') or loop_flag == frames:
            break
        if key == ord('o'):
            wrong_txt = open('/home/remo/Desktop/remo_cat_dog/dog_cat_compare/wrong_video.txt','a+')
            if(video == cur_video):
                wrong_txt.write(' '+str(loop_flag))
            else:
                wrong_txt.write('\n')
                wrong_txt.write(video+' '+str(loop_flag))
            cur_video = video
            #wrong_txt.write(video+' '+str(loop_flag)+'\n')
            wrong_txt.close()
        if key == ord('s'):
            save_v_name = video.split("/")[-1]
            save_path = "%s/%s_%s.jpg" %("/home/remo/Desktop/dog_cat/dog_cat_compare",save_v_name , str(loop_flag))
            cv2.imwrite(save_path, det.img_one)

def show_image(path,det=SsdDet()):
    dir = os.listdir(path)
    for dir_name in dir:
        for _,__,files in os.walk(path+dir_name):
            for filename in files:
                image = cv2.imread(path+dir_name+'/'+filename)
                #cv2.imshow("raw", image          )
                det.det_mode(image)
                #cv2.imshow("image", det.img_one)
                #cv2.waitKey(0)


def main(videos,img_roots,flag_video=False, flag_img=False, flag_cap=False,det=SsdDet()):
    assert flag_video + flag_img + flag_cap <= 1
    if flag_video:
        for video_name in videos:
            print(video_name)
            try:
                show_videos_with_trackbar(video_name, det)
            except:
                print("%s finised!" %video_name[0])
                cv2.destroyAllWindows()
            # if ~ex:
            #     exit()


    if flag_img == True:
        for img_root in img_roots:
            # img_lists = os.listdir(img_root)
            img_lists = glob.glob(img_root+'/*')
            #img_lists.sort()
            for num, img_name in enumerate(img_lists):
                if img_name.split(".")[-1] == 'xml':
                    img = cv2.imread('/home/remo/from_wdh/data/val2017/'+img_name[:-4] + '.jpg')
                    # 1. ------------------增广操作------------------------------
                    img = aug_img(img)
                    # 2. -------------------检测---------------------------------
                    det.det_mode(img)
                    flag_stop = show_img(det.img_one, img_name[:-4]+'.jpg', wait_time=0)
                    if flag_stop:
                        break

                if img_name.split(".")[-1] == 'jpg':
                    print img_name
                    try:
                        img = cv2.imread(img_name)
                        # 1. ------------------增广操作------------------------------
                        img = aug_img(img)
                        # 2. -------------------检测---------------------------------
                        det.det_mode_and_save(img,img_name)
                        #flag_stop = show_img(det.img_one, img_name[:-4]+'.jpg', wait_time=0)
                        #if flag_stop:
                            #break
                    except:
                        continue
    if flag_cap == True:
        cap = cv2.VideoCapture(1)
        cap_frame = 0
        while 1:
            ret, frame_org = cap.read()
            if ret == False: continue
            # 1. ------------------增广操作------------------------------
            frame_org = aug_img(frame_org)
            # 2. -------------------检测---------------------------------
            det.det_mode(frame_org)
            flag_stop = show_img(det.img_one, wait_time=5)
            cap_frame += 1
            if flag_stop:
                break


# 获取模型文件地址,以及测试的图像/视频的地址
def det_models(flag_169, flag_916):
    # models 现在只考虑169比例的图片
    if flag_169:

        # cat dog
        net_dict_info = {
            # "cat_dog_A_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A/test.prototxt" ,
            #             "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A/Models/DarkNet_cat_dog_A_iter_500000.caffemodel",
            #             0],
			# "cat_dog_A_old_26w": [0, 0, "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A/test.prototxt",
			# 				 "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A/Models/DarkNet_cat_dog_A_iter_260000.caffemodel",
			# 				 0],
            # "cat_dog_A_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A/test.prototxt" ,
            #             "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A/Models/DarkNet_cat_dog_A_iter_500000.caffemodel",
            #             0],
            # "DarkNet_90_15wcat_dog_A": [0, 0, "/home/xjx/Models/Results/Det_CatDog/DarkNet_cat_dog_A_r90/test.prototxt" ,
            #             "/home/xjx/Models/Results/Det_CatDog/DarkNet_cat_dog_A_r90/Models/DarkNet_cat_dog_A_r90_iter_150000.caffemodel",
            #             0],
            # "cat_dog_A_r90": [0, 0, "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A_r90/test.prototxt" ,
            #             "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A_r90/Models/DarkNet_cat_dog_A_r90_iter_200000.caffemodel",
            #             0],
            # "cat_dog_A_dis0.5_4w": [0, 0, "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A_dis0.5/test.prototxt" ,
            #             "/home/remo/from_wdh/00Det_CatDog/A_dis0.5/Models/DarkNet_cat_dog_A_dis0.5_iter_40000.caffemodel",
            #             0],
			# "cat_dog_A_dis0.5_8w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A_dis0.5/test.prototxt",
			# 						"/home/remo/from_wdh/Det_CatDog_llp/DarkNet_cat_dog_A_dis0.5/Models/DarkNet_cat_dog_A_dis0.5_iter_80000.caffemodel",
			# 						0],
            # "R_fpn_bn_2lr_21w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_50lr/test.prototxt",
            #                           "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_50lr/RemoNet_cat_dog_fpn_iter_210000.caffemodel",
            #                           0],
            # "cat_dog_fpn_dis0.5_35w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_dis0.5/test.prototxt",
            #                     "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_dis0.5/Models/DarkNet_fpn_iter_350000.caffemodel",
            #                     0],
            # "cat_dog_ssd_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_ssd/test.prototxt",
            #                     "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_ssd/Models/DarkNet_ssd_iter_500000.caffemodel",
            #                     0],
			# "cat_dog_A_dis0.5_old_4w": [0, 0, "/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A_dis0.5/test.prototxt",
			# 						"/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A_dis0.5/Models/DarkNet_cat_dog_A_dis0.5_iter_40000.caffemodel",
			# 						0],
            # "DarkNet_90_5wcat_dog_A": [0, 0, "/home/xjx/Models/Results/Det_CatDog/DarkNet_cat_dog_A_r90/test.prototxt" ,
            #             "/home/xjx/Models/Results/Det_CatDog/DarkNet_cat_dog_A_r90/Models/DarkNet_cat_dog_A_r90_iter_50000.caffemodel",
            #             0],
            # "cat_dog_fpn_ig_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                        "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models_ig/DarkNet_fpn_ig_iter_200000.caffemodel",
            #                        0],
            # "cat_dog_fpn_ig_dis_10w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                        "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_dis0.5/Models_ig/DarkNet_fpn_dis0.5_dark_0.5_iter_100000.caffemodel",
            #                        0],

            # "cat_dog_fpn_data_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                        "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models_data/DarkNet_fpn_mini_iter_200000.caffemodel",
            #                        0],
            # "cat_dog_fpn_mini_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini/test.prototxt",
            #                          "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini/Models/DarkNet_fpn_mini_iter_200000.caffemodel",
            #                          0],
            # "cat_dog_fpn_mini_border_20w": [0, 0,
            #                                 "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                                 "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/Models/DarkNet_fpn_mini_iter_200000.caffemodel",
            #                                 0],
            # "D_f_mini_nop_50w": [0, 0,
            #                                 "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                                 "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson/DarkNet_fpn_mini_iter_350000.caffemodel",
            #                                 0],
            # "cat_dog_fpn_data_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                           "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models_data/DarkNet_fpn_ig_iter_500000.caffemodel",
            #                           0],
            # "RemoNet_fpn_bn_imageinit_327_20w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_sameproto_327/RemoNet_cat_dog_fpn_test_bn.prototxt",
            #                           "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_327/RemoNet_fpn_bn_imageinit_327_iter_200000.caffemodel",
            #                           0],
            # "20w_RemoNet_fpn_bn_imageinit_sameproto_327": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_sameproto_327/RemoNet_cat_dog_fpn_test_bn.prototxt",
            #                           "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_sameproto_327/RemoNet_fpn_bn_imageinit_sameproto_327_iter_200000.caffemodel",
            #                           0],
            # "cat_dog_remo_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/RemoNet_fpn/test.prototxt",
            #                      "/home/remo/from_wdh/Det_CatDog_llp/RemoNet_fpn/Models/RemoNet_cat_dog_fpn_iter_200000.caffemodel",
            #                      0],
            # "cat_dog_remo_bn_3w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/RemoNet_fpn_bn/test.prototxt",
            #                      "/home/remo/from_wdh/Det_CatDog_llp/RemoNet_fpn_bn/Models/RemoNet_cat_dog_fpn_iter_30000.caffemodel",
            #                      0],
            #  "cat_dog_fpn_track_data321_10w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                           "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_track_data_321/DarkNet_fpn_track_data_321_iter_100000.caffemodel",
            #                           0],
            #  "DarkNet_fpn_track_20w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                           "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_track_data_321/DarkNet_fpn_track_data_321_iter_200000.caffemodel",
            #                           0],
            #"cat_dog_fpn_weight_dis0.3_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                         "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_weight_50w_dis0.3/DarkNet_fpn_ig_iter_500000.caffemodel",
            #                         0],
            # "remo_fpn_bn_trackdata321_43w": [0, 0,"/home/remo/from_wdh/Det_CatDog_llp/RemoNet_fpn_bn/test.prototxt",
            #                                 "/home/remo/Desktop/remo_cat_dog_models/Remo_fpn_bn_track_data_321/RemoNet_fpn_bn_track_data_321_iter_430000.caffemodel",
            #                                 0],
            # "DarkNet_fpn_10000_neg_person_402_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                            "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_10000_neg_person_402/DarkNet_fpn10000_neg_person_402_ig_iter_500000.caffemodel",
            #                            0],
            # "DarkNet_coco": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/DarkNet_cat_dog_back328/test.prototxt",
            #                            "/home/remo/Desktop/remo_cat_dog_models/DarkNet_cat_dog_back328/DarkNet_cat_dog_back328_iter_240000.caffemodel",
            #                            0],
            # "DarkNet_otherback": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                            "/home/remo/Desktop/remo_cat_dog_models/DarkNet_cat_dog_back328/DarkNet_cat_dog_back328_iter_240000.caffemodel",
            #                            0],
            # "D_fpn_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                     "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/Models/DarkNet_fpn_iter_500000.caffemodel",
            #                     0],
            # "D_fpn_random_erase_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                     "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_random_erase/DarkNet_fpn_ig_iter_500000.caffemodel",
            #                     0],
            # "D_f_mini_noperson_multi_49w": [0, 0,
            #                                 "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                                 "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi/DarkNet_fpn_mini_noperson_multi__iter_490000.caffemodel",
            #                                 0],
            # "D_back_remove_wrongdata_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn/test.prototxt",
            #                     "/home/remo/Desktop/remo_cat_dog_models/DarkNet_back_remove_wrongdata/DarkNet_back_remove_wrongdata__iter_500000.caffemodel",
            #                     0],
            # "D_f_mini_nop_mul_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                     "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi/DarkNet_fpn_mini_noperson_multi__iter_500000.caffemodel",
            #                     0],
            "D_f_mini_nop_mul_ranera_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
                                "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi_randomerase/DarkNet_fpn_mini_noperson_multi_randomerase_iter_500000.caffemodel",
                                0],
            # "D_f_mini_nop_mul_ranera_iterate_9w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                     "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi_randomerase_iterate/DarkNet_fpn_mini_noperson_multi_randomerase_iterate_iter_90000.caffemodel",
            #                     0],
            # "D_f_mini_nop_mul_ranera_iterate_50w": [0, 0, "/home/remo/from_wdh/Det_CatDog_llp/DarkNet_fpn_mini_border/test.prototxt",
            #                     "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi_randomerase_iterate/DarkNet_fpn_mini_noperson_multi_randomerase_iterate_iter_500000.caffemodel",
            #                     0],
            "D_f_mini_nop_mul_ranera_softmax_50w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi_randomerase_softmax/test.prototxt",
                                "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_169_mini_noperson_multi_randomerase_softmax/DarkNet_fpn_mini_noperson_multi_randomerase_softmax_iter_500000.caffemodel",
                                0]
         }

    elif flag_916:
        net_dict_info = {
            "RemoNet_fpn_bn_imageinit_327_196_50w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_327_196/test.prototxt",
                                       "/home/remo/Desktop/remo_cat_dog_models/RemoNet_fpn_bn_imageinit_327_196/RemoNet_cat_dog_fpn_iter_500000.caffemodel",
                                       0],
            "D_f_916_small_gt_mini_50w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916_small_gt_mini/test.prototxt",
                                       "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916_small_gt_mini/DarkNet_fpn_916_small_gt_mini_ig_iter_500000.caffemodel",
                                       0],
            "D_f_916_small_gt_50w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916_small_gt/test.prototxt",
                                       "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916_small_gt/DarkNet_fpn_916_small_gt_ig_iter_500000.caffemodel",
                                       0],
            "D_f_916_50w": [0, 0, "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916/test.prototxt",
                                       "/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916/DarkNet_fpn_916_ig_iter_500000.caffemodel",
                                       0],
        }
    # videos 测试视频
    videos = []
    if flag_169:
        # cat
        videos = []
        # v_dirs = "/home/xjx/REMO_CatDog_20181031_Original_Videos_DG/cat"
        v_root = "/home/remo/Desktop/remo_source/catdogv_0308_video/猫视频广州20190304"
        for v_name in os.listdir(v_root):
            v_dirs = v_root+'/'+v_name
            vs = os.listdir(v_dirs)
            vs.sort()
            [videos.append(["%s/%s" % (v_dirs, v), 0]) for v in vs]
        #videos.append(["/home/remo/Desktop/remo_cat_dog/catdogv_0308/猫视频广州20190304/猫20190304/NORM0002.MP4",0])


        # print(videos)
        # videos.append(["/home/xjx/videos/cat&dog/inside/cat/4 Olympic Sports Your Cat Would Dominate.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/cat/Cat Agility, CASHMERE TRICKS AND  TARGET TRAINING !!!.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/cat/I Turned My Living Room Into A $600 Indoor Custom Cat House _ Hannah Hart.mkv", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/cat/Man Turns His House Into Indoor Cat Playland and Our Hearts Explode.mkv", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/cat/Purrfect! Man's cat heaven 15 years in the making.mkv", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/dog/Dog training _ Our daycare room is full of calm dogs _ Solid K9 Training Dog Training.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/dog/K9 Classic High Jumping.mkv", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/dog/Zoom Room Dog Training.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/cat&dog/inside/dog/华农兄弟：这群小狗狗刚学会走路，就到处乱跑，小伙都有点招架不住了.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/13422520.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/203.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/37.mp4", 0])  # 夜晚光照射手     , 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/63521353.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/201.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/369.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/49211383.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/609.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射
        # videos.append(["/home/xjx/videos/DogCat/v/70505217.mp4", 0])  # 夜晚光照射手, 衣服容易产生误检 高光照射

    elif flag_916:
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/CD_0213.mp4",20])# 6米 正常背景 效果较好
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/C_1147.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/C_2932.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/CD_5614.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/C_3191.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/CD_0067.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/CD_5176.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/CD_5942.mp4",20])
        videos.append(["/home/remo/from_wdh/CatDog_Videos/cats/C_2563.mp4",20])
# img 测试图片
    img_roots = []
    if flag_169:

        img_roots.append("/home/remo/from_wdh/CatDog_Videos/猫/4_小")


    elif flag_916:
        img_roots.append("")

    return net_dict_info, videos, img_roots


if __name__ == '__main__':
    def aug_img(img):

        # img = move_pix(img, 1, 1)
        # img = copy_img(img, scale=0.6, type_border=cv2.BORDER_CONSTANT)
        # img = aug_gama_com(img, 0.5, 1, show_info=False)
        return img


    flag_169 = True

    flag_916 = not flag_169

    #net_dict_info保存的是test.prototxt和caffe_model的字典，video保存的是测试视频的列表，img_roots保存的是猫的测试图片文件夹
    net_dict_info, videos, img_roots = det_models(flag_169, flag_916)  # func: 获得model
    img_roots2 = []
    img_dirs = os.listdir("/home/remo/Desktop/remo_source/Data_CatDog/OtherBackGround_Images/OtherBackGround_Images")
    for dir in img_dirs:
        img_roots2.append("/home/remo/Desktop/remo_source/Data_CatDog/OtherBackGround_Images/OtherBackGround_Images/"+dir)
    flag = True
    flag_aug_test = not flag

    flag_video = True
    flag_cap = False
    flag_img = False

    if flag:
        ssd_det = SsdDet()#构造一个SsDet对象
        ssd_det.det_init(net_dict_info)#调用初始化函数

        if flag_169:
            ssd_det.flag_169 = True
            ssd_det.flag_916 = False
        elif flag_916:
            ssd_det.flag_169 = False
            ssd_det.flag_916 = True
        main(videos, img_roots2, flag_video, flag_img, flag_cap, ssd_det)

    if flag_aug_test:
        root = "/home/remo/from_wdh/hand/hand_img/"
        list_img = os.listdir(root)
        for img in list_img:

            img = cv2.imread(root + "/" + img)
            img = rand_rotate_im(img, set_angle=90)[0]
            # img = copy_img(img)
            # img = aug_gama_com(img, 0.5, 1, show_info=False)

            show_img(img, 0)
