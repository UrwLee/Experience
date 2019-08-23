#coding=utf-8
import sys
sys.path.insert(0, "/home/remo/Desktop/remodet_repository_LEE/python")
import caffe
print sys.path
sys.path.insert(0,'./')
import utils
import cv2
import matplotlib.pyplot as pyplt
import matplotlib
from tqdm import tqdm
import numpy as np
import cPickle
#######################PARAMS##################################################
caffe.set_mode_gpu()
caffe.set_device(0)
mean_data = [104.0,117.0,123.0]
colors = [[0,0,0],[0,255,0],[0,0,255],[0,255,255]]

def data_layer_check(data_layer_proto, weight,pkl,save_path,flag_save_pic):

    net = caffe.Net(data_layer_proto,weight, caffe.TRAIN)
    if not flag_save_pic:
        cv2.namedWindow("image", cv2.NORM_HAMMING)
        cv2.resizeWindow("image", 960, 540)
    anno = {}
    for i in tqdm(range(100000)):
        net.forward()
        data_out = net.blobs["data"].data
        data_label = net.blobs["label"].data

        # img = utils.blob_to_img(data_out, data_label, colors, mode="show_iou")
        img,anno = utils.blob_to_img(data_out, data_label, colors,pic_num = i,pkl=pkl,save_path=save_path,flag_save_pic=flag_save_pic,anno=anno)
        # img = cv2.imread("/home/xjx/hand_test/test_20180825_handposeVD05-4_display_1_color/0002412.jpg")

        # img = cv2.resize(img,(1200, 630))
        # 扩编填充
        #img = cv2.copyMakeBorder(img, 0, 100, 100 ,100, cv2.BORDER_REFLECT) #
        # img = cv2.copyMakeBorder(img, 0, 100, 100 ,100, cv2.BORDER_REPLICATE) #
        if not flag_save_pic:
            cv2.imshow("image", img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
    cPickle.dump(anno,pkl)


def crop_gt_stat(data_layer_proto,weight):
    net=caffe.Net(data_layer_proto, weight,caffe.TRAIN)
    is_border = []
    areaes = []
    for i in range(10000):
        net.forward()
        data_label = net.blobs["label"].data
        data_out = net.blobs["data"].data
        area, border = utils.gt_stastic(data_out, data_label)
        is_border.extend(border)
        areaes.extend(area)
    pyplt.hist(areaes,bins=100,normed=0)
    pyplt.show()
    print(np.mean(np.array(is_border)))




def gt_mask_check(data_layer_proto, weight):
    net = caffe.Net(data_layer_proto, weight, caffe.TRAIN)
    cv2.namedWindow("a2", cv2.NORM_HAMMING)
    cv2.resizeWindow("a2", 1920, 1080)

    for i in xrange(100):
        net.forward()
        # gt_out = net.blobs["seg_gt"].data

        data_out = net.blobs["data_pd"].data
        # data_out = net.blobs["data_minihand"].data
        data_label = net.blobs["label_pd"].data
        # data_label = net.blobs["label_minihand"].data

        img = utils.blob_to_img(data_out, data_label, colors, mode=False)

        # gt_img = utils.blob_to_seg_gt(gt_out)

        # cv2.imshow("a2", gt_img)
        cv2.imshow('a3', img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break

########################___DarkNet16:9__#####################################
data_layer_proto = '/home/remo/Desktop/remo_person_hand_face_model/person_model/FPN_BaseLine_PoseGoogle1812_Confnormconf_OneCard48_addDarkaugpic_20190121/check_data_train.prototxt'
weight = '/home/remo/Desktop/remo_person_hand_face_model/release0514/release0514_390000/release0514_390000m__iter_320000.caffemodel'
pkl = open('AIC_BIG_MEN.pkl','w')
save_path = "/home/remo/Desktop/111/"
flag_save_pic = False
# ########################___DarkNet9:16__#####################################
#data_layer_proto = '/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916/data_check_train.prototxt'
#weight = '/home/remo/Desktop/remo_cat_dog_models/DarkNet_fpn_916/DarkNet_fpn_916_ig_iter_500000.caffemodel'
# ########################___mini____#####################################
# data_layer_proto = '/home/remo/from_wdh/Det_CatDog/000MyDarkNet_cat_dog_A/Proto/train_see_mini.prototxt'
# weight = '/home/remo/from_wdh/Det_CatDog/DarkNet_cat_dog_A/Models/DarkNet_cat_dog_A_iter_260000.caffemodel'

# crop_gt_stat(data_layer_proto,weight)
data_layer_check(data_layer_proto, weight,pkl,save_path,flag_save_pic)
pkl.close()
# net = caffe.Net(data_layer_proto, caffe.TRAIN)

# seg_proto = '/home/xjx/Models/Results/DetNet/FPN/DES_Det_Release20180906_Hor_PDBBoxSThr0.05-0.5One0.05-2.0_AroundGTCrop_AICREMO0827_MinihandResizedAICREMO0827NoPerDistP0.3_PDDensBBoxLoc2.0Conf1.0PIOU0.5NIOU0.3R20_MiniHandFace_FPNB_cropScasle2-8_20181115/Proto/train.prototxt'
# seg_weight = '/home/xjx/Models/Results/DetNet/Det_Release20180906_Hor_PDBBoxSThr0.05-0.5_One0.05-2.0_thre0.9_AroundGTCrop_AICREMO0827MinihandResizedAICREMO0827NoPerDistP0.3_PDDensBBoxLoc2.0Conf1.0PIOU0.5NIOU0.3R20_MiniHandFace_20181015/Models/Det_Release20180906_Hor_PDBBoxSThr0.05-0.5_One0.05-2.0_thre0.9_AroundGTCrop_AICREMO0827MinihandResizedAICREMO0827NoPerDistP0.3_PDDensBBoxLoc2.0Conf1.0PIOU0.5NIOU0.3R20_MiniHandFace_20181015_iter_25000.caffemodel'
# gt_mask_check(seg_proto, seg_weight)


