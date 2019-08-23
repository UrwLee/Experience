import cv2
import sys
import os
import caffe
import numpy as np
import math
caffe.set_mode_gpu()
caffe.set_device(0)
mean_data = [104.0,117.0,123.0]

det_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_convf.prototxt"
det_weights = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel"
recls_proto = "/home/ethan/Models/Results/DetNet/recls_K-X_featuremap2_rpnalikeposebeforenmsbatch4_negratio3_iou0.5/test.prototxt"
recls_weights = "/home/ethan/Models/Results/DetNet/recls_K-X_featuremap2_rpnalikeposebeforenmsbatch4_negratio3_iou0.5/recls_K-X_featuremap2_rpnalikeposebeforenmsbatch4_negratio3_iou0.5_iter_60000.caffemodel"

# det_proto = "R20180606_Base_BD_convf.prototxt"
# det_weights = "R20180606_Base_BD_PD_MiniHand_CCom_Convf_V0.caffemodel"
# recls_proto = "test.prototxt"
# recls_weights = "recls_K-X_convf_dropout_rpnalike_iou0.5_iter_5000.caffemodel"

net_det = caffe.Net(det_proto,det_weights,caffe.TEST)
net_recls = caffe.Net(recls_proto,recls_weights,caffe.TEST)
blob_name_recls = "featuremap2"
blob_name_detout = "det_out"
blob_name_reclsout = "recls_fc2_sigmoid"

video_root = "/home/ethan/work/doubleVideo"
i_vid = 1
img_w = 512
img_h = 288
flag_video = True

if flag_video:
    video_str  = "d"
    video_name = "%s%d.mp4"%(video_str,i_vid)
    # video_str = "video_raw"
    # video_name = "%s%d.avi" % (video_str,i_vid)
    cap = cv2.VideoCapture(os.path.join(video_root, video_name))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("%s%d_test.avi"%(video_str,i_vid), fourcc, 60, (img_w, img_h))
else:
    img_root = "/home/ethan/work/doubleVideo/raw_img"
    img_lists = os.listdir(img_root)
    img_lists.sort()
flag = True
cnt = 0
thre = 0.5
flag_write = False
while (flag):
    if flag_video:
        ret, frame_org = cap.read()
    else:
        if cnt>len(img_lists)-1:
            break
        frame_org = cv2.imread(os.path.join(img_root,img_lists[cnt]))
        cnt += 1
    frame_resize = cv2.resize(frame_org, (img_w, img_h))
    frame_input = frame_resize.astype(np.float).transpose((2, 0, 1))
    for i in xrange(3):
        frame_input[i] -= mean_data[i]
    net_det.blobs["data"].data[...] = frame_input
    net_det.forward()
    det_out = net_det.blobs[blob_name_detout].data
    feat_recls = net_det.blobs[blob_name_recls].data

    net_recls.blobs[blob_name_recls].data[...] = feat_recls
    num_person = det_out.shape[2]
    rois = []
    score_det = []
    for i_person in xrange(num_person):
        if det_out[0][0][i_person][0] == 0:
            xmin = int(det_out[0][0][i_person][3] * img_w)
            ymin = int(det_out[0][0][i_person][4] * img_h)
            xmax = int(det_out[0][0][i_person][5] * img_w)
            ymax = int(det_out[0][0][i_person][6] * img_h)
            score_det.append(det_out[0][0][i_person][2])
            rois.append([0,xmin,ymin,xmax,ymax])
    if len(rois)>0:
        print rois
        net_recls.blobs["rois"].reshape(len(rois),5)
        net_recls.blobs["rois"].data[...] = rois
        net_recls.forward()
        score_recls = net_recls.blobs[blob_name_reclsout].data
        print score_recls.shape
        print "#########################################################"
        for i_person in xrange(len(rois)):
            xmin = rois[i_person][1]
            ymin = rois[i_person][2]
            xmax = rois[i_person][3]
            ymax = rois[i_person][4]
            # if math.sqrt(score_det[i_person]*score_recls[i_person,1])>thre:
            cv2.rectangle(frame_resize, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame_resize, '%d' % i_person, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            print i_person, score_det[i_person],score_recls[i_person,1],math.sqrt(score_det[i_person]*score_recls[i_person,1])
            # cv2.putText(frame_resize, '%0.2f' % score_recls[i_person,1], (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("a",cv2.resize(frame_resize,dsize=None,fx=2.0,fy=2.0))
        if not flag_write:
            key = cv2.waitKey(0)
            if key == 13:
                flag_write = True
        else:
            key = cv2.waitKey(1)
        if flag_write:
            out.write(frame_resize)
        print key
        if key == 27:
            break





