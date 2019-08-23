import cv2
import caffe
import numpy as np
from copy import deepcopy
import os
import matplotlib
import glob
caffe.set_mode_gpu()
caffe.set_device(0)
def compute_iou(box1,box2):
	x1_1 = box1[0]
	y1_1 = box1[1]
	x2_1 = box1[2]
	y2_1 = box1[3]
	x1_2 = box2[0]
	y1_2 = box2[1]
	x2_2 = box2[2]
	y2_2 = box2[3]
	area_inter = float(max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *max((0, min(y2_1, y2_2) - max(y1_1, y1_2))))
	area1 = float((x2_1 - x1_1)*(y2_1 - y1_1))
	area2 = float((x2_2 - x1_2)*(y2_2 - y1_2))
	return float(area_inter)/(area1 + area2 - area_inter)

def filter_boxes(boxes,iou_thre):
	#[xmin,yin,xmax,ymax,score]
	boxes_new = []
	keep_flags = []
	for i in xrange(len(boxes)):
		keep_flags.append(True)
	for i in xrange(len(boxes)):
		keep_id = i
		s_max = boxes[i][-1]
		for j in xrange(i+1,len(boxes)):
			iou = compute_iou(boxes[i],boxes[j])
			if iou>iou_thre:
				if boxes[j][-1]>s_max:
					s_max = boxes[j][-1]
					keep_flags[keep_id] = False
					keep_flags[j] = True
					keep_id = j
				else:
					keep_flags[j] = False

	for i in xrange(len(boxes)):
		if keep_flags[i]:
			boxes_new.append(boxes[i])

	return boxes_new
# #
edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [6,7], [0, 8], [8,9], [9,10], [10,11], [0, 12],
         [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19]]
video_path = "/home/ethan/work/doubleVideo/c1.mp4"
cap = cv2.VideoCapture(0)
scale = 2.0
det_proto = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHand.prototxt"
det_weights = "/home/ethan/ForZhangM/Release20180606/R20180606_Base_BD_PD_MiniHandDist0.5_V1.caffemodel"
keypoint_proto = "/home/ethan/Models/Results/HPKeypoint/test.prototxt"
keypont_weights = "/home/ethan/Models/Results/HPKeypoint/CNN_Base_handposeVD05-4_model2-7cls_keypoint_I96_softmax_R0-6-custom4_Br16_S1.4-2.0_lr-3_Drop0.9_distprob0.3_drift0.10_distmode0_iter_185000.caffemodel"
keypoint_proto = "/home/ethan/Models/Results/HPKeypoint/CNN_Base_handposeVD05-4_model2-7cls_keypoint_I96_softmax_R0-6-custom4_Br16_S1.4-2.0_lr-3_Drop0.9_distprob0.3_drift0.10_distmode0_2deconv_test.prototxt"
keypont_weights = "/home/ethan/Models/Results/HPKeypoint/CNN_Base_handposeVD05-4_model2-7cls_keypoint_I96_softmax_R0-6-custom4_Br16_S1.4-2.0_lr-3_Drop0.9_distprob0.3_drift0.10_distmode0_2deconv_iter_50000.caffemodel"
net_det = caffe.Net(det_proto,det_weights,caffe.TEST)
net_keypoint = caffe.Net(keypoint_proto,keypont_weights,caffe.TEST)
resize_wh = 96
cnt = 0
thre_keypoint = 0.2
mean_value = [104,117,123]
flag_use_camera = False
flag_use_imgdir = True
if flag_use_imgdir:
	img_root = "/media/ethan/RemovableDisk/Datasets/REMO_HandPose/Images"

	img_lists = os.listdir(img_root)

	img_root = "/media/ethan/RemovableDisk/light_condition/ImgsWhole_20180728_2/1"
	img_lists = glob.glob(os.path.join(img_root,"*.jpg"))
colors = []
v = [0,125,255]
for i in xrange(3):
	for j in xrange(3):
		for k in xrange(3):
			colors.append([v[i],v[j],v[k]])

while(True):
	cnt += 1
	if flag_use_camera:
		ret, frame = cap.read()
	if flag_use_imgdir:
		# img_path = os.path.join(img_root,img_lists[cnt-1])
		img_path = os.path.join(img_lists[cnt - 1])
		if "aic" in img_path:
			continue
		frame = cv2.imread(img_path)
	## get hands by hand detection
	frame_resize = cv2.resize(frame,(512,288))
	frame = frame_resize.transpose((2,0,1)).astype(np.float)
	for i in xrange(3):
		frame[i] -= mean_value[i]
	net_det.blobs["data"].data[...] = frame
	net_det.forward()
	det_out = net_det.blobs["det_out"].data
	num_box = det_out.shape[2]
	hand_imgs = []
	s = 0.3
	hand_boxes_org = []
	for i in xrange(num_box):
		if det_out[0,0,i,1] == 1:
			xmin = int(det_out[0,0,i,3]*512)
			ymin = int(det_out[0, 0, i, 4] * 288)
			xmax = int(det_out[0, 0, i, 5] * 512)
			ymax = int(det_out[0, 0, i, 6] * 288)
			s = det_out[0,0,i,2]
			hand_boxes_org.append([xmin,ymin,xmax,ymax,s])
	## remove hands by iou;
	hand_boxes_org = filter_boxes(hand_boxes_org,0.5)
	## get all the hand images used to detect hand keypoints
	hand_boxes = []
	for i in xrange(len(hand_boxes_org)):
		xmin = hand_boxes_org[i][0]
		ymin = hand_boxes_org[i][1]
		xmax = hand_boxes_org[i][2]
		ymax = hand_boxes_org[i][3]

		w_box = xmax - xmin
		h_box = ymax - ymin
		if w_box>h_box:
			ymin -= (w_box - h_box)/2
			ymax += (w_box - h_box)/2
		else:
			xmin -= (h_box - w_box)/2
			xmax += (h_box - w_box)/2
		w_box = xmax - xmin
		h_box = ymax - ymin
		xmin = int(max(0,xmin - s*w_box))
		xmax = int(min(512,xmax + s * w_box))
		ymin = int(max(0, ymin - s * h_box))
		ymax = int(min(288, ymax + s*h_box))

		hand_boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
		# cv2.rectangle(frame_resize,(xmin,ymin),(xmax,ymax),color=(0,0,255))
		# cv2.imshow("a",frame_resize)
		# cv2.waitKey()
		img_tmp = frame_resize[ymin:ymax,xmin:xmax,:]
		hand_imgs.append(cv2.resize(img_tmp,(resize_wh,resize_wh)))
	## perform hand keypoint detection for each hand image
	keypoints = []
	if len(hand_imgs)>0:
		hand_imgs = np.array(hand_imgs).astype(np.float)
		hand_imgs = hand_imgs.transpose((0,3,1,2))
		for i in xrange(3):
			hand_imgs[:,i,:,:] -= mean_value[i]
		net_keypoint.blobs["data"].reshape(len(hand_imgs),3,resize_wh,resize_wh)
		net_keypoint.blobs["data"].data[...] = hand_imgs
		net_keypoint.forward()
		featuremap = net_keypoint.blobs["stage3_out"].data
		stage3_out_vec = net_keypoint.blobs["stage3_out_vec"].data
		vec_show = stage3_out_vec[0]

		vec_show = np.max(vec_show,axis=0)
		vec_show[np.where(vec_show<0.05)]=0
		vec_show[np.where(vec_show >0)] = 1
		vec_show = vec_show/np.max(vec_show)*255
		vec_show = vec_show.astype(np.uint8)
		cv2.imshow("vec_show",cv2.resize(vec_show,(96,96)))
		cv2.waitKey(1)
		for ip in xrange(len(hand_imgs)):
			key_pi = []
			for ikey in xrange(20):
				feati = featuremap[ip,ikey]

				y,x = np.unravel_index(np.argmax(feati, axis=None), feati.shape)
				print feati[y,x]
				if feati[y,x]>thre_keypoint:
					key_pi.append([y*scale,x*scale])
				else:
					key_pi.append([])
			keypoints.append(key_pi)
	## show detected hand boxes and detected hand keypoints
	assert len(hand_boxes) == len(keypoints)
	for i in xrange(len(hand_boxes)):
		xmin_box, ymin_box, xmax_box, ymax_box = hand_boxes[i]
		w_box = xmax_box - xmin_box
		h_box = ymax_box - ymin_box
		cv2.rectangle(frame_resize, (xmin_box, ymin_box), (xmax_box, ymax_box), color=(0, 0, 255))
		for ie, e in enumerate(edges):
			rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
			rgb = rgb * 255
			rgb = rgb.astype(np.int)
			if len(keypoints[i][e[0]])>0 and len(keypoints[i][e[1]])>0:
				y, x = keypoints[i][e[0]]
				xpoint1 = int(xmin_box + x / float(resize_wh) * float(w_box))
				ypoint1 = int(ymin_box + y / float(resize_wh) * float(h_box))
				y, x = keypoints[i][e[1]]
				xpoint2 = int(xmin_box + x / float(resize_wh) * float(w_box))
				ypoint2 = int(ymin_box + y / float(resize_wh) * float(h_box))
				cv2.line(frame_resize, (xpoint1,ypoint1), (xpoint2,ypoint2),
						 color=(rgb[0], rgb[1], rgb[2]),thickness=1)


		for j in xrange(len(keypoints[i])):
			if len(keypoints[i][j])>0:
				y,x = keypoints[i][j]
				xpoint = int(xmin_box + x/float(resize_wh)*float(w_box))
				ypoint = int(ymin_box + y/float(resize_wh)*float(h_box))
				cv2.circle(frame_resize,(xpoint,ypoint),1,colors[j],1)
	frame_resize = cv2.resize(frame_resize,dsize=None,fx=2.0,fy=2.0)
	cv2.imshow("org",frame_resize)
	key = cv2.waitKey(0)
	if key == 27:
		break





	# cv2.imshow("a",frame_resize)
	# key=cv2.waitKey()
	# if key == 27:
	# 	break
