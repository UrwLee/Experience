import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np

# xml_list1 = "/home/remo/from_wdh/data/val_list.txt"
# xml_root1 = "/home/remo/from_wdh/data/"
no_gt = 0

xml_list = "/home/remo/cat_dog_data/COCO_clean_xml/train_xml.txt"

xml_root1 = "/home/remo/cat_dog_data/COCO_clean_xml/train_xml/"
xml_root2 = "/home/remo/cat_dog_data/COCO_clean_xml/old_xml/"
img_root = "/home/remo/cat_dog_data/COCO_ori/train2017/"

xml_list_oid = '/home/remo/cat_dog_data/OID_clean_xml/train_xml.txt'
xml_root1_oid = "/home/remo/cat_dog_data/OID_clean_xml/train_xml/"
xml_root2_oid = "/home/remo/cat_dog_data/OID_clean_xml/old_xml/"
img_root_oid = "/home/remo/cat_dog_data/OID/Images/"

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


def show_img(path, bboxes,num):
	color = [(0,255,0),(0,255,255)]
	if num==1:
		img= cv2.imread(img_root+path)
	else:
		img= cv2.imread(img_root_oid + path)
	for i in xrange(2):
		for ind in xrange(len(bboxes[i])):
			cv2.rectangle(img, (bboxes[i][ind][0],bboxes[i][ind][1]),(bboxes[i][ind][2],bboxes[i][ind][3]), color[i], 3)
	# print(img)
	return img

# from xml file read all person object (cid==0) and its bbox coordinate
def read_xml(path,alias=0):
	try:
		tree = ET.parse(path)

		root = tree.getroot()
	except Exception as e:
		print('parse  xml document fail!')
		sys.exit()

	img_name = root.find('ImagePath').text.split('/')[-1]

	num_obj = root.find('NumPerson')
	if num_obj is None:
		num_obj = root.find('NumParts')
	num_obj = int(num_obj.text)

	tmp_shape = (int(root.find('ImageHeight').text),int(root.find('ImageWidth').text))

	tmp_bbox_list = [[], []]

	for ind in xrange(num_obj):
		obj = root.find('Object_%d'% (ind+1))
		# # only select the categry id ==0
		# if obj.find('cid').text != '0':
		# 	continue
		bbox=[]
		# bbox.append(int(obj.find('cid').text)+alias)
		bbox.append(int(obj.find('xmin').text))
		bbox.append(int(obj.find('ymin').text))
		bbox.append(int(obj.find('xmax').text))
		bbox.append(int(obj.find('ymax').text))
		tmp_bbox_list[int(obj.find('cid').text)+alias].append(bbox)


	return img_name, tmp_shape, tmp_bbox_list

def filt_bboxes(bboxes,img_shape):
	ress=[]
	for it in bboxes:
		res = []
		for box in it:
			area = float(box[2]-box[0])*(box[3]-box[1])/(img_shape[0]*img_shape[1])
			print("area%f"%area)
			if area > 0.01:
				res.append(box)
		ress.append(res)
	return ress


def dif_stat(path1, path2):
	imgname, img_shape, bboxes1 = read_xml(path1, -1)
	imgname, img_shape, bboxes2 = read_xml(path2)

	# cv2.namedWindow("new_image", cv2.NORM_HAMMING)
	# cv2.resizeWindow("new_image", 960, 540)
	# cv2.namedWindow("raw_image", cv2.NORM_HAMMING)
	# cv2.resizeWindow("raw_image", 960, 540)
	# num=0
	# if path1.find('COCO')!=-1:
	# 	num =1
	# else :
	# 	num =2
	bboxes_11 = filt_bboxes(bboxes1,img_shape)


	if len(bboxes1[0]) != len(bboxes2[0]) or len(bboxes1[1]) != len(bboxes2[1]):
		print(bboxes1)
		print(bboxes2)

		bboxes1 = filt_bboxes(bboxes1,img_shape)
		bboxes2 = filt_bboxes(bboxes2,img_shape)

		# show_img(imgname, bboxes1, 'new_image', num)
		# show_img(imgname, bboxes2, 'raw_image', num)
		return len(bboxes1[0]) - len(bboxes2[0]) , len(bboxes1[1]) - len(bboxes2[1]),len(bboxes_11[0])+len(bboxes_11[1])
	return 0,0,len(bboxes_11[0])+len(bboxes_11[1])



def com2xml(path1,path2):
	imgname,_,bboxes1 = read_xml(path1, -1)
	imgname,_,bboxes2 = read_xml(path2)
	cv2.namedWindow("image", cv2.NORM_HAMMING)
	cv2.resizeWindow("image", 960, 540)

	# print(bboxes1)
	# print(bboxes2)
	num=0
	if path1.find('COCO')!=-1:
		num =1
	else :
		num =2
	if len(bboxes1[0])< len(bboxes2[0]) or len(bboxes1[1])< len(bboxes2[1]):

		img_new = show_img(imgname,bboxes1,num)
		img_old = show_img(imgname,bboxes2,num)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img_new, 'img_new', (10, 20), font, 1, (0, 0, 255), 2)
		cv2.putText(img_old, 'img_old', (10, 20), font, 1, (0, 0, 255), 2)
		print(img_new.shape)
		img = np.concatenate([img_new,img_old],axis=1)
		cv2.imshow('image',img)
		k = cv2.waitKey(0)
		if k != 'q':
			return
	# else:
	# 	show=False
	# 	for i in xrange(2):
	# 		for ind in xrange(len(bboxes2[i])):
	# 			dif = 1-iou(bboxes1[i][ind], bboxes2[i][ind])
	# 			if dif>0.25:
	# 				show=True
	# 				break
	# 	if show:
	# 		# print(dif)
	# 		show_img(imgname, bboxes1,'new_image')
	# 		show_img(imgname, bboxes2,'raw_image')




xml_path_list1 = []
xml_path_list2 = []
with open(xml_list) as f:
	for name in f.readlines():
		xml_path_list1.append(xml_root1+name[:-1].split('/')[-1])
		xml_path_list2.append(xml_root2 + name[:-1].split('/')[-1])
with open(xml_list_oid) as f:
	for name in f.readlines():
		xml_path_list1.append(xml_root1_oid+name[:-1].split('/')[-1])
		xml_path_list2.append(xml_root2_oid + name[:-1].split('/')[-1])

total = len(xml_path_list1)
cat_add = 0
cat_min = 0
dog_add = 0
dog_min = 0

for ind in xrange(len(xml_path_list1)):
	path1 = xml_path_list1[ind]
	path2 = xml_path_list2[ind]
	# cat, dog ,num = dif_stat(path1,path2)
	# if cat>0:
	# 	cat_add += cat
	# if cat<0:
	# 	cat_min -=cat
	# if dog>0:
	# 	dog_add +=dog
	# if dog<0:
	# 	dog_min -= dog
	# if num==0:
	# 	no_gt+=1

# print('cat add %d boxes, cat_minus %d boxes of total %d image'% (cat_add,cat_min, total))
# print('dog add %d boxes, dog_minus %d boxes of total %d image'% (dog_add,dog_min, total))
# print('no_gt%d'%no_gt)

	com2xml(path1,path2)
