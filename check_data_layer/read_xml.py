
import xml.etree.ElementTree as ET
import os
import sys
import sys
sys.path.insert(0, '/home/remo/llp/det_intern/python')
import caffe
import cv2
import numpy as np

xml_list = "/home/remo/from_wdh/data/val_list.txt"
xml_root = "/home/remo/from_wdh/data/"



img_root = '/home/remo/llp/data/remo_coco/'
proto_path = '/home/remo/llp/det_intern/0my_detection/base_det_deploy.prototxt'
weight_path = '/home/remo/llp/det_intern/0my_detection/my_snap_full/snap_169_test_iter_150000.caffemodel'


mean_data = np.array([[[104.0, 117.0, 123.0]]])
mean_data_color = np.array([[[104, 117, 123]]])
blob_name_detout = "detection_out_1"
Net = caffe.Net(proto_path, weight_path, caffe.TEST)


# create a path dir if it is not exist
def create_dir(pa):
	if not os.path.exists(pa):
		os.makedirs(pa)


# print all thing of the xml file
def traverse_xml(element):
	if len(element)>0:
		for child in element:
			print(child.tag, '------', child.attrib)
			traverse_xml(child)


# from xml file read all person object (cid==0) and its bbox coordinate
def read_xml(path):
	try:
		tree = ET.parse(path)

		root = tree.getroot()
	except Exception as e:
		print('parse  xml document fail!')
		sys.exit()

	img_path = img_root+root.find('ImagePath').text

	num_obj = root.find('NumPerson')
	if num_obj is None:
		num_obj = root.find('NumParts')
	num_obj = int(num_obj.text)

	tmp_shape = (int(root.find('ImageHeight').text),int(root.find('ImageWidth').text))

	tmp_bbox_list = []
	for ind in xrange(num_obj):
		obj = root.find('Object_%d'% (ind+1))
		# only select the categry id ==0
		if obj.find('cid').text != '0':
			continue
		bbox = []
		bbox.append(int(obj.find('xmin').text))
		bbox.append(int(obj.find('ymin').text))
		bbox.append(int(obj.find('xmax').text))
		bbox.append(int(obj.find('ymax').text))
		tmp_bbox_list.append(bbox)

	return img_path, tmp_shape, tmp_bbox_list


#



# test one image and show its gt bbox
def test_single(img, ind, resize_shape, bboxes):
	print(img.shape)
	raw_im = img.copy()
	for bbox in bboxes:
		cv2.rectangle(raw_im, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0, 0, 255), 3)
	raw_im = cv2.resize(raw_im, resize_shape)
	img = cv2.resize(img, resize_shape)
	im = img - mean_data
	Net.blobs["data"].reshape(1, 3, im.shape[0], im.shape[1])
	Net.blobs["data"].data[...] = np.transpose(np.expand_dims(im, 0), (0, 3, 1, 2))
	Net.forward()
	det_out = Net.blobs[blob_name_detout].data
	det_out = det_out[0][0]
	print(det_out)

	for rec in det_out:
		if rec[0] + 1 > 0.01:
			cv2.rectangle(raw_im, (int(rec[3] * im.shape[1]), int(rec[4] * im.shape[0])),
						  (int(rec[5] * im.shape[1]), int(rec[6] * im.shape[0])), (0, 255, 0), 3)
			text = str(rec[2])
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(raw_im, text, (int(rec[3] * im.shape[1]), int(rec[4] * im.shape[0])), font, 0.5, (0, 0, 255), 1)
	cv2.imshow("tst%d"%ind, raw_im)


xml_path_list = []
with open(xml_list) as f:
	for name in f.readlines():
		xml_path_list.append(xml_root+name[:-1])

# for test_path in xml_path_list:
# 	image_path, image_shape, bboxes = read_xml(test_path)
# 	image = cv2.imread(image_path)
# 	if abs((image.shape[0] / float(image.shape[1])) - 9 / 16.) > 0.0001:
# 		print('skip')
# 		continue
# 	resize_shape = (512, 288)
# 	test_single(image, 0, resize_shape, bboxes)
# 	k = cv2.waitKey(-1)
# 	if k == 113:
# 		break





