import numpy as np
import cv2
import os


test_img_path = '/home/remo/cat_dog_data/OID/Images'
test_img_pathes = [test_img_path+'/'+pa for pa in os.listdir(test_img_path)]


def adjust_channel(img, param):
	img_res = img.copy()
	param = int(param)
	if param == 0:
		return img_res
	elif param == 1:
		return np.stack([img_res[..., 0], img_res[..., 2], img_res[..., 1]], axis=-1)
	elif param == 2:
		return np.stack([img_res[..., 1], img_res[..., 0], img_res[..., 2]], axis=-1)
	elif param == 3:
		return np.stack([img_res[..., 1], img_res[..., 2], img_res[..., 0]], axis=-1)
	elif param == 4:
		return np.stack([img_res[..., 2], img_res[..., 0], img_res[..., 1]], axis=-1)
	elif param == 5:
		return np.stack([img_res[..., 2], img_res[..., 1], img_res[..., 0]], axis=-1)


def adjust_bright(img, param):
	img_res = img.copy().astype(np.int)
	img_res += param
	img_res = np.clip(img_res, 0, 255).astype(np.uint8)
	return img_res


def adjust_contrast(img, param):
	img_res = img.copy().astype(np.float32)
	img_res *= param
	img_res = np.clip(img_res+0.5, 0, 255).astype(np.uint8)
	return img_res


def adjust_hue(img, param):
	img_res = img.copy()
	img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
	img_res[..., 0] = np.clip(img_res[..., 0].astype(np.int)+param, 0, 255).astype(np.uint8)
	img_res = cv2.cvtColor(img_res, cv2.COLOR_HSV2BGR)
	return img_res


def adjust_saturation(img, param):
	img_res = img.copy()
	img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
	img_res[..., 1] = np.clip(img_res[..., 1].astype(np.int)*param+0.5, 0, 255).astype(np.uint8)
	img_res = cv2.cvtColor(img_res, cv2.COLOR_HSV2BGR)
	return img_res


def gamma_correct(img, gamma):
	table = np.zeros((256, ), dtype=np.uint8)
	inv_gamma = 1./gamma
	img_res = img.copy()
	for i in range(256):
		table[i] = np.minimum(pow(i/255., inv_gamma)*255., 255)
	cv2.LUT(img, table, img_res)
	return img_res


def show_img(pathes, funcc, param_init, interval, step):
	for img_path in pathes:
		img = cv2.imread(img_path)
		print(img.shape)
		param = param_init
		cv2.namedWindow("a", cv2.NORM_HAMMING)
		cv2.resizeWindow("a", 1920, 1080)
		while(1):
			param = np.clip(param, interval[0], interval[1])
			img_res = funcc(img, param)
			print(img_res.shape)
			cv2.putText(img_res, 'param%.4f' % param, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
			cv2.imshow('a', img_res)
			key = cv2.waitKey(0)
			if key==ord('q'):
				exit(0)
			if key==ord('b'):
				param=param+step
				continue
			if key==ord('s'):
				param = param-step
			if key==ord('n'):
				break


show_img(test_img_pathes, gamma_correct, 1, [0.5, 1.0], 0.01)
# show_img(test_img_pathes, adjust_contrast, 1, [0.5, 1.5], 0.05)
# show_img(test_img_pathes, adjust_bright, 0, [-40, 40], 2)
# show_img(test_img_pathes, adjust_hue, 0, [-20, 20], 2)
# show_img(test_img_pathes, adjust_saturation, 1, [0.5, 1.5], 0.05)
# show_img(test_img_pathes, adjust_channel, 0, [0, 5], 1)
