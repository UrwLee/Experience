import sys
sys.dont_write_bytecode = True
sys.path.append('../')
from PyLib.Utils.path import *
from PyLib.Utils.par import *
from PyLib.Utils.dict import *
from PyLib.Utils.data import *
from PyLib.Utils.parse_log import *
from PyLib.Utils.plot import *


# def plot_for_file(file_path="/home/yjh/work/caffe/Projects/Results/Res50/VOC/SSD/300/Base/0/Pics",loss_begin=0,loss_end=1000000):
# 	plot_pics(flag="file",save_pics_path=file_path,loss_begin=loss_begin)

def main(path_base="/home/yjh/work/caffe/Projects/Results/Res50/VOC/SSD/300/Base/3",begin=20000,end=1000000):
	#train()
	#path_base = "/home/yjh/work/caffe/Projects/Results/Res50/VOC/SSD/300/Base/0"
	prefix = path_base.split("/")[-1]
	path_log = "{}/Logs".format(path_base)
	path_data = "{}/Pics".format(path_base)
	make_if_not_exist(path_data)
	info = parse_model_logs(path_log)
	if info:
		plot_pics(data=info,save_pics_path=path_data,loss_begin=begin,loss_end=end,prefix=prefix)
	else:
		plot_pics(flag="file",save_pics_path=path_data,loss_begin=begin,loss_end=end,prefix=prefix)

main("/home/yjh/Models/Results/100/Yolo/VOC_COCO/YOLO_SSD/416/Base/8",20000,3500000)
#main("/home/yjh/Models/Results/200/Yolo/VOC_COCO/SSD/416/Base/3",20000,3500000)
#plot_for_file("/home/yjh/work/caffe/Projects/Results/PVA/VOC_COCO/YOLO/416/Base/15/Pics",20000)
