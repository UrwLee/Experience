import json
import sys
sys.dont_write_bytecode = True
sys.path.append('../')
from PyLib.Utils.path import *
from PyLib.Utils.par import *
from PyLib.Utils.dict import *
from PyLib.Utils.data import *
from PyLib.Utils.parse_log import *
from PyLib.Utils.plot import *
import matplotlib.pyplot as plt

parse_items = ["IOU@0.50/SIZE@0.00","IOU@0.50/SIZE@0.01","IOU@0.50/SIZE@0.05","IOU@0.50/SIZE@0.10","IOU@0.50/SIZE@0.15","IOU@0.50/SIZE@0.20","IOU@0.50/SIZE@0.25",\
               "IOU@0.75/SIZE@0.00","IOU@0.75/SIZE@0.01","IOU@0.75/SIZE@0.05","IOU@0.75/SIZE@0.10","IOU@0.75/SIZE@0.15","IOU@0.75/SIZE@0.20","IOU@0.75/SIZE@0.25",\
               "IOU@0.90/SIZE@0.00","IOU@0.90/SIZE@0.01","IOU@0.90/SIZE@0.05","IOU@0.90/SIZE@0.10","IOU@0.90/SIZE@0.15","IOU@0.90/SIZE@0.20","IOU@0.90/SIZE@0.25"]


def parse_log(path):
	file_list = []
	yolo_ssd_2 = "{}/yolo_ssd_2.log".format(path)	
	yolo_ssd_3 = "{}/yolo_ssd_3.log".format(path)
	yolo_ssd_neg_10 = "{}/yolo_ssd_neg_10.log".format(path)
	yolo_ssd_512_256_nodiff = "{}/yolo_ssd_512_256_nodiff.log".format(path)
	ssd = "{}/ssd.log".format(path)
	yolo = "{}/yolo.log".format(path)
	new = "{}/new.log".format(path)
	middle = "{}/middle.log".format(path)
	new_ssd = "{}/new_ssd.log".format(path)
	new_ssd_big = "{}/new_ssd_big.log".format(path)
	new_ssd_r = "{}/new_ssd_r.log".format(path)
	new_ssd_s = "{}/new_ssd_s.log".format(path)
	new_ssd_u_simple = "{}/new_ssd_u_simple.log".format(path)
	advanced_ssd = "{}/advanced_ssd.log".format(path)
	new_yolo_ssd_base = "{}/new_yolo_ssd_base.log".format(path)
	new_ssd_v = "{}/new_ssd_v.log".format(path)
	new_ssd_x = "{}/new_ssd_x.log".format(path)
	new_ssd_y = "{}/new_ssd_y.log".format(path)
	new_ssd_z = "{}/new_ssd_z.log".format(path)
	new_ssd_w_v_data = "{}/new_ssd_w_v_data.log".format(path)
	#file_list.append(["yolo_ssd_2",yolo_ssd_2])
	#file_list.append(["ssd",ssd])
	#file_list.append(["yolo_ssd_3",yolo_ssd_3])
	#file_list.append(["yolo",yolo])
	#file_list.append(["yolo_ssd_neg_10",yolo_ssd_neg_10])
	#file_list.append(["yolo_ssd_512_256_nodiff",yolo_ssd_512_256_nodiff])
	#file_list.append(["new",new])
	#file_list.append(["middle",middle])
	#file_list.append(["new_ssd",new_ssd])
	file_list.append(["new_ssd_big",new_ssd_big])
	#file_list.append(["new_ssd_r",new_ssd_r])
	file_list.append(["new_ssd_s",new_ssd_s])
	#file_list.append(["advanced_ssd",advanced_ssd])
	file_list.append(["new_yolo_ssd_base",new_yolo_ssd_base])
	#file_list.append(["new_ssd_u_simple",new_ssd_u_simple])
	#file_list.append(["new_ssd_v",new_ssd_v])
	#file_list.append(["new_ssd_x",new_ssd_x])
	file_list.append(["new_ssd_y",new_ssd_y])
	file_list.append(["new_ssd_z",new_ssd_z])
	#file_list.append(["new_ssd_w_v_data",new_ssd_w_v_data])

	result = {}
	for file in file_list:
		name = file[0]
		# if "2" in name:
		# 	continue
		file_path = file[1]
		result[name] = {"iteration":[]}

		for parse_item in parse_items:
			result[name][parse_item] = []

		i = 0
		with open(file_path,"r") as f:
			for line in f:
				if "$" in line and "mAP" in line:
					for parse_item in parse_items:
						if parse_item in line:			
							data = line.split("$")[1].replace("\n","").strip()
							data = json.loads(data)
							result[name][parse_item].append(float(data["Value"]))
							if i % len(parse_items) == 0:
								result[name]["iteration"].append(int(data["Iteration"]))
							i += 1
	#print result
	return result

def plot_multi_model(info,path,start,end):
	for parse_item in parse_items:
		for key,value in info.items():
			name = key
			#print len(value["iteration"][start:]),len(value[parse_item])
			for i in range(len(value["iteration"])):
				if value["iteration"][i] >= start * 1000:
					break
			plt.plot(value["iteration"][i:end],value[parse_item][i:end],"-+",label=name)
			#print value["iteration"]
		plt.xlabel("iterations")
		plt.ylabel("AP") 
		plt.legend(loc="best")
		#plt.ylim(0.7,1.0)
		plt.grid()
		plt.savefig(path+"/"+parse_item.replace("/","_")+".jpg")
		#plt.show()
		plt.close()	

def main(path,start,end):
	result = parse_log(path)
	plot_multi_model(result,path,start,end)

main("/home/yjh/Models/Model_comp",start=3,end=200000)








