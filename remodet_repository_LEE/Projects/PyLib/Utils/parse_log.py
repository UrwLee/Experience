# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True
import os
import json
from data import *
from path import * 
from plot import LOG_INFO

def getloglist(log_dir_path):
	# 得到当前日志文件夹下以.log结尾且未被解析过得日志文件名称列表
	result = []
	loglist = os.listdir(log_dir_path)
	for item in loglist:
		if item.endswith(".log") and "Parsed" not in item:
			result.append(item)
	return result

def add_data_from_log(data,info):
	# 将日志文件中解析出数据汇总到给定字典info中
	Type = data["Type"]
	Iteration = data["Iteration"]
	Key = data["Key"]
	Value = data["Value"]
	if Type in info:
		type_info = info[Type]
		if Iteration in type_info:
			iteration_info = type_info[Iteration]
			if Key in iteration_info:
				key_info = iteration_info[Key]
				if key_info != Value:
					print("Type:{},Iteration:{},Key:{}--Value comflics")
			else:
				info[Type][Iteration][Key]=Value
				#return info
		else:
			info[Type][Iteration]={Key:Value}
			#return info
	else:
		info[Type]={Iteration:{Key:Value}}
	return info 

def parse_model_log(file_path,info):
	# 解析单个日志文件的所有数据，并将数据添加到info中
	with open(file_path,"r") as f:
		for line in f:
			if "$" in line and "test" not in line:
				data = line.split("$")[1].replace("\n","").strip()
				data = json.loads(data)
				info = add_data_from_log(data,info)
	return info

def parse_model_logs(log_dir_path):
	# 解析多个日志文件的所有数据，并将数据添加到info中。解析过得文件增加前缀“Parsed”，info结果存储在日志文件夹下的log_data.txt中，并将结果返回。
	loglist = getloglist(log_dir_path)
	#print "Loglist",loglist
	if len(loglist) == 0:
		return 0

	info = {}
	log_data_file = "{}/log_data.txt".format(log_dir_path)
	if check_if_exist(log_data_file):
		with open(log_data_file,"r") as f:
			for line in f:
				info = json.loads(line)
				break

	for logfile in loglist:
		logfile_path = "{}/{}".format(log_dir_path,logfile)
		info = parse_model_log(logfile_path,info)
		logfile_parsed = "{}/{}_{}".format(log_dir_path,"Parsed",logfile)
		os.rename(logfile_path,logfile_parsed)
	#print info
	savedata(info,log_data_file)

	data_file_path = log_dir_path.replace("Logs","Pics") + "/pics_data.txt"
	result = change_data_format_for_plot(info)

	savedata(result,data_file_path)
	return result

def change_data_format_for_plot(info):
	# 将从日志中汇总得到的数据转换存储形式，便于接下来的画图
    result = {}
    #print info
    for key,value in info.items():  
        data = {"Iterations":[]}
        log_info = LOG_INFO[key]
        #keys = log_info["keys"]
        key_nums = log_info["key_nums"]
        value_sort = sorted(value.iteritems(),key=lambda x:int(x[0]),reverse = False)
        for item in value_sort:
            Iteration = item[0]
            value_dict = item[1]
            if len(value_dict) != key_nums:
            	print len(value_dict),"wrong",key
                continue
            else:
            	data["Iterations"].append(int(Iteration))
                for subkey,subvalue in value_dict.items():
                    if subkey in data:
                        data[subkey].append(float(subvalue))
                    else:
                        data[subkey] = [float(subvalue)]
        result[key] = data
    return result