import json
import matplotlib.pyplot as plt
import numpy

def parse_model_log(file_path):
	iteration = []
	smooth_loss = []
	with open(file_path,"r") as f:
		for line in f:
			if "$" in line and "train_smoothed_loss" in line:
				data = line.split("$")[1].replace("\n","").strip()
				data = json.loads(data)
				#info = add_data_from_log(data,info)
				iteration.append(int(data["Iteration"]))
				smooth_loss.append(float(data["Value"]))
	return iteration,smooth_loss

def main(path="",n_list=[1],start=10000):
	iterations,smooth_loss = parse_model_log(path)
	#smooth_loss = [1,2,3,4,5,6,7,8]
	#smooth_iter = 20 * n
	#smoothed_loss = []
	for n in n_list:
		smoothed_losses = []
		for i in range(n-1):
			temp_list = smooth_loss[:i+1]
			smoothed_losses.append(sum(temp_list)*1.0/len(temp_list))
		for i in range(n-1,len(smooth_loss)):
			start = i-n+1
			end = i+1
			temp_list = smooth_loss[start:i+1]
			smoothed_losses.append(sum(temp_list)*1.0/n)

		iteration = iterations[start/20:]
		smoothed_loss = smoothed_losses[start/20:]
		# if n >= 5:
		# 	var_iter = []
		# 	var_loss = []
		# 	for i in range(n-1,len(smoothed_loss)):
		# 		temp_list = smoothed_loss[i-4:i+1]
		# 		narray=numpy.array(temp_list)
		# 		sum1=narray.sum()
		# 		narray2=narray*narray
		# 		sum2=narray2.sum()
		# 		mean=sum1/n
		# 		var=sum2/n-mean**2
		# 		var_iter.append(iteration[i])
		# 		var_loss.append(var)
		# 	plt.plot(var_iter,var_loss,"-+") 
		# 	plt.xlabel("iterations")
		# 	plt.savefig("smoothed_loss_var_"+str(n*20)+".jpg")
		# 	plt.close()	

		plt.plot(iteration,smoothed_loss,"-+") 
		min_value = 10
		iteration_min = []
		smoothed_loss_min = []
		for i in range(len(iteration)):
			loss = smoothed_loss[i]
			if loss < min_value:
				min_value = loss
				iteration_min.append(iteration[i])
				smoothed_loss_min.append(smoothed_loss[i])
		plt.plot(iteration_min,smoothed_loss_min,"r+") 
		plt.xlabel("iterations")
		#plt.ylabel(key)
		#plt.legend(loc="best")
		plt.savefig(path+"_smoothed_loss_"+str(n*20)+".jpg")
		#plt.show()
		plt.close()	
		print "#"*5 + str(n*20) + "#"*5
		for i in range(len(smoothed_loss_min)-1):
			iter_before = iteration_min[i]
			iter_after = iteration_min[i+1]
			print "%6d   %6d   %6f" % (iter_before,iter_after-iter_before,smoothed_loss_min[i])


main("/home/yjh/Models/Model_comp/yolo_ssd.log",range(1,2))