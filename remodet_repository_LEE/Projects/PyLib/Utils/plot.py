# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import json

LOG_INFO = {"Loss":
                {#"keys":["train_smoothed_loss","train_loss", "train_loc_loss","train_conf_loss"],
                 "keys":["train_smoothed_loss","train_loss"],
                 "key_nums":2,
                 "pic_keys":[],
                 "plot_policy":"struct",
                 "struct":[["",["train_smoothed_loss","train_loss"]],["",["train_loss"]],["",["train_smoothed_loss"]]]
                },
            "Train_lr":
                {"keys":["train_lr"],
                 "key_nums":1,
                 "pic_keys":[],
                 "plot_policy":"struct",
                 "struct":[["",["train_lr"]]],
                },
            "AP":
                {"keys":{},
                 "key_nums":7*3*1,
                 "pic_keys":[["IOU","CAT"]],
                 "plot_policy":"auto",
                 "struct":[],
                },
            "mAP":
                {"keys":{},
                 "key_nums":7*3,
                 "pic_keys":[["IOU"]],
                 "plot_policy":"auto",
                 "struct":[], 
                },
            }

# NAME_INFO = {"IOU":["0.90","0.75","0.50"],
#            "SIZE":["0.00","0.01","0.05","0.10","0.15","0.20","0.25"],
#            "CAT":["person"],
#           }

def get_plot_struct(keys):
    # 对AP，mAP类型的数据按照键名分组得到struct
    result = {}
    for key in keys:
        key_split = key.split("/")
        if len(key_split) == 3:
            key_name = key_split[0] + "+" + key_split[2]
        else:
            key_name = key_split[0]
        if key_name in result:
            result[key_name].append(key)
        else:
            result[key_name] = [key]
    return result


def plot_pics(data={},flag="data",save_pics_path="",loss_begin=0,loss_end=10000000,prefix="0"):
    # 绘制图片主函数，可以直接绘制数据，或者从文件中得到数据
    if flag == "file":
        data_file_path = "{}/pics_data.txt".format(save_pics_path)
        with open(data_file_path,"r") as f:
            for line in f:
                datas = json.loads(line)
                break
    else:
        datas = data

    for key,value in datas.items():
        log_info = LOG_INFO[key]
        plot_policy = log_info["plot_policy"]

        if plot_policy == "auto":
            keys = value.keys()
            #pic_keys = log_info["pic_keys"]
            struct_dict = get_plot_struct(keys)
            del struct_dict["Iterations"]

        else:       
            keys = log_info["keys"]
            struct_list = log_info["struct"]
            struct_dict = {}
            for struct in struct_list:
                name = struct[0]
                items = struct[1]
                if name == "":
                    for item in items:
                        if name == "":
                            name = item
                        else:
                            name = name + "_vs_" + item
                struct_dict[name] = items

        iterations = value["Iterations"]
        start = 0
        end = len(iterations)
        #print iterations[-1]
        if iterations[-1] > loss_begin:
            for i in range(len(iterations)):
                if iterations[i] >= loss_begin:
                    start = i
                    break
            for j in range(len(iterations)-1,i+1,-1):
                #print j,iterations[j]
                if iterations[j] <= loss_end:
                    end = j
                    break
        #print iterations[i],iterations[j]
        for pic_name,pic_items in struct_dict.items():
            # xlim_min = 10000
            # #xlim_min = min(iterations)
            # plt.xlim(xlim_min,max(iterations))
            # if key == "Loss":
            #     plt.ylim(0,5)
            for item in pic_items:  
                current_value = value[item]
                if "/" in item:
                    name_split = item.split("/")
                    item = name_split[1].split("@")[1]

                plt.plot(iterations[start:end],current_value[start:end],"-+",label=item) 
            plt.xlabel("iterations")
            plt.ylabel(key)
            plt.legend(loc="best")
            plt.grid()
            plt.savefig(save_pics_path+"/"+prefix+"_"+key+"("+pic_name+").jpg")
            #plt.show()
            plt.close()




 