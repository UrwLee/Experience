# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True
import json
def get_model_params_csv(keys,data,path):
    # 按照给定的键顺序，将数据转换成csv形式。后一排与前一排相比，相同的填“-”，不同的填当前值，不含该键的填“/”。
    row = len(data)
    del data["state"]
    data = sorted(data.iteritems(),key=lambda x:x[0],reverse = False)
    f = open(path,"w")
    line_first = "Params,"

    for item in data:
        col_name = item[0]
        line_first += "{},".format(col_name)
    line_first += "\n"
    f.write(line_first)

    for key in keys:
        line_content = "{},".format(key)
        for item in data:
            col_name = item[0]
            col_dict = item[1]
            if col_name == 0:
                if key in col_dict:
                    current_value = str(col_dict[key]).replace(","," ")
                else:
                    current_value = "/"
                line_content += "{},".format(current_value)
            else:
                last_value = current_value
                if key in col_dict:
                    current_value = str(col_dict[key]).replace(","," ")
                else:
                    current_value = "/"
                if last_value == "/":
                    line_content += "{},".format(current_value)
                else:
                    if last_value == current_value:
                        line_content += "{},".format("-")
                    else:
                        line_content += "{},".format(current_value)

        line_content += "\n"
        f.write(line_content)

    f.close()

def savedata(data,data_save_path):
    # 将数据保存在指定路径下
    with open(data_save_path,"w") as f:
        f.write(json.dumps(data))