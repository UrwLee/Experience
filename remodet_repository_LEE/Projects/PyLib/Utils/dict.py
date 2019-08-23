# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True
def cmp_dict(src_data,dst_data):
    # 比较两个字典是否完全一致，若一致返回True
    flag = True
    if not (type(src_data) == type(dst_data)):
        flag = False
    else:
        if isinstance(src_data,dict):
            if not (len(src_data) == len(dst_data)):
                flag = False
            else:
                for key in src_data:
                    if not dst_data.has_key(key):
                        flag = False
                        break
                    else:
                        flag = (flag and cmp_dict(src_data[key],dst_data[key]))

        elif isinstance(src_data,list):
            if not (len(src_data) == len(dst_data)):
                flag = False
            else:
                for src_list, dst_list in zip(sorted(src_data), sorted(dst_data)):
                    flag = (flag and cmp_dict(src_list, dst_list))
                    if not flag:
                        break
        else:
            if not (src_data == dst_data):
                flag = False
    return flag

def get_diff_keys(cover_keys,cmp_dict,current_dict):
    # 不同模型参数比较，参数相同的键放入same_keys中，不同的放入diff_keys，便于csv文件中参数顺序的排列
    cmp_keys = cmp_dict.keys()
    current_keys = current_dict.keys()
    diff_keys = []
    same_keys = []
    for key in cover_keys:
        if (key in cmp_keys) and (key in current_keys):
            if cmp_dict[key] == current_dict[key]:
                same_keys.append(key)
            else:
                diff_keys.append(key)
        else:
            diff_keys.append(key)
    return diff_keys,same_keys