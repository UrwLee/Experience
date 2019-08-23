#encoding:utf/8
import xml.dom.minidom
from xml.dom.minidom import Document
import os
import cv2

def read_remoanno_xml(xml_path, cid=1, check_area=10, flag_giveup = False): #  cid=1 (hand), "path/xxx.xml"
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    meta = {}
    # --------------------------------------------------------------------------
    # ImagePath
    image_path_node = root.getElementsByTagName('filename')[0]
    image_path = image_path_node.childNodes[0].data
    meta['image_path'] = str(image_path)+'.jpg'
    # width & height
    width_node = root.getElementsByTagName('size')[0].getElementsByTagName('width')[0]
    height_node = root.getElementsByTagName('size')[0].getElementsByTagName('height')[0]
    width = int(width_node.childNodes[0].data)
    height = int(height_node.childNodes[0].data)
    meta['width'] = width
    meta['height'] = height
    area_Img = float(width)*float(height)
    # num of person
    num_person_nodes = root.getElementsByTagName('object')
    num_person = len(num_person_nodes)
    meta['boxes'] = []
    # --------------------------------------------------------------------------
    for i in range(num_person):
        pnode = num_person_nodes[i]
        box = {}
        box['cid'] = cid
        xmin = int(pnode.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(pnode.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(pnode.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(pnode.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].childNodes[0].data)
        # active check
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0 or xmin < 0 or xmax <= 0 or ymin < 0 or ymax <= 0 or xmin > width or xmax > width or ymin > height or ymax > height:
            continue
        if check_area < 1.0:
            check_area = area_Img * check_area
        if bw * bh < check_area:
            if flag_giveup:
                break
            else:
                continue
        box['xmin'] = xmin
        box['ymin'] = ymin
        box['xmax'] = xmax
        box['ymax'] = ymax
        meta['boxes'].append(box)
    meta['num'] = len(meta['boxes'])
    return meta


img_root = "/home/remo/Downloads/Images/"
anno_root = "/home/remo/Downloads/Annotation/"
anno_dirs = os.listdir(img_root)
for anno_dir in anno_dirs:
    annos_path = os.path.join(anno_root,anno_dir)
    annos = os.listdir(annos_path)
    #print xmls
    for anno in annos:
        anno_path = os.path.join(annos_path,anno)
        pic_path = os.path.join(img_root,anno_dir,anno)[:-3]+'jpg'
        #print pic_path
        meta = {}
        meta = read_remoanno_xml(anno_path)
        # print pic_path
        try:
            img = cv2.imread(pic_path)
            cv2.imshow('dsda',img)
        except:
            print pic_path
        boxes = meta['boxes']
        #print boxes[0]
        xmin = boxes[0]['xmin']
        ymin = boxes[0]['ymin']
        xmax = boxes[0]['xmax']
        ymax = boxes[0]['ymax']
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),5)
        cv2.imshow('im',img)
        cv2.waitKey(0 )

