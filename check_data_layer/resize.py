import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document

root = "/home/zhangming/SSD_DATA/Datasets/RemoBlackHandWithoutFace201957/imgs_RemoBlackHandWithoutFace201957"
xml_root = "/home/zhangming/SSD_DATA/Datasets/RemoBlackHandWithoutFace201957/XML"
resize_root = "/home/zhangming/SSD_DATA/Datasets/RemoBlackHandWithoutFace201957/resize_imgs_RemoBlackHandWithoutFace201957"
new_xml_root = "/home/zhangming/SSD_DATA/Datasets/RemoBlackHandWithoutFace201957/new_xml"
imgs = os.listdir(root)

for im in imgs:
    xml = im.replace('jpg','xml')
    im_path = os.path.join(root,im)
    xml_path = os.path.join(xml_root,xml)
    image = cv2.imread(path)
    w = image.shape[1]
    h = image.shape[0]
    image = cv2.resize(image,(768,432))
    cv2.imwrite(os.path.join(resize_root,im))
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    num_gt = root.getElementsByTagName('NumPerson')[0].firstChild.data
    for j in range(1,int(num_gt)+1):
        xmin = root.getElementsByTagName('Object_'+str(j))[0].getElementsByTagName('xmin')[0]
        xmax = root.getElementsByTagName('Object_'+str(j))[0].getElementsByTagName('xmax')[0]
        ymin = root.getElementsByTagName('Object_'+str(j))[0].getElementsByTagName('ymin')[0]
        ymax = root.getElementsByTagName('Object_'+str(j))[0].getElementsByTagName('ymax')[0]
        xmin.firstChild.data = xmin.firstChild.data/w*768
        xmax.firstChild.data = xmax.firstChild.data/w*768
        ymin.firstChild.data = ymin.firstChild.data/h*432
        ymax.firstChild.data = ymax.firstChild.data/h*432
        #修改cid
        #cid.firstChild.data = 3-int(cid.firstChild.data)
        #将修改后的xml文件保存
    with open(os.path.join(new_xml_root, xml), 'w') as fh:
        dom.writexml(fh)
        print('写入name/pose OK!')



