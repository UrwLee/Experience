#encoding:utf-8
import sys
sys.path.insert(0,"/home/remo/caffe-master/python")
import caffe
import cv2

img = cv2.imread("/home/remo/Desktop/1.png")
img = cv2.resize(img,(28,28))
img = img[:,:,0]
print img.shape
net = caffe.Net("/home/remo/caffe-master/examples/mnist/test.prototxt","/home/remo/caffe-master/examples/mnist/lenet_iter_10000.caffemodel",caffe.TEST)
net.blobs["data"].reshape(1,1,28,28)
net.blobs["data"].data[...] = img
net.forward()
det_out = net.blobs["det_out"].data
print det_out
