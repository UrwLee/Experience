# import caffe
import h5py
import numpy as np
# proto_file = "train.prototxt"
# net = caffe.Net(proto_file,caffe.TRAIN)
# net.forward()
fh = h5py.File("aa.h5","w")
fh.create_dataset("data",data=np.random.random((600000,128)))
fh.close()
