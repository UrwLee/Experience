import caffe
import cv2
import numpy as np
train_proto = "/home/ethan/Models/Results/HPKeypoint/CNNAll64C_Sigma5RT40_1A/Proto/train.prototxt"
net = caffe.Net(train_proto,caffe.TRAIN)
for i in xrange(100):
    net.forward()
    data = net.blobs["data"].data
    label_heat = net.blobs["label_heat"].data
    N = data.shape[0]
    for n in xrange(N):
        data_img = data[n]
        data_img[0] += 104
        data_img[1] += 117
        data_img[2] += 123
        data_img = data_img.transpose((1, 2, 0))
        img = np.array(data_img).astype(np.uint8)
        img = cv2.resize(img, (512, 512))

        label  = label_heat[n]
        for ic in xrange(label.shape[0]):
            li = label[ic]
            max_v = li.max()
            print ic,max_v
            if max_v>0:
                ys,xs=np.where(li == max_v)
                for ix in xrange(len(xs)):
                    cv2.circle(img,(xs[ix]*8,ys[ix]*8),5,(0,0,255),-1)


            # max_w=np.max(w)
            # ys,xs=np.where(w==max_w)
            # for imax in xrange(len(xs)):
            #     data_img[ys[imax],xs[imax],:] = [0,0,255]
        cv2.imshow("a", img)
        cv2.waitKey()


    # exit()