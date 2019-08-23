import math
import cv2
import numpy as np
boxsizes = [[0.06,0.12],[0.18,0.24,0.32],[0.4,0.6,0.8,0.95]]
aspect_ratios = [[[1, 0.25, 0.5], \
                               [1, 0.25, 0.5]],
                               [[1, 0.25, 0.5], \
                               [1, 0.25, 0.5], \
                               [1, 0.25, 0.5]],\
                              [[1, 0.25, 0.5], \
                               [1, 0.25, 0.5], \
                               [1,0.25,0.5],        \
                               [1,0.25,0.5]]]
img_w = 512
img_h = 288
img_show = np.ones((img_h,img_w,3)).astype(np.uint8)*255
colors = [(0,0,255),(0,255,0),(255,0,0)]
for i in range(len(boxsizes)):
    boxsizes_per_layer = boxsizes[i]
    pro_widths_per_layer = []
    pro_heights_per_layer = []
    for j in range(len(boxsizes_per_layer)):
        boxsize = boxsizes_per_layer[j]
        # aspect_ratio = aspect_ratios[0]
        # if not len(aspect_ratios) == 1:
        aspect_ratio = aspect_ratios[i][j]
        for each_aspect_ratio in aspect_ratio:
            w = boxsize * math.sqrt(each_aspect_ratio)
            h = boxsize / math.sqrt(each_aspect_ratio)
            w = min(w, 1.0)
            h = min(h, 1.0)
            pro_widths_per_layer.append(w)
            pro_heights_per_layer.append(h)
    pro_widths_per_layer_fixed = [int(wi*img_w) for wi in pro_widths_per_layer]
    pro_heights_per_layer_fixed = [int(hi*img_h) for hi in pro_heights_per_layer]
    print  "width",i, pro_widths_per_layer_fixed
    print  "height",i, pro_heights_per_layer_fixed
    for i_pro in xrange(len(pro_widths_per_layer)):
        xmin = 0.5 - pro_widths_per_layer[i_pro] / 2.0
        xmax = 0.5 + pro_widths_per_layer[i_pro] / 2.0
        ymin = 0.5 - pro_heights_per_layer[i_pro] / 2.0
        ymax = 0.5 + pro_heights_per_layer[i_pro] / 2.0
        xmin = int(xmin * img_w)
        xmax = int(xmax * img_w)
        ymin = int(ymin * img_h)
        ymax = int(ymax * img_h)
        img_show = cv2.rectangle(img_show,(xmin,ymin),(xmax,ymax),color=colors[i],thickness=1)
        index = np.where(img_show!=0)
        cv2.imshow("img_show",img_show)
cv2.waitKey()

