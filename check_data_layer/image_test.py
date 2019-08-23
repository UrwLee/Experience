import cv2
import numpy as np
import glob
image_path = '/home/remo/Desktop/mao/'
image_dir = glob.glob(image_path + '*')

img = "/home/remo/from_wdh/data/val2017/000000058111.jpg"
cv2.namedWindow("a", cv2.NORM_HAMMING)
cv2.resizeWindow("a", 960, 540)

im = cv2.imread(img)
# im = cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_WRAP)
cv2.imshow('a',im)
k = cv2.waitKey(0)
if k == ord('q'):
    cv2.destroyAllWindows()
