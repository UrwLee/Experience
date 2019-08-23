import numpy as np
import os
import cv2

video_root = "./"

video_strings = ["muliscale","I_L"]
video_names = ["DetBD_DarkNet20180514MultiScaleNoBN_d3",
			   "DetBD_I_L_d3"]
img_w = 960#1280
img_h = 480#720
caps = []
for vid in video_names:
	vid_full = os.path.join(video_root,vid+".avi")
	caps.append(cv2.VideoCapture(vid_full))
num_videos = len(video_names)
flag = True
for cap in caps:
	flag = flag and cap.isOpened()
cnt = 0
while(flag):
	for i in xrange(num_videos):
		ret, frame = caps[i].read()
		cv2.putText(frame,str(cnt),(50,80),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame,(img_w,img_h))
		cv2.imshow(video_strings[i],frame)
	# if c/nt == 0:
	key = cv2.waitKey()
	# else:
	# 	cv2.waitKey(1)
	cnt += 1
	flag = True
	for cap in caps:
		flag = flag and cap.isOpened()
	if key == 27:
		break


