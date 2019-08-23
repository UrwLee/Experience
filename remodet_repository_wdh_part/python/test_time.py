import time
import sys
sys.path.append("/home/ethan/work/remodet_repository/python")
from utils.cython_bbox import bbox_overlaps
import numpy as np
def compute_iou(box1,box2):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1)*(y2_1 - y1_1)
    area2 = (x2_2 - x1_2)*(y2_2 - y1_2)
    # print area_inter,area1,area2,float(area_inter)/float(area1)/float(area2)
    return float(area_inter)/float(area1+ area2 - area_inter)
anchors = np.random.random((25000,4))
gt = np.random.random((5,4))
t1 = time.time()
overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(gt, dtype=np.float))
t2 = time.time()
print t2 - t1

overlaps = np.zeros((anchors.shape[0],gt.shape[0]))
t1 = time.time()
for ianchor in xrange(anchors.shape[0]):
    for igt in xrange(gt.shape[0]):
        iou = compute_iou(anchors[ianchor], gt[igt])
        overlaps[ianchor][igt] = iou
t2 = time.time()
print t2 - t1