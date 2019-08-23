# coding=utf-8

import os
import caffe
import yaml
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
import json
DEBUG = False
import time
import sys
print sys.path


class LossTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        ##bottom[0],conf_pred: Nx(2xnum_anchors)
        ##bottom[1], anchors: 1x2x(4xnum_anchors)
        ##bottom[2], label:1x1xnum_gtx9 (0:batchid; 1:cid; 2:pid; 3:is_diff; 4: iscrowd; 5: x1_; 6: y1_;7: x2_; 8: y2_.
        print bottom[0].data.shape
        print bottom[1].data.shape
        print bottom[2].data.shape

        layer_params = yaml.load(self.param_str)
        self._layer_params = layer_params
        self._batchsize = bottom[0].data.shape[0]#loc_pred: Nx(4xnum_anchors)
        self._num_anchors = bottom[1].data.shape[2]/4
        self._neg_overlap = float(self._layer_params.get("neg_overlap"))
        self._overlap_threshold = float(self._layer_params.get("overlap_threshold"))
        self._neg_pos_ratio = float(self._layer_params.get("neg_pos_ratio"))
        self._loc_weight = float(self._layer_params.get("loc_weight"))
        self._conf_weight = float(self._layer_params.get("conf_weight"))
        num_anchors = bottom[0].data.shape[1]/2

        assert num_anchors == self._num_anchors

        # labels
        top[0].reshape(self._batchsize, 2*self._num_anchors)
        # bbox_targets
        top[1].reshape(self._batchsize,4*self._num_anchors)
        # bbox_inside_weights 回归参数
        top[2].reshape(self._batchsize,4*self._num_anchors)
        # bbox_outside_weights 回归参数
        top[3].reshape(self._batchsize,4*self._num_anchors)
        # class_weight
        top[4].reshape(self._batchsize, 2 * self._num_anchors)
        # class_normalizer
        top[5].reshape(1)

    def reshape(self, bottom, top):
        self._batchsize = bottom[0].data.shape[0]  # loc_pred: Nx(4xnum_anchors)
        self._num_anchors = bottom[1].data.shape[2] / 4
        num_anchors = bottom[0].data.shape[1] / 2
        assert num_anchors == self._num_anchors

        # labels
        top[0].reshape(self._batchsize, 2 * self._num_anchors)
        # bbox_targets
        top[1].reshape(self._batchsize, 4 * self._num_anchors)
        # bbox_inside_weights 回归参数
        top[2].reshape(self._batchsize, 4 * self._num_anchors)
        # bbox_outside_weights 回归参数
        top[3].reshape(self._batchsize, 4 * self._num_anchors)
        # class_weight
        top[4].reshape(self._batchsize, 2 * self._num_anchors)
        # class_normalizer
        top[5].reshape(1)


    def forward(self, bottom, top):

        t1 = time.time()
        self._batchsize = bottom[0].data.shape[0]
        conf_pred = bottom[0].data.reshape((self._batchsize,self._num_anchors,2))
        conf_pred = _sigmoid(conf_pred)
        gt_labels = bottom[2].data #1x1xnum_gtx9: (0:batchid; 1:cid; 2:pid; 3:is_diff; 4: iscrowd; 5: x1_; 6: y1_;7: x2_; 8: y2_).
        anchors = bottom[1].data[0, 0, :].reshape(self._num_anchors, 4).astype(np.float)
        labels = np.empty((self._batchsize, self._num_anchors), dtype=np.float32)
        labels.fill(-1)
        bbox_targets_all = []
        # print self._neg_overlap,self._overlap_threshold,self._neg_pos_ratio
        # for i in xrange(self._num_anchors):
        #     print anchors[i]
        #     raw_input()
        for ib in xrange(self._batchsize):
            flag = np.bitwise_and(gt_labels[0,0,:,0]==ib, gt_labels[0,0,:,1]==0)
            gt_batch_i = gt_labels[0,0,flag,5:9].astype(np.float)
            if gt_batch_i.size == 0:
                bbox_targets_all.append(np.zeros((self._num_anchors,4)))
            else:

                overlaps = bbox_overlaps(
                    # 生成符合c编码的连续内存
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(gt_batch_i, dtype=np.float))
                # 全为0的向量，长度和overlaps一样
                # max gt ids for each anchor, len(argmax_overlaps) = K
                argmax_overlaps = overlaps.argmax(axis=1)
                # 取overlaps所有行argmax_overlaps的那一个（其实就是第一个）
                max_overlaps = overlaps[np.arange(self._num_anchors), argmax_overlaps]
                # 取和gt相交最大的IOU位置
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                # 取和gt相交最大的IOU值
                gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                           np.arange(overlaps.shape[1])]
                # 找到最大IOU多次出现的位置
                # get all the
                gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
                # print "compare", (max_overlaps >= self._overlap_threshold).sum(), (
                # max_overlaps_new >= self._overlap_threshold).sum()
                # print overlaps[0:50],overlaps_new[0:50]
                # print ib,len(gt_argmax_overlaps),overlaps.min(),overlaps.max(),(max_overlaps >= self._overlap_threshold).sum(),max_overlaps.shape,overlaps.shape
                labels[ib,gt_argmax_overlaps] = 1
                labels[ib, max_overlaps >= self._overlap_threshold] = 1
                labels[ib, max_overlaps < self._neg_overlap] = 0
                # print ib,(labels[ib,:]==0).sum(),(labels[ib,:]==1).sum(),(labels[ib,:]==-1).sum(),labels[ib,:].size
                bbox_targets = _compute_targets(anchors, gt_batch_i[argmax_overlaps, :])
                bbox_targets_all.append(bbox_targets)

        num_pos = len(np.where(labels == 1)[0])

        ids_neg_rows, ids_neg_cols = np.where(labels == 0)
        scores_neg = conf_pred[:,:,1][labels == 0]
        labels[labels == 0] = -1
        ids_sort_neg = np.argsort(-scores_neg) # hard negative
        num_neg = int(num_pos*self._neg_pos_ratio)

        id_rows = ids_neg_rows[ids_sort_neg[:num_neg]]
        id_cols = ids_neg_cols[ids_sort_neg[:num_neg]]
        labels[id_rows, id_cols] = 0
        # for i in xrange(int(num_pos*self._neg_pos_ratio)):
        #
        #     id_row = ids_neg_rows[ids_sort_neg[i]]
        #     id_col = ids_neg_cols[ids_sort_neg[i]]
        #     labels[id_row,id_col] = 0

        bbox_inside_weights = np.zeros((self._batchsize,self._num_anchors,4))
        bbox_inside_weights[labels == 1,:] = 1
        bbox_outside_weights = np.ones((self._batchsize, self._num_anchors, 4))
        bbox_outside_weights *= (self._loc_weight/float(num_pos))
        class_weights = np.ones((self._batchsize,self._num_anchors,2))
        class_weights[labels == -1,:] = 0
        class_labels = np.ones((self._batchsize,self._num_anchors,2))
        class_labels[labels== 0,0] = 1
        class_labels[labels==1, 1] = 1
        bbox_targets_all = np.array(bbox_targets_all)
        # labels
        top[0].data[...] = class_labels.reshape((self._batchsize,-1))
        # bbox_targets
        top[1].data[...] = bbox_targets_all.reshape((self._batchsize,-1))
        # bbox_inside_weights 回归参数
        top[2].data[...] = bbox_inside_weights.reshape((self._batchsize,-1))
        # bbox_outside_weights 回归参数
        top[3].data[...] = bbox_outside_weights.reshape((self._batchsize,-1))
        # class_weight
        top[4].data[...] = class_weights.reshape((self._batchsize,-1))
        # class_scale
        top[5].data[...] = self._conf_weight/float(num_pos + num_neg)




    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass



def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    targets = _bbox_transform(ex_rois, gt_rois[:, :4]).astype(
        np.float32, copy=False)
    # true

    targets -= np.array(([0,0,0,0]))
    targets /= np.array(([0.1,0.1,0.2,0.2]))
    return targets
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
def _bbox_transform(ex_rois, gt_rois):
    """
    dx = (Gx-Ex)/Ew
    dx = (Gy-Ey)/Eh
    dw = log(Gw/Ew)
    dh = log(Gh/Eh)
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
def _compute_iou(box1,box2):
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