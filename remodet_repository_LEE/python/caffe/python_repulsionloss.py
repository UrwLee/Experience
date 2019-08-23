import unittest
import tempfile
import os
import six
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import caffe


class RepulsionLossLayer(caffe.Layer):
    """A layer that just multiplies by the numeric value of its param string"""

    def setup(self, bottom, top):
        self.IoG_gt = np.zeros((1,4))
        self.IoU_const = np.zeros((1,4))
        self.sigma =0
        self.target_labels = [0,]
        self.num_classes = len(self.target_labels) + 1
        self.prior_variances = [0.1,0.1,0.2,0.2]
        try:
            self.value = float(self.param_str)
        except ValueError:
            raise ValueError("Parameter string must be a legible float")

    def reshape(self, bottom, top):
        shape = (1,)
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        #bottom: "mbox_1_loc" Nx(num_priorsx4)
        #bottom: "mbox_1_conf" Nx(num_priorsx2)
        #bottom: "mbox_1_priorbox" 1x2x(num_priorsx4)
        #bottom: "label_det" (1x1xnum_gtx9)
        #   0       1      2       3       4       5       6       7       8
        #  bid     cid    pid  is_diff   is_crow   x1      y1      x2       y2
        Nbatch = bottom[0].data.shape[0]

        num_gt = labels.shape[2]
        self.GTs_All = np.zeros((num_gt,6))
        num_priors = bottom[2].data.shape[2]/4
        self.loc_preds =  bottom[0].data.reshape((Nbatch,num_priors,4))
        self.conf_preds = bottom[2].data.reshape((Nbatch, num_priors, self.num_classes))
        self.priorboxes = bottom[2].data.reshape((1, 2, num_priors, 4))
        labels = bottom[3].data
        for i in xrange(num_gt):
            cid = labels[0,0,i,1]
            if cid in self.target_labels:
                bid = labels[0, 0, i, 0]
                x1 = labels[0, 0, i, 5]
                y1 = labels[0, 0, i, 6]
                x2 = labels[0, 0, i, 7]
                y2 = labels[0, 0, i, 8]
                self.GTs_All[i,0] = cid
                self.GTs_All[i, 1] = bid
                self.GTs_All[i, 2] = x1
                self.GTs_All[i, 3] = y1
                self.GTs_All[i, 4] = x2
                self.GTs_All[i, 5] = y2
        self.max_scores = np.max(self.conf_preds[:,:,1:],axis=2)#only used for logistic loss





        top[0].data[...] = self.value * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.value * top[0].diff
    def IoG(self,box):
        inter_xmin = np.maximum(box[:, 0], self.IoG_gt[:, 0])
        inter_ymin = np.maximum(box[:, 1], self.IoG_gt[:, 1])
        inter_xmax = np.minimum(box[:, 2], self.IoG_gt[:, 2])
        inter_ymax = np.minimum(box[:, 3], self.IoG_gt[:, 3])
        Iw = np.clip(inter_xmax - inter_xmin, 0, 5)
        Ih = np.clip(inter_ymax - inter_ymin, 0, 5)
        I = Iw * Ih
        G = (self.IoG_gt[:, 2] - self.IoG_gt[:, 0]) * (self.IoG_gt[:, 3] - self.IoG_gt[:, 1])
        iog = I / G
        smln = smoothln(iog)
        n_p = float(I.shape[0])
        return smln.sum() / n_p
    def IoU(self,box):
        inter_xmin = np.maximum(box[:, 0], self.IoU_const[:, 0])
        inter_ymin = np.maximum(box[:, 1], self.IoU_const[:, 1])
        inter_xmax = np.minimum(box[:, 2], self.IoU_const[:, 2])
        inter_ymax = np.minimum(box[:, 3], self.IoU_const[:, 3])
        Iw = np.clip(inter_xmax - inter_xmin, 0, 5)
        Ih = np.clip(inter_ymax - inter_ymin, 0, 5)
        I = Iw * Ih
        A1 = (self.box[:, 2] - self.box[:, 0]) * (self.box[:, 3] - self.box[:, 1])
        A2 = (self.IoU_const[:, 2] - self.IoU_const[:, 0]) * (self.IoU_const[:, 3] - self.IoU_const[:, 1])
        iou = I / (A1 + A2 - I)
        smln = smoothln(iog)
        n_p = float(I.shape[0])
        return smln.sum()

    def smoothln(self,x):
        x[(x > 0) & (x <= self.sigma)] = -np.log(x[(x > 0) & (x <= self.sigma)])
        x[(x > 0) & (x > self.sigma)] = (x[(x > 0) & (x > self.sigma)] - self.sigma) / (1.0 - sigma) - np.log(1 - self.sigma)
        return x
    def decodebbox(self):
        num_priors = self.priorboxes.shape[2]
        prior_width = (self.priorboxes[0,0,:,2] - self.priorboxes[0,0,:,0]).reshape((1,num_priors))
        prior_height = (self.priorboxes[0,0,:,3] - self.priorboxes[0,0,:,1]).reshape((1,num_priors))
        prior_center_x = ((self.priorboxes[0,0,:,2]  + self.priorboxes[0,0,:,0])/2.0).reshape((1,num_priors))
        prior_center_y = ((self.priorboxes[0, 0, :, 3] + self.priorboxes[0, 0, :, 1]) / 2.0).reshape((1,num_priors))

        decodebox_center_x = self.loc_preds[:,:,0]*prior_width*self.prior_variances[0] + prior_center_x
        decodebox_center_y = self.loc_preds[:,:,1]*prior_height*self.prior_variances[1] + prior_center_y
        decodebox_width = np.exp(self.loc_preds[:,:,2]*self.prior_variances[2])*prior_width
        decodebox_height = np.exp(self.loc_preds[:,:,3]*self.prior_variances[3])*prior_height

        decodebox = np.zeros(self.loc_preds.shape)
        decodebox[:,:,0] = decodebox_center_x - decodebox_width/2.0
        decodebox[:, :, 1] = decodebox_center_y - decodebox_height / 2.0
        decodebox[:, :, 2] = decodebox_center_x + decodebox_width / 2.0
        decodebox[:, :, 3] = decodebox_center_y + decodebox_height / 2.0
        return decodebox




