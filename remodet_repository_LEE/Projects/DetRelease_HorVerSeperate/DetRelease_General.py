import math
from caffe import params as P
import time
########train_net_id##############################
"""
train_net_id = 0: training first bd and pd
train_net_id = 1: train second pd and minihand
"""
train_net_id = 0
flag_169_global = True#if train_net_id==0: it does not matter where flag_169 is True or False
flag_noperson_body = True
flag_noperson_part = False #if train_net_id=0,default: False; if train_net_id=1, defaut: True.
body_loss_type = "BBoxLoss"
#body_loss_type = "BBoxLossWTIOUCKCOVER"#BBoxLoss;BBoxLossWTIOUCKCOVER
matchtype_anchorgt = P.BBoxLoss.REMOVELARGMARGIN
margin_ratio = 0.25
sigma_angtdist = 0.1
only_w = False
###MergeSingePerson
single_person_size=80
merge_single_person_prob=0.25
###body
bboxloss_overlap_threshold_body = 0.5
bboxloss_neg_overlap_body = 0.3
bboxloss_neg_pos_ratio_body = 3
###part
bboxloss_overlap_threshold_part = 0.5
bboxloss_neg_overlap_part = 0.3
bboxloss_neg_pos_ratio_part = 3
###mini
bboxloss_overlap_threshold_mini = 0.5
bboxloss_neg_overlap_mini = 0.3
bboxloss_neg_pos_ratio_mini = 20
flag_miniresizeddata = True
flag_mininetwithface = False#if true: minihandata has face; if false: minihanndata only has hand

data_str_body = "AICGoogle0817"
#data_str_body = "OnlyAIC"
data_str_part = "AICREMO0827"
data_str_mini = "AICREMO0827"
###Batchsize###

###AnchorType###
AnchorFixed=False

###Dist_Param
dist_prob = 0.3#default0.2
flag_eqhist = False
brightness_delta = 20#20
contrast_lower = 0.5#0.5
contrast_upper = 1.5#1.5
hue_delta = 18
saturation_lower = 0.5
saturation_upper = 1.5
pose_img_w = 368#default :368
pose_img_h = 368#default :368
####Information for Body and Part
if train_net_id == 0:
    lr_conv1_conv5 = 0.1
    lr_conv6_adap = 1.0
    lr_inter_loss = 1.0
    lr_pose = 0.1
    min_scale = [0.75]*4#default [0.75]*4 
    max_scale = [2.0]*4#default [2.0]*4
#    max_scale = [1.5,2.0,2.0,2.0]
    for_body = True
    batch_size = 12#batch_size_per_device
    batch_size_pose = 6#default:12
    bboxloss_loc_weight_body = 3.0#default:4.0
    bboxloss_conf_weight_body = 2.0#default:1.0
    bboxloss_loc_weight_part = 2.0 /4.0#default:2.0
    bboxloss_conf_weight_part = 1.0/ 4.0##default:1.0
    base_bindex = 0
else:
    lr_conv1_conv5 = 0.0
    lr_conv6_adap = 0.0
    lr_inter_loss = 1.0
    lr_pose = 0.1#no use
    min_scale = [0.05]*4 #default [0.05]*4
    max_scale = [1.0,1.0,2.0,2.0] #default [2.0]*4
    for_body = False
    batch_size = 24#batch_size_per_device
    batch_size_pose = 12  # NOUSE
    bboxloss_loc_weight_body = 4.0  # default:4.0
    bboxloss_conf_weight_body = 1.0  # default:1.0
    bboxloss_loc_weight_part = 2.0  # default:2.0
    bboxloss_conf_weight_part = 1.0  ##default:1.0
    if flag_mininetwithface:
        base_bindex = batch_size
    else:
        base_bindex = 0
def get_scale_str(min_scale,max_scale):
    num2str = {0:"Zero",1:"One",2:"Two",3:"Three",4:"Four"}
    s_dict = {}
    for i in range(len(min_scale)):
        k = (min_scale[i],max_scale[i])
        try:
            s_dict[k] += 1
        except:
            s_dict[k] = 1
    s_s = ""
    for key in s_dict.keys():
        s_s +="%s"%num2str[s_dict[key]]
        s_s += str(key[0])
        s_s += "-"
        s_s += str(key[1])
    return s_s

if train_net_id == 0:#train Body and Part
    Project = "DetPose_JointTrain"
    BaseNet = "JtTrPoseB%dI%dx%d_DarkNet20180514"%(batch_size_pose,pose_img_w,pose_img_h)
    if flag_169_global:
        model_pre = "Hor"
    else:
        model_pre = "Ver"
    noperson_str = ""
    if flag_noperson_body:
        noperson_str += "BDNoPer"
    if flag_noperson_part:
        noperson_str += "PDNoPer"
    if body_loss_type == "BBoxLossWTIOUCKCOVER":
        bdloss_str = "BBoxWTIOUCK"
        if matchtype_anchorgt == P.BBoxLoss.REMOVELARGMARGIN:
            bdloss_str += "Marg%s"%margin_ratio
        else:
            bdloss_str += "WTIOU%s" % sigma_angtdist
        if only_w:
	    bdloss_str += "OnlyW"
    else:
        bdloss_str = "BBox"

    Models = "%s_BDPDBBox%s%s_BD%sLWLoc%sConf%sPIOU%sNIOU%sR%d_PDDenseBBoxLoc%sConf%sPosIOU%sNegIOU%sR%d" \
             %(model_pre,noperson_str,data_str_body,bdloss_str,str(bboxloss_loc_weight_body),str(bboxloss_conf_weight_body),
               str(bboxloss_overlap_threshold_body),str(bboxloss_neg_overlap_body),bboxloss_neg_pos_ratio_body,
               str(bboxloss_loc_weight_part),str(bboxloss_conf_weight_part),str(bboxloss_overlap_threshold_part),str(bboxloss_neg_overlap_part),bboxloss_neg_pos_ratio_part)

    Models += "_SPVecSize%sMpprob%s"%(str(single_person_size),str(merge_single_person_prob))

else:#train Part And MininHand
    Project = "DetNet"
    BaseNet = "Det_Release20180906"
    if flag_169_global:
        model_pre = "Hor"
    else:
        model_pre = "Ver"
    noperson_str = ""
    if flag_noperson_part:
        noperson_str += "NoPer"
    if flag_miniresizeddata:
        mini_str = "Resized"
    else:
        mini_str = ""
    if not flag_mininetwithface:
        Models = "%s_PDBBoxS%s%sMinihand%s%s%sDistP%s_PDDensBBoxLoc%sConf%sPIOU%sNIOU%sR%d_MiniBBoxLoc%sConf%sPIOU%sNIOU%sR%d" \
                 % (model_pre, get_scale_str(min_scale,max_scale), data_str_part,mini_str,noperson_str,data_str_mini,str(dist_prob),str(bboxloss_loc_weight_part), str(bboxloss_conf_weight_part),
                    str(bboxloss_overlap_threshold_part), str(bboxloss_neg_overlap_part),bboxloss_neg_pos_ratio_part,
                    str(bboxloss_loc_weight_part), str(bboxloss_conf_weight_part), str(bboxloss_overlap_threshold_mini), str(bboxloss_neg_overlap_mini),bboxloss_neg_pos_ratio_mini)
    else:
        Models = "%s_PDBBoxS%s%sMinihand%s%s%sDistP%s_PDDensBBoxLoc%sConf%sPIOU%sNIOU%sR%d_MiniHandFace" \
                 % (model_pre, get_scale_str(min_scale,max_scale), data_str_part,mini_str,noperson_str,data_str_mini,str(dist_prob),str(bboxloss_loc_weight_part), str(bboxloss_conf_weight_part),
                    str(bboxloss_overlap_threshold_part), str(bboxloss_neg_overlap_part),bboxloss_neg_pos_ratio_part)
Ver = time.strftime("%Y%m%d",time.localtime())
# Pretained_Model = "/home/zhangming/Models/PretainedModels/Release20180906_AllFeaturemap_WithParamName_V0.caffemodel"
Pretained_Model = "/home/zhangming/Models/PretainedModels/Pose_DarkNet2018515_HisiDataWD5e-3_512x288_iter_240000_merge.caffemodel"
print "{}_{}_{}".format(BaseNet,Models,Ver)



