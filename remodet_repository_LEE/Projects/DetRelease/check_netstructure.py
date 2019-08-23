from __future__ import print_function
from DetRelease_Data import *
sys.path.append('../')
from PyLib.NetLib.VggNet import VGG16_BaseNet_ChangeChannel
from DetNet_Param import *
from AddC6 import *
from DetRelease_Net import *
#
# net = caffe.NetSpec()
# net = get_DAPDataLayer(net, train=True, batchsize=batch_size,data_name = "data",label_name = "label",flag_169=True)
# net = get_DAPDataLayer(net, train=True, batchsize=batch_size, data_name="data_288x512", label_name="label_288x512", flag_169=False)
# net = get_poseDataLayer(net, train=True, batch_size=batch_size,data_name="data_pose", label_name="label_pose")
#
# lr_mult = 0.1
# decay_mult = 1.0
# use_bn = False
# channels = ((32,), (64,), (128, 64, 128), (192, 96, 192, 96, 192), (256, 128, 256, 128, 256))
# strides = (True, True, True, False, False)
# kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
# pool_last = (False,False,False,True,True)
# net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
#                                       kernels=kernels,freeze_layers=[], pool_last=pool_last,flag_withparamname=True,add_string='',
#                             use_bn=use_bn,lr_mult=lr_mult,decay_mult=decay_mult,use_global_stats=None)
# net = VGG16_BaseNet_ChangeChannel(net, from_layer="data_288x512", channels=channels, strides=strides,
#                                   kernels=kernels, freeze_layers=[], pool_last=pool_last, flag_withparamname=True,
#                                   add_string='_288x512',use_bn=use_bn, lr_mult=lr_mult, decay_mult=decay_mult, use_global_stats=None)
# # pool_last = (False, False, False, True, False)
# # net = VGG16_BaseNet_ChangeChannel(net, from_layer="data_pose", channels=channels, strides=strides,
# #                                   kernels=kernels, freeze_layers=[], pool_last=pool_last, flag_withparamname=True,
# #                                   add_string='_pose', use_bn=use_bn, lr_mult=lr_mult, decay_mult=decay_mult,
# #                                   use_global_stats=None)
# lr_detnetperson = 1.0
# # Add Conv6
# conv6_output = Conv6_Param.get('conv6_output',[])
# conv6_kernal_size = Conv6_Param.get('conv6_kernal_size',[])
#
# from_layer = "pool5"
# net = addconv6(net, from_layer=from_layer, use_bn=use_bn, conv6_output=conv6_output,
#     conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=lr_detnetperson, decay_mult=1,n_group=1,flag_withparamname=True)
# from_layer = "pool5_288x512"
# net = addconv6(net, from_layer=from_layer, use_bn=use_bn, conv6_output=conv6_output, post_name = "_288x512",
#     conv6_kernal_size=conv6_kernal_size, pre_name="conv6",start_pool=False,lr_mult=lr_detnetperson, decay_mult=1,n_group=1,flag_withparamname=True)
# ###Create featuremap1,featuremap2,featuremap3
# layers = ["conv3_3", "conv4_5"]
# kernels = [3, 3]
# strides = [1, 1]
# out_layer = "featuremap1"
# num_channels = 128
# add_str = ""
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
# layers = ["conv4_5", "conv5_5"]
# kernels = [3, 3]
# strides = [2, 1]
# out_layer = "featuremap2"
# num_channels = 128
# add_str = ""
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
# layers = ["conv5_5", "conv6_5"]
# kernels = [3, 3]
# strides = [2, 1]
# out_layer = "featuremap3"
# num_channels = 128
# add_str = ""
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
#
# layers = ["conv3_3", "conv4_5"]
# kernels = [3, 3]
# strides = [1, 1]
# out_layer = "featuremap1"
# num_channels = 128
# add_str = "_288x512"
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
# layers = ["conv4_5", "conv5_5"]
# kernels = [3, 3]
# strides = [2, 1]
# out_layer = "featuremap2"
# num_channels = 128
# add_str = "_288x512"
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
# layers = ["conv5_5", "conv6_5"]
# kernels = [3, 3]
# strides = [2, 1]
# out_layer = "featuremap3"
# num_channels = 128
# add_str = "_288x512"
# MultiScaleEltLayer(net, layers=layers, kernels=kernels, strides=strides, out_layer=out_layer,
#                    num_channels=num_channels, lr=1.0, decay=1.0, use_bn=use_bn, add_str=add_str,
#                    flag_withparamname=True)
net = DetRelease_SecondPartAllNet(False)
# net = DetRelease_FirstBodyPartPoseNet(False)

if isinstance(net,list):
    for i in xrange(len(net)):
        net_param = net[i].to_proto()
        fh = open('trunck_det%d.prototxt'%i,'w')
        print(net_param, file=fh)
        fh.close()
else:
    net_param = net.to_proto()
    fh = open('trunck_det.prototxt', 'w')
    print(net_param, file=fh)
    fh.close()