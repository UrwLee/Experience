from __future__ import print_function
from DetRelease_Data import *
sys.path.append('../')
from PyLib.NetLib.VggNet import VGG16_BaseNet_ChangeChannel
from DetNet_Param import *
from AddC6 import *
from DetRelease_Net import *
# dims = [1, 3, 96, 96]
# net = caffe.NetSpec()
# net["data"] = L.DummyData(shape={'dim': dims})
# use_bn = False
# channels = ((32,), (64,), (128, 64, 128), (128, 96, 128, 96, 128), (256, 128, 256))
# strides = (True, True, True, False, False)
# kernels = ((3,), (3,), (3, 1, 3), (3, 1, 3, 1, 3), (3, 1, 3, 1, 3))
# pool_last = (False,False,False,True,True)
# net = VGG16_BaseNet_ChangeChannel(net, from_layer="data", channels=channels, strides=strides,
#                                           kernels=kernels,freeze_layers=[], pool_last=pool_last,flag_withparamname=True,add_string='',
#                                 use_bn=use_bn,lr_mult=lr_conv1_conv5,decay_mult=1.0,use_global_stats=None)
net = DetRelease_SecondPartAllNetMiniHandFace(False)
# net = DetRelease_FirstBodyPartPoseNet(False)

if isinstance(net,list):
    for i in xrange(len(net)):
        net_param = net[i].to_proto()
        fh = open('trunck_det%d.prototxt'%i,'w')
        print(net_param, file=fh)
        fh.close()
else:
    net_param = net.to_proto()
    # del net_param.layer[0]
    # net_param.input.extend(["data"])
    # net_param.input_shape.extend([caffe_pb2.BlobShape(dim=dims)])
    fh = open('trunck_det.prototxt', 'w')
    print(net_param, file=fh)
    fh.close()