import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def tdm(net,current_from_layer,high_from_layer,featuremap_num,freeze = False):

    if freeze:
        kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)
        }
        de_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        }
    else:
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)
        }
        de_kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        }

        # net[high_from_layer+'_lateral'] = L.Convolution(net[high_from_layer], num_output=128,
        #                                kernel_size=1, pad=0, stride=1, group=1, **kwargs)

        net[high_from_layer+'_Deconv'] = L.Deconvolution(net[high_from_layer],
                                        convolution_param = dict(weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0),num_output=128,
                                        kernel_size=2, pad=0, stride=2), **de_kwargs)

        net[current_from_layer+'_addtop'] = L.Convolution(net[current_from_layer], num_output=128,
                                       kernel_size=1, pad=0, stride=1, group=1, **kwargs)
        # net[current_from_layer+'_addtop' + '_relu'] = L.ReLU(net[current_from_layer+'_addtop'], in_place=True)

        net['featuremap'+str(featuremap_num)] = L.Eltwise(net[current_from_layer+'_addtop'], net[high_from_layer+'_Deconv'])

        net['featuremap'+str(featuremap_num)+'_relu'] = L.ReLU(net['featuremap'+str(featuremap_num)], in_place=True)
    
        return net,'featuremap'+str(featuremap_num)

def single_conv_relu(net,from_layer,prefix,num_output,kernel_size,pad,stride,freeze = False):
    
    if freeze:
        kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)
        }
        de_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        }
    else:
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)
        }
        de_kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        }

    net[from_layer+prefix] = L.Convolution(net[from_layer], num_output=num_output,
                                   kernel_size=kernel_size, pad=pad, stride=stride, group=1, **kwargs)
    net[from_layer+prefix+'_relu'] = L.ReLU(net[from_layer+prefix], in_place=True)

    return net