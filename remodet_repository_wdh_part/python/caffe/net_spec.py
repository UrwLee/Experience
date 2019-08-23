# -*- coding: utf-8 -*-
"""Python net specification.

This module provides a way to write nets directly in Python, using a natural,
functional style. See examples/pycaffe/caffenet.py for an example.

Currently this works as a thin wrapper around the Python protobuf interface,
with layers and parameters automatically generated for the "layers" and
"params" pseudo-modules, which are actually objects using __getattr__ magic
to generate protobuf messages.

Note that when using to_proto or Top.to_proto, names of intermediate blobs will
be automatically generated. To explicitly specify blob names, use the NetSpec
class -- assign to its attributes directly to name layers, and call
NetSpec.to_proto to serialize all assigned layers.

This interface is expected to continue to evolve as Caffe gains new capabilities
for specifying nets. In particular, the automatically generated layer names
are not guaranteed to be forward-compatible.
"""

from collections import OrderedDict, Counter

from .proto import caffe_pb2
from google import protobuf
import six

import sys
sys.dont_write_bytecode = True

# 返回参数字典
# 键值对形式：Type: Name
# 例如：Convolution: convolution
def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    # 获取所有的层参数
    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    # 遍历查找以_param结尾的参数描述符：参数名称
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
    # 获得参数的类型名称
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    # 去掉尾部的_param，返回参数名
    param_names = [s[:-len('_param')] for s in param_names]
    # 去掉参数类型的尾部Parameter，然后返回
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    # 返回参数字典：类型：参数名
    return dict(zip(param_type_names, param_names))


def to_proto(*tops):
    """Generate a NetParameter that contains all layers needed to compute
    all arguments."""
    # 创建一个排序字典
    layers = OrderedDict()
    # Count类的目的是跟踪某个元素出现的次数
    # 是字典的子类，以键值对方式存储
    # Key：元素，Value：该元素出现的次数
    # Counter类可以从字典中或可迭代对象或键值对列表中进行创建
    # 也可以创建一个空的Counter对象
    autonames = Counter()
    # 遍历tops列表中的对象
    for top in tops:
        # 调用其创建者的_to_proto方法，将layer写入到layers中
        # name列表为空，意味着所有层名/blobs名由autonames依据类型自动创建
        top.fn._to_proto(layers, {}, autonames)
    # 创建一个网络
    net = caffe_pb2.NetParameter()
    # 为该网络添加定义的层
    net.layer.extend(layers.values())
    return net


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""
    # print proto,name
    is_repeated_field = hasattr(getattr(proto, name), 'extend')
    if is_repeated_field and not isinstance(val, list):
        val = [val]
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in six.iteritems(item):
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in six.iteritems(val):
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)

# 输出的top类对象
# 通过L.type(params)方法返回的就是该类
class Top(object):
    """A Top specifies a single output blob (which could be one of several
    produced by a layer.)"""
    # fn -> 函数体，可以理解为层的对象，它往往由Function创建，并将self传递进来
    # 因此，可以认为其fn代表的是创建该Top的Function本身
    # n：该Layer（fn）的第几个输出Blob
    def __init__(self, fn, n):
        self.fn = fn
        self.n = n
    # 实际上也是调用其创建者的_to_proto方法
    def to_proto(self):
        """Generate a NetParameter that contains all layers needed to compute
        this top."""

        return to_proto(self)
    # 调用其创建者的update方法
    def _update(self, params):
        self.fn._update(params)
    # 调用其创建者的_to_proto方法
    def _to_proto(self, layers, names, autonames):
        return self.fn._to_proto(layers, names, autonames)

# layer的定义函数
class Function(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, type_name, inputs, params):
        # 层类型
        self.type_name = type_name
        # 层输入，*args
        self.inputs = inputs
        # 层参数，字典类型，**kwargs
        self.params = params
        # 解析参数，将参数中的ntop/in-place读取后删除
        # 输出Blob数，默认为1
        self.ntop = self.params.get('ntop', 1)
        # use del to make sure kwargs are not double-processed as layer params
        # 将参数字典中的'ntop'变量删除
        if 'ntop' in self.params:
            del self.params['ntop']
        # in-place参数，默认为false
        self.in_place = self.params.get('in_place', False)
        # 删除参数中的in-place参数
        if 'in_place' in self.params:
            del self.params['in_place']
        # 创建top-blobs元组对象
        self.tops = tuple(Top(self, n) for n in range(self.ntop))

    def _get_name(self, names, autonames):
        if self.params.has_key('name'):
            names[self] = self.params['name']
        elif self not in names and self.ntop > 0:
            names[self] = self._get_top_name(self.tops[0], names, autonames)
        elif self not in names:
            autonames[self.type_name] += 1
            names[self] = self.type_name + str(autonames[self.type_name])
        return names[self]

    # 获取top名称
    def _get_top_name(self, top, names, autonames):
        if top not in names:
            autonames[top.fn.type_name] += 1
            names[top] = top.fn.type_name + str(autonames[top.fn.type_name])
        return names[top]

    # 更新该层的参数
    def _update(self, params):
        self.params.update(params)

    # 将该层添加到输出网络描述结构内
    # 创建该层，将其加入到layers列表中，layers是一个有序字典
    # names：
    def _to_proto(self, layers, names, autonames):
        # layers是排序字典
        # names是名称列表,来自于net中的top-blobs字典{v:k}
        # autonames是一个Counter类
        # 如果该层已经在layers中出现，则pass
        if self in layers:
            return
        # 开始添加该层的信息
        # bottom初始化为空
        bottom_names = []
        # 遍历其输入blobs对象
        for inp in self.inputs:
            # 对其输入也添加,防止遗漏
            inp._to_proto(layers, names, autonames)
            # 输入列表添加
            # layers[inp.fn]：层
            # top[inp.n]： 第n个Top-blob
            bottom_names.append(layers[inp.fn].top[inp.n])
        # 创建层
        layer = caffe_pb2.LayerParameter()
        # 定义层类型
        layer.type = self.type_name
        # 加入输入Blobs
        layer.bottom.extend(bottom_names)
        # 如果是in_place
        # top与bottom相同
        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
        # 遍历其tops列表
        # 添加到layer.top之中
            for top in self.tops:
                layer.top.append(self._get_top_name(top, names, autonames))
        # 默认层名是top[0]的名称
        layer.name = self._get_name(names, autonames)

        # 开始写入参数
        # 字典，参数迭代
        for k, v in six.iteritems(self.params):
            # special case to handle generic *params
            # _param结束，则为参数类型
            # 通过assign_proto描述
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer,
                        _param_names[self.type_name] + '_param'), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        layers[self] = layer

    # 获取层类型
    def get_type(self):
        return self.type_name

    # 获取输入blobs数
    def get_bottom_blobs_num(self):
        return len(self.inputs)

    # 获取参数
    def get_params(self):
        return self.params

    # 获取输出blobs数
    def get_top_blobs_num(self):
        return self.ntop

    # 是否是in_place类型
    def is_in_place(self):
        return self.in_place


class NetSpec(object):
    """A NetSpec contains a set of Tops (assigned directly as attributes).
    Calling NetSpec.to_proto generates a NetParameter containing all of the
    layers needed to produce all of the assigned Tops, using the assigned
    names."""
    # 初始化阶段，为tops分配一个空的有序字典
    def __init__(self):
        super(NetSpec, self).__setattr__('tops', OrderedDict())

    # 设置tops[name] = value
    # 为网络添加一个top
    def __setattr__(self, name, value):
        self.tops[name] = value

    # get
    def __getattr__(self, name):
        return self.tops[name]

    # set
    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    # get
    def __getitem__(self, item):
        return self.__getattr__(item)

    # del
    def __delitem__(self, name):
        del self.tops[name]

    # 调用keys方法获取tops的名称
    def keys(self):
        keys = [k for k, v in six.iteritems(self.tops)]
        return keys
    # vals方法获取tops的值
    # top的值其实就是Top类，包含其创建者（fn，layer）和输出编号
    def vals(self):
        vals = [v for k, v in six.iteritems(self.tops)]
        return vals
    # 将tops[name]的参数更新为params
    # 实际上，是调用该top-blob所属的layer进行更新
    def update(self, name, params):
        self.tops[name]._update(params)

    # 将网络参数定义为layers/names/autonames构建的结构
    def to_proto(self):
        # names是 {v:k}构建的字典
        names = {v: k for k, v in six.iteritems(self.tops)}
        # Counter类
        autonames = Counter()
        # 排序字典
        # layer_name: layer ->字典结构
        layers = OrderedDict()
        # 遍历tops，每个top是一个键值对构成：name : top
        for name, top in six.iteritems(self.tops):
            # 调用top的_to_proto方法完成本layer的更新过程
            top._to_proto(layers, names, autonames)
        # 定义一个网络
        net = caffe_pb2.NetParameter()
        # 将所有的层添加，结束
        # layers.values()为其所有的layer
        # layers.keys()为所得层名
        net.layer.extend(layers.values())
        return net

    # 根据网络设置名称:layers/blobs
    def _naming_top_blobs(self):
        # 获取top-blobs的名称和对象
        names = {v: k for k, v in six.iteritems(self.tops)}
        # 自定义名称
        autonames = Counter()
        # 网络的layers列表,[名称和layer对象]
        layers = OrderedDict()
        # 遍历所有的top-blobs,创建layer
        for name, top in six.iteritems(self.tops):
            top._to_proto(layers, names, autonames)
        # 返回layers,names,autonames
        return layers, names, autonames

    # 获取网络的所有layers,对象
    def _get_layersfn(self):
        layers, names, autonames = self._naming_top_blobs()
        return layers.keys()

    # 获取网络的所有layers
    def _get_layers(self):
        layers, names, autonames = self._naming_top_blobs()
        return layers.values()

    # 获取网络的层数
    def get_layers_size(self):
        layers, names, autonames = self._naming_top_blobs()
        return len(layers.keys())

    # 获取网络的所有blobs
    def _get_blobs(self):
        return self.tops

    # 获取网络的所有blobs的名称
    def get_blobs_name(self):
        return self.tops.keys()

    # 获取网络第i层layer的参数
    def _get_layer_by_index(self, index):
        if (index < self.get_layers_size()):
            return self._get_layers()[index]
        else:
            raise IndexError("the layer index exceeds the net layers size.")

    # 获取网络第i层layer的对象
    def _get_layerfn_by_index(self, index):
        if (index < self.get_layers_size()):
            return self._get_layersfn()[index]
        else:
            raise IndexError("the layer index exceeds the net layers size.")

    # 获取层名
    def get_layer_name(self, index):
        return self._get_layer_by_index(index).name

    # 获取层类型
    def get_layer_type(self, index):
        return self._get_layer_by_index(index).type

    # 获取层的输入blobs
    def get_layer_bottom(self, index):
        bottom = self._get_layer_by_index(index).bottom
        cstr = ""
        for p in bottom:
            cstr = cstr + p + ", "
        cstr = cstr[:-2]
        return cstr

    # 获取层的输出blobs
    def get_layer_top(self, index):
        top = self._get_layer_by_index(index).top
        cstr = ""
        for p in top:
            cstr = cstr + p + ", "
        cstr = cstr[:-2]
        return cstr

    # 获取层的参数列表
    def get_layer_params(self, index):
        return self._get_layerfn_by_index(index).params

    # 打印网络基本信息
    def print_net_info(self):
        print("################################################################")
        print("Network Layer Size: {}".format(self.get_layers_size()))
        print("################################################################")
        for i in range(self.get_layers_size()):
            print("Layer ID     : {}".format(i))
            print("Layer Type   : {}".format(self.get_layer_type(i)))
            print("Layer Name   : {}".format(self.get_layer_name(i)))
            print("Layer Bottom : {}".format(self.get_layer_bottom(i)))
            print("Layer Top    : {}".format(self.get_layer_top(i)))
            print("Layer Params : {}".format(self.get_layer_params(i)))
            print("----------------------------------------------------------------")

class Layers(object):
    """A Layers object is a pseudo-module which generates functions that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""
    # 这是一个伪模块的类，只定义了一个属性，name
    # 其属性直接返回一个构建层的函数句柄fn
    # 层的实际构造函数为这个函数句柄，它接收args[作为输入列表]，**kwargs作为参数字典
    # 构造函数的返回就是层的tops元组
    # 例如：L.layertype(input_blob0,input_blob1,...,arg0=val0,...,dict_list)
    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            if fn.ntop == 0:
                return fn
            elif fn.ntop == 1:
                return fn.tops[0]
            else:
                return fn.tops
        return layer_fn

class Parameters(object):
    """A Parameters object is a pseudo-module which generates constants used
    in layer parameters; e.g., Parameters().Pooling.MAX is the value used
    to specify max pooling."""
    # 参数结构也只是定义了一个属性：name，作为参数的类型
    # 该属性又定义了一个属性：param_name
    # 以P.Pooling.MAX为例
    # P.Pooling返回一个Param()类
    # 在caffe_pb2中查找属性PoolingParameter，然后进一步在PoolingParameter中
    # 查找后面定义的MAX属性，并将其结果返回
    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()


_param_names = param_name_dict()
layers = Layers()
params = Parameters()
