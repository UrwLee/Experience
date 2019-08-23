# -*- coding: utf-8 -*-
# 参数解析
import argparse
import os
import shutil
import subprocess
import sys

from caffe.proto import caffe_pb2
from google.protobuf import text_format

# 执行该模块，而非导入
# 该模块的执行函数
if __name__ == "__main__":
  # 创建命令行参数解析器，并添加描述信息
  parser = argparse.ArgumentParser(description="Create AnnotatedDatum database")
  # 添加参数root，包含图像和标注的根目录
  parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
  # 添加参数listfile， 包含图像路径和标注信息的路径文件
  parser.add_argument("listfile",
      help="The file which contains image paths and annotation info.")
  # 创建参数outdir，是保存输出数据文件的目录
  parser.add_argument("outdir",
      help="The output directory which stores the database file.")
  # 创建参数exampledir，保存数据文件链接的路径
  parser.add_argument("exampledir",
      help="The directory to store the link of the database files.")
  # 创建--redo参数
  parser.add_argument("--redo", default = False, action = "store_true",
      help="Recreate the database.")
  # 创建--anno-type参数
  parser.add_argument("--anno-type", default = "classification",
      help="The type of annotation {classification, detection}.")
  # 创建参数--label-type，标注类型，默认是xml
  parser.add_argument("--label-type", default = "xml",
      help="The type of label file format for detection {xml, json, txt}.")
  # 添加参数--backend，默认是lmdb
  parser.add_argument("--backend", default = "lmdb",
      help="The backend {lmdb, leveldb} for storing the result")
  # 创建参数--check-size，默认是False
  parser.add_argument("--check-size", default = False, action = "store_true",
      help="Check that all the datum have the same size.")
  # 创建参数--encode-type，默认是无
  parser.add_argument("--encode-type", default = "",
      help="What type should we encode the image as ('png','jpg',...).")
  # 创建参数--encoded，默认False
  parser.add_argument("--encoded", default = False, action = "store_true",
      help="The encoded image will be save in datum.")
  # 创建参数--gray，默认是False，作为灰度图像处理
  parser.add_argument("--gray", default = False, action = "store_true",
      help="Treat images as grayscale ones.")
  # 创建参数--label-map-file，作为LabelMap的描述文件
  parser.add_argument("--label-map-file", default = "",
      help="A file with LabelMap protobuf message.")
  # --min-dmim/--max-dim，resize参数
  parser.add_argument("--min-dim", default = 0, type = int,
      help="Minimum dimension images are resized to.")
  parser.add_argument("--max-dim", default = 0, type = int,
      help="Maximum dimension images are resized to.")
  # 添加参数--resize_height/width
  parser.add_argument("--resize-height", default = 0, type = int,
      help="Height images are resized to.")
  parser.add_argument("--resize-width", default = 0, type = int,
      help="Width images are resized to.")
  # 创建参数--shuffle,随机调整图像的顺序和label
  parser.add_argument("--shuffle", default = False, action = "store_true",
      help="Randomly shuffle the order of images and their labels.")
  # 创建参数--check-label，检查是否有重名的label或名称
  parser.add_argument("--check-label", default = False, action = "store_true",
      help="Check that there is no duplicated name/label.")

  # 获取命令行参数
  args = parser.parse_args()
  # 以下是必须提供的参数
  # root路径
  root_dir = args.root
  # list文件
  list_file = args.listfile
  # 输出目录
  out_dir = args.outdir
  # 样例目录
  example_dir = args.exampledir

  # 以下是可选擦书
  # 是否redo
  redo = args.redo
  # 标注类型，detection
  anno_type = args.anno_type
  # 标签类型，xml
  label_type = args.label_type
  # 数据库后缀，lmdb
  backend = args.backend
  # 检查datum的size，false
  check_size = args.check_size
  # 图像编码类型，jpg
  encode_type = args.encode_type
  # 编码，false
  encoded = args.encoded
  # 图像作为灰度图像处理，false
  gray = args.gray
  # Labelmap文件
  label_map_file = args.label_map_file
  # 最小最大输入尺寸，0
  min_dim = args.min_dim
  max_dim = args.max_dim
  # resize参数，0
  resize_height = args.resize_height
  resize_width = args.resize_width
  # 随机乱序排列false
  shuffle = args.shuffle
  # 检查重命名问题，false
  check_label = args.check_label

  # check if root directory exists
  # 检查根目录是否存在！
  if not os.path.exists(root_dir):
    print "root directory: {} does not exist".format(root_dir)
    sys.exit()
  # add "/" to root directory if needed
  # 为根目录的最后一个字符添加路径分隔符'/'
  if root_dir[-1] != "/":
    root_dir += "/"
  # check if list file exists
  # 检查图像和标注的list路径文件是否存在！
  if not os.path.exists(list_file):
    print "list file: {} does not exist".format(list_file)
    sys.exit()
  # check list file format is correct
  with open(list_file, "r") as lf:
    for line in lf.readlines():
      # 移除行尾部的回车，使用空格将行分割为图像路径和标注路径
      img_file, anno = line.strip("\n").split(" ")
      # 首先检查图像的路径是否存在？
      if not os.path.exists(root_dir + img_file):
        print "image file: {} does not exist".format(root_dir + img_file)
      # 如果标注类型是分类，则检查anno信息是否是数字
      if anno_type == "classification":
        if not anno.isdigit():
          print "annotation: {} is not an integer".format(anno)
      # 如果是检测类型，还需要检查其标注路径是否存在！
      elif anno_type == "detection":
        if not os.path.exists(root_dir + anno):
          print "annofation file: {} does not exist".format(root_dir + anno)
          sys.exit()
      break
  # check if label map file exist
  # 如果是检测类型，还需要判断LabelMap是否存在！
  if anno_type == "detection":
    if not os.path.exists(label_map_file):
      print "label map file: {} does not exist".format(label_map_file)
      sys.exit()
    # 使用caffe_pb2->LabelMap方法创建一个labelmap
    label_map = caffe_pb2.LabelMap()
    # 打开labelmap文件
    lmf = open(label_map_file, "r")
    # 读取文件中的内容，使用text_format方法赋值到Labelmap
    try:
      text_format.Merge(str(lmf.read()), label_map)
    except:
      print "Cannot parse label map file: {}".format(label_map_file)
      sys.exit()
  # 获取输出路径的上一级目录
  out_parent_dir = os.path.dirname(out_dir)
  # 如果该上级路径不存在，则创建之
  if not os.path.exists(out_parent_dir):
    os.makedirs(out_parent_dir)
  # 如果输出路径已存在，且不需要redo
  # 则直接退出，不需要进行了
  if os.path.exists(out_dir) and not redo:
    print "{} already exists and I do not hear redo".format(out_dir)
    sys.exit()
  #  在进行创建之前，务必保证输出路径为空，下面递归删除输出路径中的所有内容
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

  # get caffe root directory
  # os.path.realpath(__file__) -> 当前脚本的完整路径
  # dirname -> 获取其路径名称
  # 连续两次执行，获取caffe的根目录
  caffe_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  # 检测类型，参数的列表
  # 标注转换工具：caffe/build/tools/convert_annoset
  #
  if anno_type == "detection":
    cmd = "{}/build/tools/convert_annoset" \
        " --anno_type={}" \
        " --label_type={}" \
        " --label_map_file={}" \
        " --check_label={}" \
        " --min_dim={}" \
        " --max_dim={}" \
        " --resize_height={}" \
        " --resize_width={}" \
        " --backend={}" \
        " --shuffle={}" \
        " --check_size={}" \
        " --encode_type={}" \
        " --encoded={}" \
        " --gray={}" \
        " {} {} {}" \
        .format(caffe_root, anno_type, label_type, label_map_file, check_label,
            min_dim, max_dim, resize_height, resize_width, backend, shuffle,
            check_size, encode_type, encoded, gray, root_dir, list_file, out_dir)
  elif anno_type == "classification":
    cmd = "{}/build/tools/convert_annoset" \
        " --anno_type={}" \
        " --min_dim={}" \
        " --max_dim={}" \
        " --resize_height={}" \
        " --resize_width={}" \
        " --backend={}" \
        " --shuffle={}" \
        " --check_size={}" \
        " --encode_type={}" \
        " --encoded={}" \
        " --gray={}" \
        " {} {} {}" \
        .format(caffe_root, anno_type, min_dim, max_dim, resize_height,
            resize_width, backend, shuffle, check_size, encode_type, encoded,
            gray, root_dir, list_file, out_dir)
  # 打印命令信息
  print cmd
  # 创建子进程，其输出流通过PIPE（管道）汇聚到一起
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
  # communicate会阻塞父进程，直到其结束
  # 获取子进程的返回值
  output = process.communicate()[0]
  # 链接目录不存在，则创建之
  if not os.path.exists(example_dir):
    os.makedirs(example_dir)
  # 路径拼接
  link_dir = os.path.join(example_dir, os.path.basename(out_dir))
  if os.path.exists(link_dir):
    os.unlink(link_dir)
  # 创建符号连接，使example_dir下的链接文件指向真实的输出文件
  os.symlink(out_dir, link_dir)
