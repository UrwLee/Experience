#!/bin/bash
# 图片和数据的源目录
root_dir=$HOME/data/VOCdevkit/
# txt描述文件的子目录
sub_dir=ImageSets/Main
# 获取当前路径
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 指向trainval和test
for dataset in trainval test
do
  # 目的文件：trainval.txt & test.txt
  dst_file=$bash_dir/$dataset.txt
  # 如果已经存在该文件，则删除
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  # 数据库名称：VOC2007 VOC2012
  for name in VOC2007 VOC2012
  do
    # 注意，VOC2012是没有test的！！！
    # 直接结束
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    # 打印信息...
    echo "Create list for $name $dataset..."
    # 获取描述图像的文本文件
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt
    # 将数据集中的img-txt文件复制到本地
    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file

    # sed: 在线编辑器，一次处理一行内容
    # sed - i : 直接修改读取的文件内容，不用输出到终端
    # s: 替换命令
    # g: 全局有效，替换所有
    # s/[src]/[replace]/g -> 将所有的src字符全部替换为replace字符
    # s/^/[...]/g -> 将所有行的开头都插入...字符串
    # s/$/[...]/g -> 将所有行的尾部加上...字符串
    # \/ -> 转义字符'/'
    # 第一行的解释： 在行首插入$name/JPEGImages/
    # 第二行解释：在行尾插入.jpg
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    # 复制标注文件的路径
    # 仍然使用图像的名称文件
    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    # 在行首插入$name/Annotations/
    sed -i "s/^/$name\/Annotations\//g" $label_file
    # 在尾部插入.xml
    sed -i "s/$/.xml/g" $label_file

    # paste: 将两个不同文件的数据按行合并为一行
    # -d： 指定不同于空格或\t的分隔符
    # 格式： f1 f2 >> dst_file
    paste -d' ' $img_file $label_file >> $dst_file

    # 复制完毕后，删除中间的文件
    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  # 对test数据集：需要输出名称和尺寸信息
  # 使用caffe/build/tools/get_image_size方法
  # root_dir -> 数据集根目录
  # dst_file -> 图像和标注的路径文件
  # 输出文本文件： ./test_name_size.txt
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  #
  if [ $dataset == "trainval" ]
  then
    # trainval.txt.random -> 作为随机排序后的新文件名称
    rand_file=$dst_file.random
    # 调用shuffle模块进行随机排序
    # -e： 执行的一行命令
    # 命令： shuffle(STDIN<>) -> cat $dst_file的输出作为{管道}其输入，shuffle是乱序的执行者
    # shuffle()的输出结果作为{print}标准输入流写入（>）$rand_file
    # 最终的效果是：dst_file被随机排序，并写入rand_file
    # 第一条命令：cat $dst_file，其输出输入：
    # 第二条命令： perl -MList::Util=shuffle -e 'print shuffle(<STDIN>); 其输出写入：
    # > $rand_file
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    # 处理完毕后，重新命名为trainval.txt
    mv $rand_file $dst_file
  fi
done
