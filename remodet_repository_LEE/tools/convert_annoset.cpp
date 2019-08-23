// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
// gflags，本文用于命令行参数解析
#include "gflags/gflags.h"
// 开源日志库
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

/**
 * 设置命令行选项：
 * 名称-默认值-帮助说明
 */
// gray，是否作为灰度图片处理
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
// shuffle，是否随机乱序排列
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
// 后缀，LMDB
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
// 标注类型：detection
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection}.");
// 标注文件的后缀xml
DEFINE_string(label_type, "xml",
    "The type of annotation file format.");
// labelmap文件名
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
// 是否检查label重名
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
// resize的极限尺寸，默认0，不处理，保持长宽比
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
// resize尺寸，直接resize到制定尺寸
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
// 检查尺寸
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
// 是否在datum对图像进行编码，默认False
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
// 编码方式，默认jpg，但一般不编码！
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

// 应用程序
int main(int argc, char** argv) {
#ifdef USE_OPENCV

/**
 * glog的日志系统分为四级：INFO/WARNING/ERROR/FATAL
 * 打印log的语句与c++中的io流对象类似
 * LOG(INFO)宏返回的是一个继承自std::ostrstream类的对象
 * 每个级别的日志会输出到不同的文件中。并且高级别日志文件会同样输入到低级别的日志文件中。
 * 即：FATAL的信息会同时记录在INFO，WARNING，ERROR，FATAL日志文件中。
 * 默认情况下，glog还会将会将FATAL的日志发送到stderr中。
 *
 *
 */
  // 日志初始化，定义日志文件的名称
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  // LOG(INFO)也作为stderr输出信息
  // 不添加此句，LOG(INFO) 不起作用
  FLAGS_alsologtostderr = 1;

// 版本兼容问题，务必加上
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // 设置本程序的帮助信息
  // 还可以通过google::SetVersionString来设置版本信息
  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  // 解析命令行参数
  // 最后一个参数True -> 函数解析完成后，argv中只保留argv[0]，argc会被设置为1
  // False -> argv和argc会被保留，但是函数会调整argv中的顺序
  // 如此，FLAGS_变量名可以完成对应参数的访问
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //检查参数数量
  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    return 1;
  }

  // 获取参数信息
  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  const string label_type = FLAGS_label_type;
  const string label_map_file = FLAGS_label_map_file;
  const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;

  // 打开ListFile输入文件流
  std::ifstream infile(argv[2]);
  // boost::variant支持多种类型的声明
  // 对于classification，其类型为int，表示其分类号
  // 对于detection，其类型为string，表示xml文件的路径
  // 每行代表一个样本
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
  // 图像文件路径
  std::string filename;
  // 类号
  int label;
  // 标注文件路径
  std::string labelname;
  // 如果是分类
  if (anno_type == "classification") {
    // 每行依次读取图像路径和标注分类
    while (infile >> filename >> label) {
      // 将其压入line容器中
      lines.push_back(std::make_pair(filename, label));
    }
  } else if (anno_type == "detection") {
    // 如果是检测任务
    // 类型为BBOX，目前只有这一种标注类型
    type = AnnotatedDatum_AnnotationType_BBOX;
    // labelMap
    LabelMap label_map;
    // 从labelMap文件中读入
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    // 从LabelMap中映射到name_to_label键值对中【map】
    // check_label：检查label是否有重名
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    // 从文件中读取图像文件路径/标注文件路径，压入lines
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }
  // 上面已经获取了lines容器对象，每一个对象代表一个文件及其标注信息
  // 是否乱序排泄
  if (FLAGS_shuffle) {
    // randomly shuffle data
    // 写入日志
    LOG(INFO) << "Shuffling data";
    // 使用shuffle函数对lines【vector】对象进行乱序
    shuffle(lines.begin(), lines.end());
  }
  // 写入日志
  LOG(INFO) << "A total of " << lines.size() << " images.";

  // 不处理
  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // 创建一个新的db对象
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  // 打开，名称由argv[3]确定
  // [1] -> root dir
  // [2] -> listfile
  // [3] -> outputfile
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // 获取图像文件根目录
  std::string root_folder(argv[1]);
  // 定义AnnotatedDatum进行转换
  AnnotatedDatum anno_datum;
  // 获取datum元素的数据指针
  Datum* datum = anno_datum.mutable_datum();
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  // 遍历lines中每个对象
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status = true;
    std::string enc = encode_type;
    // 获取文件的扩展名，全部改为小写,.JPG -> .jpg
    if (encoded && !enc.size()) {
      // 获取图像路径
      string fn = lines[line_id].first;
      // 查找扩展名
      size_t p = fn.rfind('.');
      // 如果未找到，则显示警告
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      // 讲扩展名全部改为小写
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    // 文件路径
    filename = root_folder + lines[line_id].first;
    // 分类
    if (anno_type == "classification") {
      // 获取类别
      label = boost::get<int>(lines[line_id].second);
      // 使用ReadImageToDatum
      /**
       * filename:图像文件路径
       * label:标签
       * resize_height:resize尺寸
       * resize_width:
       * min_dim:极限尺寸
       * max_dim:
       * is_color: 是否使用彩色
       * enc:编码方式
       * datum:输出datum数据指针
       */
      status = ReadImageToDatum(filename, label, resize_height, resize_width,
          min_dim, max_dim, is_color, enc, datum);
    } else if (anno_type == "detection") {
      // 获取标注文件路径
      labelname = root_folder + boost::get<std::string>(lines[line_id].second);
      //labelname： 标注文件路径
      //type： BBOX
      //label_type：xml
      //name_to_label：类名映射map
      //anno_datum：AnnotatedDatum类型指针
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, type, label_type,
          name_to_label, &anno_datum);
      // 目前只支持这一种类型：BOX
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
    // 如果status变为False，则出现错误：
    if (status == false) {
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }

    // 检查尺寸
    if (check_size) {
      // 获取第一张图片的尺寸
      if (!data_size_initialized) {
        data_size = datum->channels() * datum->height() * datum->width();
        data_size_initialized = true;
      } else {
        // 检查后续的图片尺寸是否相同，否则报错！
        const std::string& data = datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    // caffe::format_int(line_id, 8)：8位整数号码代表其在db中的位置
    // _【图像路径】
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // 将序列化结果输出字节流out中
    string out;
    // 序列化
    CHECK(anno_datum.SerializeToString(&out));
    // 名称：完成其键值设定：str
    txn->Put(key_str, out);

    // 提交，每1000个datum提交一次
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }

  // write the last batch
  // 最后剩下的部分提交一次
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  // 如果不适用OPENCV，报警！
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
