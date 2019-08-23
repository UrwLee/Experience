#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using namespace boost::property_tree;  // NOLINT(build/namespaces)
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
LOG(INFO)<<"raw_input "<<raw_input;
  bool success = proto->ParseFromCodedStream(coded_input);
  LOG(INFO)<<"success "<<success; 

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim,
    const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (min_dim > 0 || max_dim > 0) {
    int num_rows = cv_img_origin.rows;
    int num_cols = cv_img_origin.cols;
    int min_num = std::min(num_rows, num_cols);
    int max_num = std::max(num_rows, num_cols);
    float scale_factor = 1;
    if (min_dim > 0 && min_num < min_dim) {
      scale_factor = static_cast<float>(min_dim) / min_num;
    }
    if (max_dim > 0 && static_cast<int>(scale_factor * max_num) > max_dim) {
      // Make sure the maximum dimension is less than max_dim.
      scale_factor = static_cast<float>(max_dim) / max_num;
    }
    if (scale_factor == 1) {
      cv_img = cv_img_origin;
    } else {
      cv::resize(cv_img_origin, cv_img, cv::Size(0, 0),
                 scale_factor, scale_factor);
    }
  } else if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim) {
  return ReadImageToCVMat(filename, height, width, min_dim, max_dim, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  return ReadImageToCVMat(filename, height, width, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.') + 1;
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim,
                                    is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          !min_dim && !max_dim && matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      EncodeCVMatToDatum(cv_img, encoding, datum);
      datum->set_label(label);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

void GetImageSize(const string& filename, int* height, int* width) {
  cv::Mat cv_img = cv::imread(filename);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return;
  }
  *height = cv_img.rows;
  *width = cv_img.cols;
}

bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnotatedDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annotation_group();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  switch (type) {
    case AnnotatedDatum_AnnotationType_BBOX:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "xml") {
        return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else if (labeltype == "json") {
        return ReadJSONToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum);
      } else if (labeltype == "txt") {
        return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

// Parse VOC/ILSVRC detection annotation.
bool ReadXMLToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_xml(labelfile, pt);

  // Parse annotation.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    ptree pt1 = v1.second;
    if (v1.first == "object") {
      Annotation* anno = NULL;
      bool difficult = false;
      ptree object = v1.second;
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        if (v2.first == "name") {
          string name = pt2.data();
          if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
          }
          int label = name_to_label.find(name)->second;
          bool found_group = false;
          for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
            AnnotationGroup* anno_group =
                anno_datum->mutable_annotation_group(g);
            if (label == anno_group->group_label()) {
              if (anno_group->annotation_size() == 0) {
                instance_id = 0;
              } else {
                instance_id = anno_group->annotation(
                    anno_group->annotation_size() - 1).instance_id() + 1;
              }
              anno = anno_group->add_annotation();
              found_group = true;
            }
          }
          if (!found_group) {
            // If there is no such annotation_group, create a new one.
            AnnotationGroup* anno_group = anno_datum->add_annotation_group();
            anno_group->set_group_label(label);
            anno = anno_group->add_annotation();
            instance_id = 0;
          }
          anno->set_instance_id(instance_id++);
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
        } else if (v2.first == "bndbox") {
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          CHECK_NOTNULL(anno);
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // Store the normalized bounding box.
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_xmin(static_cast<float>(xmin) / width);
          bbox->set_ymin(static_cast<float>(ymin) / height);
          bbox->set_xmax(static_cast<float>(xmax) / width);
          bbox->set_ymax(static_cast<float>(ymax) / height);
          bbox->set_difficult(difficult);
        }
      }
    }
  }
  return true;
}

// Parse MSCOCO detection annotation.
bool ReadJSONToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_json(labelfile, pt);

  // Get image info.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("image.height");
    width = pt.get<int>("image.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";

  // Get annotation info.
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type& v1, pt.get_child("annotation")) {
    Annotation* anno = NULL;
    bool iscrowd = false;
    ptree object = v1.second;
    // Get category_id.
    string name = object.get<string>("category_id");
    if (name_to_label.find(name) == name_to_label.end()) {
      LOG(FATAL) << "Unknown name: " << name;
    }
    int label = name_to_label.find(name)->second;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group =
          anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);

    // Get iscrowd.
    iscrowd = object.get<int>("iscrowd", 0);

    // Get bbox.
    vector<float> bbox_items;
    BOOST_FOREACH(ptree::value_type& v2, object.get_child("bbox")) {
      bbox_items.push_back(v2.second.get_value<float>());
    }
    CHECK_EQ(bbox_items.size(), 4);
    float xmin = bbox_items[0];
    float ymin = bbox_items[1];
    float xmax = bbox_items[0] + bbox_items[2];
    float ymax = bbox_items[1] + bbox_items[3];
    CHECK_NOTNULL(anno);
    LOG_IF(WARNING, xmin > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
        " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
        " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(iscrowd);
  }
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnotatedDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  int label;
  float xmin, ymin, xmax, ymax;
  while (infile >> label >> xmin >> ymin >> xmax >> ymax) {
    Annotation* anno = NULL;
    int instance_id = 0;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group = anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);
    LOG_IF(WARNING, xmin > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
      " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
      " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(false);
  }
  return true;
}

// Only Read Body of person
bool ReadPersonAnnotationFromXml(const string& labelfile, const int img_height,
    const int img_width, const int img_channels, const float boxsize_thre,
    vector<PersonBBox>* item_anno_person) {
  item_anno_person->clear();
  ptree pt;
  read_xml(labelfile, pt);
  // check size
  int width = 0, height = 0, channels = 0;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
    channels = pt.get<int>("annotation.size.channels");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
    channels = img_channels;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  LOG_IF(WARNING, channels != img_channels) << labelfile <<
      " inconsistent image channels.";
  CHECK(width != 0 && height != 0 && channels != 0) << labelfile <<
      " no valid image width/height/channels.";

  // 遍历annotation内部的所有元素:v1
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    ptree pt1 = v1.second;
    // 查找到object:人物对象
    if (v1.first == "object") {
      // 定义一个person对象
      PersonBBox* person = NULL;
      PersonBBox  person_bbox;
      // default值
      string name = "";
      bool difficult = false;

      ptree object = v1.second;
      // 遍历该object内的所有元素:v2
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        // name元素
        if (v2.first == "name") {
          name = pt2.data();
          CHECK_EQ(name.compare("person"), 0) << "Now only person-class is supported." <<
             "Unknown object name.";
          // 定义一个person的对象
          person = &person_bbox;
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
        } else if (v2.first == "mainobj") {
          // mainobj = pt2.data() == "1";
        } else if (v2.first == "truncated") {
          // truncated = pt2.data() == "1";
        } else if (v2.first == "pose") {
          // string pose_name = pt2.data();
        } else if (v2.first == "bndbox") {
          CHECK_NOTNULL(person);
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          // 检查
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // 设置person的body数据
          NormalizedBBox* body = person->mutable_body();
          body->set_xmin(static_cast<float>(xmin) / width);
          body->set_ymin(static_cast<float>(ymin) / height);
          body->set_xmax(static_cast<float>(xmax) / width);
          body->set_ymax(static_cast<float>(ymax) / height);
          body->set_difficult(difficult);
          float bsize = (body->xmax() - body->xmin()) * (body->ymax() - body->ymin());
          body->set_size(bsize);

          int body_label = 1;
          body->set_label(body_label);
        } else if (v2.first == "part" || v2.first == "location") {
          LOG(FATAL) << "Unsupport for part layout task: " << v2.first;
        } else {
          LOG(FATAL) << "Unknown node for object: " << v2.first;
        }
      }//all object-elements end
      // 将该person对象压入vector<PersonBBox>
      if (person != NULL) {
        if (person_bbox.body().size() > boxsize_thre)
          item_anno_person->push_back(person_bbox);
        else
          DLOG(INFO) << "[Note] A person is rejected for too small size.";
      }
    }// object end
  }
  // LOG(INFO) << item_anno_person->size() << " persons are found in this image.";
  return true;
}

bool ReadPersonAnnotationFromXml(const string& labelfile, const int img_height,
    const int img_width, const int img_channels, const std::map<string, int>& part_name_label,
    const std::map<string, int>& pose_name_label, const std::map<string, int>& dir_name_label,
    vector<PersonBBox>* item_anno_person) {
  item_anno_person->clear();
  ptree pt;
  read_xml(labelfile, pt);
  // check size
  int width = 0, height = 0, channels = 0;
  try {
    height = pt.get<int>("annotiation.size.weight");
    width = pt.get<int>("annotiation.size.width");
    channels = pt.get<int>("annotiation.size.channels");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
    channels = img_channels;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  LOG_IF(WARNING, channels != img_channels) << labelfile <<
      " inconsistent image channels.";
  CHECK(width != 0 && height != 0 && channels != 0) << labelfile <<
      " no valid image width/height/channels.";

  int person_id = 0;
  // 遍历annotation内部的所有元素:v1
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotiation")) {
    ptree pt1 = v1.second;
    // 查找到object:人物对象
    if (v1.first == "object") {
      // 定义一个person对象
      PersonBBox* person = NULL;
      PersonBBox  person_bbox;
      // default值
      string name = "";
      bool difficult = false;
      bool truncated = false;
      bool mainobj = false;
      int pose = 0;
      // 部位的计数器
      int body_cnt = 0, head_cnt = 0, torso_cnt = 0;
      int arm_cnt = 0, leg_cnt = 0, hand_cnt = 0, foot_cnt = 0;

      ptree object = v1.second;
      // 遍历该object内的所有元素:v2
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        // name元素
        if (v2.first == "name") {
          name = pt2.data();
          CHECK_EQ(name.compare("person"), 0) << "Now only person-class is supported." <<
             "Unknown object name.";
          // 定义一个person的对象
          person = &person_bbox;
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
        } else if (v2.first == "mainobj") {
          mainobj = pt2.data() == "1";
        } else if (v2.first == "truncated") {
          truncated = pt2.data() == "1";
        } else if (v2.first == "pose") {
          string pose_name = pt2.data();
          if (pose_name_label.find(pose_name) == pose_name_label.end()) {
            LOG(FATAL) << "Unknown pose name: " << pose_name;
          }
          pose = pose_name_label.find(pose_name)->second;
        } else if (v2.first == "bndbox") {
          CHECK_NOTNULL(person);
          body_cnt ++;
          CHECK_LE(body_cnt, 1) << "only one body is allowed in one person-object.";
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          // 检查
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // 设置person的body数据
          NormalizedBBox* body = person->mutable_body();
          body->set_xmin(static_cast<float>(xmin) / width);
          body->set_ymin(static_cast<float>(ymin) / height);
          body->set_xmax(static_cast<float>(xmax) / width);
          body->set_ymax(static_cast<float>(ymax) / height);
          body->set_difficult(difficult);
          if (part_name_label.find("body") == part_name_label.end()) {
            LOG(FATAL) << "body-part not found in part_name_label map.";
          }
          int body_label = part_name_label.find("body")->second;
          body->set_label(body_label);
          body->set_pose(pose);
          body->set_truncated(truncated);
          body->set_main(mainobj);
        } else if (v2.first == "part" || v2.first == "location") {
          string part_name = "";
          int part_label = 0;
          int part_dir = 0;
          bool part_diff = false;
          ptree part = v2.second;
          // 遍历part节点下的所有元素:v3
          BOOST_FOREACH(ptree::value_type &v3, part.get_child("")) {
            ptree pt3 = v3.second;
            if (v3.first == "name") {
              part_name = pt3.data();
              if (part_name_label.find(part_name) == part_name_label.end()) {
                LOG(FATAL) << "Unknown part name: " << part_name;
              }
              part_label = part_name_label.find(part_name)->second;
            } else if (v3.first == "direction") {
              string dir_name = pt3.data();
              if (dir_name_label.find(dir_name) == dir_name_label.end()) {
                LOG(FATAL) << "Unknown dir name: " << dir_name;
              }
              part_dir = dir_name_label.find(dir_name)->second;
            } else if (v3.first == "difficult") {
              part_diff = pt3.data() == "1";
            } else if (v3.first == "bndbox") {
              CHECK_NOTNULL(person);
              CHECK(person->has_body()) << "the body is not valid while some parts are found.";
              NormalizedBBox* part_bbox = NULL;
              if (part_name.compare("head") == 0) {
                head_cnt++;
                CHECK_LE(head_cnt, 1) << "head count should not be more than 1.";
                part_bbox = person->mutable_head();
              }
              else if (part_name.compare("torso") == 0) {
                torso_cnt++;
                CHECK_LE(torso_cnt, 1) << "torso count should not be more than 1.";
                part_bbox = person->mutable_torso();
              }
              else if (part_name.compare("arm") == 0) {
                arm_cnt++;
                CHECK_LE(arm_cnt, 2) << "arm count should not be more than 2.";
                part_bbox = person->add_arm();
              }
              else if (part_name.compare("leg") == 0) {
                leg_cnt++;
                CHECK_LE(leg_cnt, 2) << "leg count should not be more than 2.";
                part_bbox = person->add_leg();
              }
              else if (part_name.compare("hand") == 0) {
                hand_cnt++;
                CHECK_LE(hand_cnt, 2) << "hand count should not be more than 2.";
                part_bbox = person->add_hand();
              }
              else if (part_name.compare("foot") == 0) {
                foot_cnt++;
                CHECK_LE(foot_cnt, 2) << "foot count should not be more than 2.";
                part_bbox = person->add_foot();
              }
              else {
                LOG(FATAL) << "Unknown part name: " << part_name;
              }

              // 获取box
              int xmin = pt3.get("xmin", 0);
              int ymin = pt3.get("ymin", 0);
              int xmax = pt3.get("xmax", 0);
              int ymax = pt3.get("ymax", 0);
              // 检查
              LOG_IF(WARNING, xmin > width) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, ymin > height) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, xmax > width) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, ymax > height) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, xmin < 0) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, ymin < 0) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, xmax < 0) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, ymax < 0) << labelfile <<
                  " bounding box exceeds image boundary.";
              LOG_IF(WARNING, xmin > xmax) << labelfile <<
                  " bounding box irregular.";
              LOG_IF(WARNING, ymin > ymax) << labelfile <<
                  " bounding box irregular.";
              part_bbox->set_xmin(static_cast<float>(xmin) / width);
              part_bbox->set_ymin(static_cast<float>(ymin) / height);
              part_bbox->set_xmax(static_cast<float>(xmax) / width);
              part_bbox->set_ymax(static_cast<float>(ymax) / height);
              part_bbox->set_difficult(part_diff);
              part_bbox->set_label(part_label);
              part_bbox->set_dir(part_dir);
            } else {
              LOG(FATAL) << "Unknown node for part: " << v3.first;
            }
          }
        } else {
          LOG(FATAL) << "Unknown node for object: " << v2.first;
        }
      }//all object-elements end
      // 将该person对象压入vector<PersonBBox>
      if (person != NULL) {
        item_anno_person->push_back(person_bbox);
        person_id++;
        // print part info.
        // just for Debug
        // ##################################################################
        std::cout << "----------------------------------------------------------" << '\n';
        std::cout << "A person is found." << '\n';
        std::cout << "BBox: " << person_bbox.body().xmin() * width
                              << ", " << person_bbox.body().ymin() * height
                              << ", " << person_bbox.body().xmax() * width
                              << ", " << person_bbox.body().ymax() * height << '\n';
        std::cout << "Pose: " << person_bbox.body().pose() << '\n';
        std::cout << "Diff: " << person_bbox.body().difficult() << '\n';
        std::cout << "Main: " << person_bbox.body().main() << '\n';
        std::cout << "Trun: " << person_bbox.body().truncated() << '\n';
        if (person_bbox.has_head()) {
          std::cout << "#############################" << '\n';
          std::cout << "A head is found." << '\n';
          std::cout << "BBox: " << person_bbox.head().xmin() * width
                                << ", " << person_bbox.head().ymin() * height
                                << ", " << person_bbox.head().xmax() * width
                                << ", " << person_bbox.head().ymax() * height << '\n';
          std::cout << "Dir:  " << person_bbox.head().dir() << '\n';
          std::cout << "Diff: " << person_bbox.head().difficult() << '\n';
        }
        if (person_bbox.has_torso()) {
          std::cout << "#############################" << '\n';
          std::cout << "A torso is found." << '\n';
          std::cout << "BBox: " << person_bbox.torso().xmin() * width
                                << ", " << person_bbox.torso().ymin() * height
                                << ", " << person_bbox.torso().xmax() * width
                                << ", " << person_bbox.torso().ymax() * height << '\n';
          std::cout << "Dir:  " << person_bbox.torso().dir() << '\n';
          std::cout << "Diff: " << person_bbox.torso().difficult() << '\n';
        }
        if (person_bbox.arm_size() > 0) {
          std::cout << "#############################" << '\n';
          std::cout << person_bbox.arm_size() << " arm(s) are found." << '\n';
          for (int l = 0; l < person_bbox.arm_size(); ++l) {
            std::cout << "----------------" << '\n';
            std::cout << "arm_id: " << l << '\n';
            std::cout << "BBox: " << person_bbox.arm(l).xmin() * width
                                  << ", " << person_bbox.arm(l).ymin() * height
                                  << ", " << person_bbox.arm(l).xmax() * width
                                  << ", " << person_bbox.arm(l).ymax() * height << '\n';
            std::cout << "Diff: " << person_bbox.arm(l).difficult() << '\n';
          }
        }
        if (person_bbox.leg_size() > 0) {
          std::cout << "#############################" << '\n';
          std::cout << person_bbox.leg_size() << " leg(s) are found." << '\n';
          for (int l = 0; l < person_bbox.leg_size(); ++l) {
            std::cout << "----------------" << '\n';
            std::cout << "leg_id: " << l << '\n';
            std::cout << "BBox: " << person_bbox.leg(l).xmin() * width
                                  << ", " << person_bbox.leg(l).ymin() * height
                                  << ", " << person_bbox.leg(l).xmax() * width
                                  << ", " << person_bbox.leg(l).ymax() * height << '\n';
            std::cout << "Diff: " << person_bbox.leg(l).difficult() << '\n';
          }
        }
        if (person_bbox.hand_size() > 0) {
          std::cout << "#############################" << '\n';
          std::cout << person_bbox.hand_size() << " hand(s) are found." << '\n';
          for (int l = 0; l < person_bbox.hand_size(); ++l) {
            std::cout << "----------------" << '\n';
            std::cout << "hand_id: " << l << '\n';
            std::cout << "BBox: " << person_bbox.hand(l).xmin() * width
                                  << ", " << person_bbox.hand(l).ymin() * height
                                  << ", " << person_bbox.hand(l).xmax() * width
                                  << ", " << person_bbox.hand(l).ymax() * height << '\n';
            std::cout << "Diff: " << person_bbox.hand(l).difficult() << '\n';
          }
        }
        if (person_bbox.foot_size() > 0) {
          std::cout << "#############################" << '\n';
          std::cout << person_bbox.foot_size() << " foot(s) are found." << '\n';
          for (int l = 0; l < person_bbox.foot_size(); ++l) {
            std::cout << "----------------" << '\n';
            std::cout << "foot_id: " << l << '\n';
            std::cout << "BBox: " << person_bbox.foot(l).xmin() * width
                                  << ", " << person_bbox.foot(l).ymin() * height
                                  << ", " << person_bbox.foot(l).xmax() * width
                                  << ", " << person_bbox.foot(l).ymax() * height << '\n';
            std::cout << "Diff: " << person_bbox.foot(l).difficult() << '\n';
          }
        }
        // ##################################################################
      }
    }// object end
  }
  LOG(INFO) << person_id << " persons are found in this image.";
  return true;
}

bool ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map) {
  // cleanup
  map->Clear();

  std::ifstream file(filename.c_str());
  string line;
  // Every line can have [1, 3] number of fields.
  // The delimiter between fields can be one of " :;".
  // The order of the fields are:
  //  name [label] [display_name]
  //  ...
  int field_size = -1;
  int label = 0;
  LabelMapItem* map_item;
  // Add background (none_of_the_above) class.
  if (include_background) {
    map_item = map->add_item();
    map_item->set_name("none_of_the_above");
    map_item->set_label(label++);
    map_item->set_display_name("background");
  }
  while (std::getline(file, line)) {
    vector<string> fields;
    fields.clear();
    boost::split(fields, line, boost::is_any_of(delimiter));
    if (field_size == -1) {
      field_size = fields.size();
    } else {
      CHECK_EQ(field_size, fields.size())
          << "Inconsistent number of fields per line.";
    }
    map_item = map->add_item();
    map_item->set_name(fields[0]);
    switch (field_size) {
      case 1:
        map_item->set_label(label++);
        map_item->set_display_name(fields[0]);
        break;
      case 2:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[0]);
        break;
      case 3:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[2]);
        break;
      default:
        LOG(FATAL) << "The number of fields should be [1, 3].";
        break;
    }
  }
  return true;
}

bool MapNameToLabel(const LabelMap& map, const bool strict_check,
    std::map<string, int>* name_to_label) {
  // cleanup
  name_to_label->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!name_to_label->insert(std::make_pair(name, label)).second) {
        LOG(FATAL) << "There are many duplicates of name: " << name;
        return false;
      }
    } else {
      (*name_to_label)[name] = label;
    }
  }
  return true;
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name) {
  // cleanup
  label_to_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_name->insert(std::make_pair(label, name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_name)[label] = name;
    }
  }
  return true;
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name) {
  // cleanup
  label_to_display_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& display_name = map.item(i).display_name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_display_name->insert(
              std::make_pair(label, display_name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_display_name)[label] = display_name;
    }
  }
  return true;
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
