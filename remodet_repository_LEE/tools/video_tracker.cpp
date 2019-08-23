#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracker/alov_loader.hpp"
#include "caffe/tracker/vot_loader.hpp"
#include "caffe/tracker/image_loader.hpp"
#include "caffe/tracker/regressor.hpp"
#include "caffe/tracker/tracker_viewer.hpp"
#include "caffe/tracker/tracker_video.hpp"
#include "caffe/tracker/tracker_base.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

using std::ostringstream;
using std::map;
using std::pair;
using std::vector;

/**
 * 该程序是基于GOTURN的跟踪算法演示脚本。
 * 该程序仅能作为demo演示使用。
 */

DEFINE_string(source_type, "video",
    "Optional; use webcam or video to input, default is video_type.");
DEFINE_string(video, "",
    "Using video input: define input video file.");
DEFINE_int32(skip_frames, 0,
    "Using video input: The first frames to skip.");
DEFINE_int32(webcam_id, 0,
    "Using webcam input: define the cam index.");
DEFINE_int32(webcam_width, 1280,
    "Using webcam input: define the input width.");
DEFINE_int32(webcam_height, 720,
    "Using webcam input: define the input height.");
DEFINE_bool(save_videos, false,
    "If saving the output frames to videos.");
DEFINE_string(output_folder, "",
    "Using save_videos: define the save folder.");
DEFINE_string(network_proto, "",
    "Define the network proto of the regressor.");
DEFINE_string(caffe_model, "",
    "Define the caffemodel of the regressor.");
DEFINE_int32(gpu_id, 0,
    "Define the gpu used.");

int main(int argc, char** argv) {
  // print out to stderr
  FLAGS_alsologtostderr = 1;
  // set caffe version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // set usage message
  gflags::SetUsageMessage("Video Tracker demo for video or webcam.\n"
    "usage: video_tracker <args>\n\n"
    "args examples:\n"
    "  --source_type=video (source type: video or webcam)\n"
    "  --video=/.../... (the path of your video file)\n"
    "  --skip_frames=0 (skip frames)\n"
    "  --save_videos=true (save the tracking results in video)\n"
    "  --output_folder=/... (output path for saving)\n"
    "  --network_proto=/.../....prototxt (the path of your network description)\n"
    "  --caffe_model=/.../....caffemodel (the path of your pretrained model)\n"
    "  --gpu_id=0 (the gpu device selected for computation)\n"
    "\n\n"
    "  --webcam_id=0 (if using webcam input, please define the webcam device used)\n"
    "  --webcam_width=1280 (if using webcam input, define the webcam input width)\n"
    "  --webcam_height=720 (if using webcam input, define the webcam input height)");
  // parse args and run or show message
  caffe::GlobalInit(&argc, &argv);
  // normal process
  if (argc == 1) {
    // input source
    if (FLAGS_source_type == "video") {
      CHECK_GT(FLAGS_video.size(), 0) << "Must define the video file for video_type.";
      LOG(INFO) << "Video: " << FLAGS_video;
    } else if (FLAGS_source_type == "webcam") {
      LOG(INFO) << "Webcam: " << FLAGS_webcam_id << ", width: " << FLAGS_webcam_width << ", height: " << FLAGS_webcam_height;
    } else {
      LOG(FATAL) << "Unknown source_type: " << FLAGS_source_type << ", Only support video or webcam.";
    }
    // output saving results
    if (FLAGS_save_videos) {
      CHECK_GT(FLAGS_output_folder.size(), 0) << "Must define output folder when saving results (video output).";
    }
    // regressor
    CHECK_GT(FLAGS_network_proto.size(), 0) << "Need a model definition to initialize the regressor.";
    CHECK_GT(FLAGS_caffe_model.size(), 0) << "Must define the pretrained model for the network.";

    // VideoTrackerParameter definition
    VideoTrackerParameter param;
    const bool use_video_type = (FLAGS_source_type == "video") ? true : false;
    param.set_is_type_video(use_video_type);
    param.set_video_file(FLAGS_video);
    param.set_initial_frame(FLAGS_skip_frames);
    param.set_webcam_width(FLAGS_webcam_width);
    param.set_webcam_height(FLAGS_webcam_height);
    param.set_device_id(FLAGS_webcam_id);
    param.set_save_videos(FLAGS_save_videos);
    param.set_output_folder(FLAGS_output_folder);
    // create Regressor
    Regressor<float> regressor(FLAGS_network_proto,FLAGS_caffe_model,FLAGS_gpu_id);
    // create TrackerBase
    TrackerBase<float> tracker(false);

    // create videotracker
    VideoTracker<float> video_tracker(param,&regressor,&tracker);
    // run tracking method
    video_tracker.Tracking();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/video_tracker");
  }
}
