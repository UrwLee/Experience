#ifndef CAFFE_REMO_FRONT_VISUALIZER_H
#define CAFFE_REMO_FRONT_VISUALIZER_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/caffe.hpp"
#include "caffe/remo/frame_reader.hpp"
#include "caffe/remo/net_wrap.hpp"
#include "caffe/remo/basic.hpp"
#include "caffe/remo/data_frame.hpp"

namespace caffe {

/**
 * 该类是一个工程类，提供了从视频流构建，到输出可视化的所有过程。
 * 该类内部集成了一个数据读取类FrameReader以及一个网络封装类NetWrapper
 * 流程如下：
 * １．FrameReader不断获取新的数据帧
 * ２．NetWrapper不断处理新的帧，然后进行可视化
 */

template <typename Dtype>

class RemoFrontVisualizer {
public:
  RemoFrontVisualizer(
       int cam_id, int width, int height,       // WEBCAM
       int resized_width, int resized_height,   // Network Size
       const std::string& network_proto,        // Net & Model
       const std::string& caffe_model,
       const bool mode, const int gpu_id,       // GPU ID
       const std::string& proposals,            // Feature Name (proposals & heatmaps)
       const std::string& heatmaps,
       const int max_dis_size,                  // Maximun Display Size (e.g., 1000)
       const DrawnMode drawn_mode);             // visualze mode
  /**
  * @Params
  * <Remo>
  *      1. network_proto,  网络描述文件
  *      2. caffe_model,    权值文件
  *      3. gpu_id,         GPU编号
  *      4. proposals,      网络输出的proposal特征blob名
  *      5. heatmaps,       网络输出的heat/vec-map特征blob名
  *      6. max_dis_size,   输出可视化的最大尺寸　【最大长或宽】
  *    7_1. cam_id / width / int 摄像头：定义ID/输入长宽
  *    7_2. video_file, start_frame　视频文件：定义源文件和初始帧编号
  *      8. resized_width / resized_height, 网络的图像输入尺寸
  **/
  /**
   * 构造方法：网络摄像头
   */
  RemoFrontVisualizer(
       int cam_id, int width, int height,       // WEBCAM
       int resized_width, int resized_height,   // Network Size
       const std::string& network_proto,        // Net & Model
       const std::string& caffe_model,
       const int gpu_id,                        // GPU ID
       const std::string& proposals,            // Feature Name (proposals & heatmaps)
       const std::string& heatmaps,
       const int max_dis_size,                  // Maximun Display Size (e.g., 1000)
       const DrawnMode drawn_mode);             // visualze mode

  RemoFrontVisualizer(
      const std::string& video_file,           // Video File
      int start_frame,                         // Frames Skip
      int resized_width, int resized_height,   // Network Size
      const std::string& network_proto,        // Net & Model
      const std::string& caffe_model,
      const bool mode, const int gpu_id,       // GPU ID
      const std::string& proposals,            // Feature Name (proposals & heatmaps)
      const std::string& heatmaps,
      const int max_dis_size,                  // Maximun Display Size (e.g., 1000)
      const DrawnMode drawn_mode);             // visualze mode

  /**
   * 构造方法：视频文件
   */
  RemoFrontVisualizer(
       const std::string& video_file,           // Video File
       int start_frame,                         // Frames Skip
       int resized_width, int resized_height,   // Network Size
       const std::string& network_proto,        // Net & Model
       const std::string& caffe_model,
       const int gpu_id,                        // GPU ID
       const std::string& proposals,            // Feature Name (proposals & heatmaps)
       const std::string& heatmaps,
       const int max_dis_size,                  // Maximun Display Size (e.g., 1000)
       const DrawnMode drawn_mode);             // visualze mode

   /**
    * 构造方法：RTSP数据流
    */
   RemoFrontVisualizer(
        const std::string& ip_addr,              // stream-server address
        int resized_width, int resized_height,   // Network Size
        const std::string& network_proto,        // Net & Model
        const std::string& caffe_model,
        const int gpu_id,                        // GPU ID
        const std::string& proposals,            // Feature Name (proposals & heatmaps)
        const std::string& heatmaps,
        const int max_dis_size,                  // Maximun Display Size (e.g., 1000)
        const DrawnMode drawn_mode);             // visualze mode

  /**
   * 网络摄像头的默认构造方法
   *    CamId = 0, size: 1280x720
   *    Netsize: 512x288
   *    use GPU: 0
   *    proposals & heatmaps -> "proposals" & "resized_map"
   *    Maximun display size -> 1000
   *    DrawnMode: SKELETON_BOX
   **/
  RemoFrontVisualizer(
       const std::string& network_proto,
       const std::string& caffe_model);

   /**
    * 视频文件的默认构造方法
    *    Frame skip: 0
    *    Netsize: 512x288
    *    use GPU: 0
    *    proposals & heatmaps -> "proposals" & "resized_map"
    *    Maximun display size -> 1000
    *    DrawnMode: SKELETON_BOX
    **/
  RemoFrontVisualizer(
       const std::string& video_file,
       const std::string& network_proto,
       const std::string& caffe_model);

  /**
   * 单次运行：　数据帧读取一次，网络计算一次，可视化一次
   * @param  meta [该帧计算返回的目标对象数据结构]
   * @return      [0-数据流读取正常，1-数据流读取结束]
   */
  int step(std::vector<std::vector<Dtype> >* meta);

  /**
   * 单次运行：　数据帧读取一次，网络计算一次
   * @param  image [该帧计算返回的可视化图像]
   * @param  meta  [该帧计算返回的目标对象数据结构]
   * @return       [0-数据流读取正常，1-数据流读取结束]
   */
  int step(cv::Mat* image, std::vector<std::vector<Dtype> >* meta);

  /**
   * 保存操作：输入数据流每interval帧处理的原始图像和输出可视化图像按照提供的指定路径进行保存
   * @param  interval    [保存间隔帧数]
   * @param  save_orig   [是否保存原始图像]
   * @param  orig_dir    [原始图像保存路径]
   * @param  process_dir [输出可视化图像保存路径]
   * @return             [0-数据流读取正常，1-数据流读取结束]
   */
  int save(const int interval, const bool save_orig, const std::string& orig_dir, const std::string& process_dir);

  /**
   * 工程运行的时间统计信息，单位：us
   * @param frame_loader [帧读取的时间]
   * @param preprocess   [图像预处理的时间]
   * @param forward      [前向计算时间]
   * @param drawn        [输出图像绘制时间]
   * @param display      [OPENCV显示时间]
   */
  void time_us(Dtype* frame_loader, Dtype* preprocess, Dtype* forward, Dtype* drawn, Dtype* display);

  /**
   * 工程运行的速度FPS信息，单位：fps
   * @param net_fps     [网络前向计算的FPS]
   * @param project_fps [工程运行的平均FPS]
   */
  void fps(Dtype* net_fps, Dtype* project_fps);

  /**
   * 获取工程已经处理的帧数
   * @return [已处理的帧数]
   */
  int get_frames() { return frames_; }

  // 网络
  boost::shared_ptr<NetWrapper<Dtype> > net_wrapper_;
  // 帧读取
  boost::shared_ptr<FrameReader<Dtype> > frame_reader_;

protected:
  // 绘制模式，在basic.hpp中定义
  DrawnMode drawn_mode_;
  // 帧读取所花的时间统计信息：平均值和累计值
  Dtype frame_loader_time_;
  Dtype frame_loader_time_sum_;
  // 图像显示所花的时间信息统计：平均值和累计值
  Dtype display_time_;
  Dtype display_time_sum_;
  // 已处理帧数累计
  int frames_;
};

}

#endif
