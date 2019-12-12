/**
    g++ $(pkg-config --cflags --libs
/usr/local/Cellar/opencv/4.1.1_2/lib/pkgconfig/opencv4.pc) -std=c++11  local.cpp
-o local
**/
#include <array>
#include <iomanip>  // for controlling float print precision
#include <iostream> // for standard I/O
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream> // string to number conversion
#include <string>  // for strings

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "std_msgs/String.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sstream>
#include <stdio.h>

#include <actionflow/frame.h>
#include <actionflow/pos.h>

using namespace std;
using namespace cv;

static const int FLOOR_WIDTH_PIX = 1080;
static const int FLOOR_LENGTH_PIX = 720;

static const float FLOOR_WIDTH_M = 3.0;
static const float FLOOR_LENGTH_M = 2.0;

static void help() {
  cout << "--------------------------------------------------------------------"
          "----------"
       << endl
       << "This script locates the robot and obstacles via open CV," << endl
       << " and publishes their positions." << endl
       << "Usage:" << endl
       << "./local num_obstacles camera_index refresh_rate" << endl
       << "--------------------------------------------------------------------"
          "------"
       << endl
       << endl;
}

static Mat unwarp(Mat input, Point2f *inputQuad) {
  Mat output;

  // Input Quadilateral or Image plane coordinates
  // Output Quadilateral or World plane coordinates
  Point2f outputQuad[4];

  // Lambda Matrix
  Mat lambda(2, 4, CV_32FC1);

  // Set the lambda matrix the same type and size as input
  lambda = Mat::zeros(FLOOR_WIDTH_PIX, FLOOR_LENGTH_PIX, input.type());

  // The 4 points that select quadilateral on the input , from top-left in
  // clockwise order These four pts are the sides of the rect box used as input

  // The 4 points where the mapping is to be done , from top-left in clockwise
  // order
  outputQuad[0] = Point2f(0, 0);
  outputQuad[1] = Point2f(FLOOR_WIDTH_PIX - 1, 0);
  outputQuad[2] = Point2f(FLOOR_WIDTH_PIX - 1, FLOOR_LENGTH_PIX - 1);
  outputQuad[3] = Point2f(0, FLOOR_LENGTH_PIX - 1);

  // Get the Perspective Transform Matrix i.e. lambda
  lambda = getPerspectiveTransform(inputQuad, outputQuad);
  // Apply the Perspective Transform just found to the src image
  warpPerspective(input, output, lambda, output.size());

  return output;
}

float dist(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) * 1.0);
}

struct DistanceFunc {
  DistanceFunc(const Point &_p) : p(_p) {}

  bool operator()(const Point &lhs, const Point &rhs) const {
    return dist(p, lhs) < dist(p, rhs);
  }

private:
  Point p;
};

Point get_closest(Point p, vector<Point> list) {
  sort(list.begin(), list.end(), DistanceFunc(p));

  return list[0];
}
void onMouse(int evt, int x, int y, int flags, void *param) {
  if (evt == EVENT_LBUTTONDOWN) {
    cv::Point2f *ptPtr = (cv::Point2f *)param;
    ptPtr->x = x;
    ptPtr->y = y;
  }
}

float orientation(Point p1, Point p2, float &v) {
  v = dist(p1, p2);
  if ((p1.y - p2.y) != 0)
    return ((float)(p1.y - p2.y) / (float)(p1.x - p2.x));
  else {
    return INFINITY;
  }
}

const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_detection_name = "Object Detection";
int robot_low_H = 64, robot_low_S = 0, robot_low_V = 237;
int robot_high_H = 83, robot_high_S = 48, robot_high_V = 255;

int obstacle_low_H = 40, obstacle_low_S = 50, obstacle_low_V = 150;
int obstacle_high_H = 60, obstacle_high_S = 100, obstacle_high_V = 255;

// inRange(frame_HSV, Scalar(36, 105, 25), Scalar(86, 220, 220), mask_BGR);
static void on_low_H_thresh_trackbar(int, void *) {
  robot_low_H = min(robot_high_H - 1, robot_low_H);
  setTrackbarPos("Low H", window_detection_name, robot_low_H);
}
static void on_high_H_thresh_trackbar(int, void *) {
  robot_high_H = max(robot_high_H, robot_low_H + 1);
  setTrackbarPos("High H", window_detection_name, robot_high_H);
}
static void on_low_S_thresh_trackbar(int, void *) {
  robot_low_S = min(robot_high_S - 1, robot_low_S);
  setTrackbarPos("Low S", window_detection_name, robot_low_S);
}
static void on_high_S_thresh_trackbar(int, void *) {
  robot_high_S = max(robot_high_S, robot_low_S + 1);
  setTrackbarPos("High S", window_detection_name, robot_high_S);
}
static void on_low_V_thresh_trackbar(int, void *) {
  robot_low_V = min(robot_high_V - 1, robot_low_V);
  setTrackbarPos("Low V", window_detection_name, robot_low_V);
}
static void on_high_V_thresh_trackbar(int, void *) {
  robot_high_V = max(robot_high_V, robot_low_V + 1);
  setTrackbarPos("High V", window_detection_name, robot_high_V);
}

static bool cluster_points(const Mat &frame_HSV, int num_objs, const Mat &mask,
                           Mat &centers, bool show_image) {

  int attempts = 5;
  try {
    // Apply threshold
    Mat result_BGR, labels;
    Mat erode_frame, dilate_frame, gray_frame, b_frame;

    bitwise_and(frame_HSV, frame_HSV, result_BGR, mask);

    if (show_image && !result_BGR.empty()) {
      imshow(window_detection_name, result_BGR);
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * 1 + 1, 2 * 1 + 1),
                                       Point(1, 1));

    erode(result_BGR, erode_frame, kernel);
    dilate(erode_frame, dilate_frame, kernel);
    cvtColor(dilate_frame, gray_frame, COLOR_BGR2GRAY);
    threshold(gray_frame, b_frame, 0, 255.0, THRESH_BINARY);

    Mat locations; // output, locations of non-zero pixels
    findNonZero(b_frame, locations);
    Mat locations_mat(locations.rows, 2, CV_32F);
    for (int i = 0; i < locations.rows; i++) {
      locations_mat.at<float>(i, 0) = locations.at<Point>(i).x;
      locations_mat.at<float>(i, 1) = locations.at<Point>(i).y;
    }

    locations_mat.convertTo(locations_mat, CV_32F);

    kmeans(locations_mat, num_objs, labels,
           TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                        10000, 0.0001),
           attempts, KMEANS_PP_CENTERS, centers);
    return true;
  } catch (...) {
    return false;
  }
}

int main(int argc, char *argv[]) {
  int radius = 1;
  std::string param;
  ros::init(argc, argv, "pos_publisher");

  ros::NodeHandle nh;
  nh.getParam("param", param);
  ROS_INFO("Got parameter : %s", param.c_str());
  ros::Time timeros = ros::Time::now();
  ros::Rate loop_rate(atoi(argv[3]));
  ros::Publisher pos_pub = nh.advertise<actionflow::frame>("robot_pos", 1000);
  // 定义节点句柄
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("video_image_complete", 1);
  sensor_msgs::ImagePtr msgPicture;

  image_transport::Publisher pub_unwrap = it.advertise("video_image_unwrap", 1);
  sensor_msgs::ImagePtr msgPicture_unwrap;
  const int num_obstacles = atoi(argv[1]);
  vector<vector<Point>> ob_centers_by_time(num_obstacles);
  Scalar color_map[14] = {
      Scalar(0, 0, 0),     Scalar(0, 0, 127),    Scalar(0, 147, 0),
      Scalar(255, 0, 0),   Scalar(127, 0, 0),    Scalar(156, 0, 156),
      Scalar(252, 127, 0), Scalar(255, 255, 0),  Scalar(0, 252, 0),
      Scalar(0, 147, 147), Scalar(0, 255, 255),  Scalar(0, 0, 252),
      Scalar(255, 0, 255), Scalar(147, 147, 147)};

  help();
  VideoCapture cap;
  cap.open(atoi(argv[2])); // atoi(param);
  if (!cap.isOpened()) {

    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(CAP_PROP_FRAME_HEIGHT, 720);
  int frame_width = int(cap.get(3));
  int frame_height = int(cap.get(4));

  cout << "Cap frame of size: " << frame_width << ", " << frame_height << endl;
  // Define the codec and create VideoWriter object.The output is stored in
  // 'outcpp.avi' file. VideoWriter video("outcpp.mov", CAP_OPENCV_MJPEG, 10,
  // Size(frame_width,frame_height));

  vector<Point2f> point_get(5);
  Vec3b color;
  int numf = 0;
  Mat frame_temp;

  Mat hsv;
  namedWindow(window_detection_name);
  // Trackbars to set thresholds for HSV values
  createTrackbar("Low H", window_detection_name, &robot_low_H, max_value_H,
                 on_low_H_thresh_trackbar);
  createTrackbar("High H", window_detection_name, &robot_high_H, max_value_H,
                 on_high_H_thresh_trackbar);
  createTrackbar("Low S", window_detection_name, &robot_low_S, max_value,
                 on_low_S_thresh_trackbar);
  createTrackbar("High S", window_detection_name, &robot_high_S, max_value,
                 on_high_S_thresh_trackbar);
  createTrackbar("Low V", window_detection_name, &robot_low_V, max_value,
                 on_low_V_thresh_trackbar);
  createTrackbar("High V", window_detection_name, &robot_high_V, max_value,
                 on_high_V_thresh_trackbar);

  while (1) {
    cout << "Frame itr: " << numf << endl;
    ++numf;
    Mat frame; // Capture frame-by-frame
    cap >> frame;

    if (numf == 5) {
      frame_temp = frame;
      int i = 0;
      while (1) {
        imshow("Quad Select", frame_temp);

        cv::setMouseCallback("Quad Select", onMouse, (void *)&point_get[0]);
        circle(frame_temp, point_get[0], 200 / 32, Scalar(255, 0, 0), 4,
               LINE_8);
        imshow("Quad Select", frame_temp);
        i++;
        char c_ = (char)waitKey(25);
        if (c_ == 27)
          break;
      }
      i = 0;
      while (1) {
        imshow("Quad Select", frame_temp);
        cv::setMouseCallback("Quad Select", onMouse, (void *)&point_get[1]);
        circle(frame_temp, point_get[1], 200 / 32, Scalar(255, 0, 0), 4,
               LINE_8);
        imshow("Quad Select", frame_temp);

        i++;

        char c_ = (char)waitKey(25);
        if (c_ == 27)
          break;
      }

      i = 0;

      while (1) {
        imshow("Quad Select", frame_temp);
        cv::setMouseCallback("Quad Select", onMouse, (void *)&point_get[2]);
        circle(frame_temp, point_get[2], 200 / 32, Scalar(255, 0, 0), 4,
               LINE_8);
        imshow("Quad Select", frame_temp);
        i++;

        char c_ = (char)waitKey(25);
        if (c_ == 27)
          break;
      }
      i = 0;
      while (1) {
        imshow("Quad Select", frame_temp);
        cv::setMouseCallback("Quad Select", onMouse, (void *)&point_get[3]);
        circle(frame_temp, point_get[3], 200 / 32, Scalar(255, 0, 0), 4,
               LINE_8);
        imshow("Quad Select", frame_temp);
        i++;

        char c_ = (char)waitKey(25);
        if (c_ == 27)
          break;
      }
      cout << "Waiting for user input." << endl;
    }
    msgPicture_unwrap =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub_unwrap.publish(msgPicture_unwrap);

    Point2f inputQuad[4];
    inputQuad[0] = Point2f(206, 253);
    inputQuad[1] = Point2f(648, 110);
    inputQuad[2] = Point2f(1122, 264);
    inputQuad[3] = Point2f(757, 698);
    int maxX, maxY = 0;

    inputQuad[0] = point_get[0];
    inputQuad[1] = point_get[1];
    inputQuad[2] = point_get[2];
    inputQuad[3] = point_get[3];

    circle(frame, inputQuad[0], 200 / 32, Scalar(255, 0, 0), 4, LINE_8);
    circle(frame, inputQuad[1], 200 / 32, Scalar(255, 0, 0), 4, LINE_8);
    circle(frame, inputQuad[2], 200 / 32, Scalar(255, 0, 0), 4, LINE_8);
    circle(frame, inputQuad[3], 200 / 32, Scalar(255, 0, 0), 4, LINE_8);
    if (frame.empty())
      break;
    // Calibrate the image
    frame = unwarp(frame, inputQuad);

    cout << frame.size() << endl;

    Mat frame_HSV, robot_mask, obstacle_mask;
    Mat robot_pos, curr_obstacle_poses;

    // Convert from BGR to HSV colorspace
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    // Detect the object based on HSV Range Value
    inRange(frame_HSV, Scalar(robot_low_H, robot_low_S, robot_low_V),
            Scalar(robot_high_H, robot_high_S, robot_high_V), robot_mask);

    inRange(frame_HSV, Scalar(obstacle_low_H, obstacle_low_S, obstacle_low_V),
            Scalar(obstacle_high_H, obstacle_high_S, obstacle_high_V),
            obstacle_mask);

    // const Mat &frame, int num_objs, Mat &centers
    bool clustered_robot =
        cluster_points(frame_HSV, 1, robot_mask, robot_pos, true);

    bool clustered_obstacles = false;
    if (num_obstacles != 0) {
      clustered_obstacles = cluster_points(
          frame_HSV, num_obstacles, obstacle_mask, curr_obstacle_poses, false);
    }

    actionflow::frame frame_obs;
    if (!clustered_robot) {
      cout << "Could not cluster robot points on this frame!" << endl;
    } else {
      frame_obs.robot_pos.x_pos =
          robot_pos.ptr<float>(0)[0] * FLOOR_WIDTH_M / FLOOR_WIDTH_PIX;
      frame_obs.robot_pos.y_pos =
          robot_pos.ptr<float>(0)[1] * FLOOR_LENGTH_M / FLOOR_LENGTH_PIX;

      Point robot_center =
          Point(robot_pos.ptr<float>(0)[0], robot_pos.ptr<float>(0)[1]);

      circle(frame, robot_center, 20, color_map[0], 6);
      std::stringstream ss;
      ss << "R:"
         << "," << round(frame_obs.robot_pos.x_pos * 100) / 100 << ","
         << round(frame_obs.robot_pos.y_pos * 100) / 100;
      string pos_string = ss.str();
      putText(frame, pos_string, robot_center, FONT_HERSHEY_DUPLEX, 1,
              Scalar(0, 0, 0), 2);
    }
    if (!clustered_obstacles) {
      cout << "Did not cluster any obstacles!" << endl;
    } else {
      vector<Point> cur_obstacle_centers;
      for (int i = 0; i < curr_obstacle_poses.rows; i++) {
        const float *Mi = curr_obstacle_poses.ptr<float>(i);
        Point center = Point(Mi[0], Mi[1]);
        cur_obstacle_centers.push_back(center);
      }

      if (ob_centers_by_time[0].size() == 0) {
        // nearest neighbor for finding velocity
        for (int i = 0; i < curr_obstacle_poses.rows; i++) {
          const float *Mi_ = curr_obstacle_poses.ptr<float>(i);
          Point center = Point(Mi_[0], Mi_[1]);
          ob_centers_by_time[i].push_back(center);
        }
      } else {
        for (int i = 0; i < curr_obstacle_poses.rows; i++) {
          // for each of the centers from the previous step,
          // Greedily pick the closest center, and delete it from the list
          // why not just use hungarian matching
          const float *Mi_ = curr_obstacle_poses.ptr<float>(i);
          Point closest =
              get_closest(ob_centers_by_time[i].back(), cur_obstacle_centers);

          float v = 0.0; // velocity
          float o = orientation(ob_centers_by_time[i].back(), closest, v);
          float theta = atan(o); // heading

          cur_obstacle_centers.erase(std::find(cur_obstacle_centers.begin(),
                                               cur_obstacle_centers.end(),
                                               closest));
          ob_centers_by_time[i].push_back(closest);

          float x_out, y_out, s_pass;

          // scale and shift x and y
          x_out =
              ob_centers_by_time[i].back().x * FLOOR_WIDTH_M / FLOOR_WIDTH_PIX;
          y_out = ob_centers_by_time[i].back().y * FLOOR_LENGTH_M /
                  FLOOR_LENGTH_PIX;

          circle(frame, ob_centers_by_time[i].back(), 20, color_map[i + 1], 5);
          std::stringstream ss;
          ss << i + 1 << ":" << round(x_out * 100.0) / 100 << ","
             << round(y_out * 100.0) / 100;
          string pos_string = ss.str();
          putText(frame, pos_string, ob_centers_by_time[i].back(),
                  FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2);

          actionflow::pos pos_msg;
          pos_msg.id = i;
          pos_msg.x_pos = x_out;
          pos_msg.y_pos = y_out;
          // pos_msg.vel = v;
          // pos_msg.heading = theta;

          frame_obs.obstacle_poses.push_back(pos_msg);
        }
      }
    }
    pos_pub.publish(frame_obs);
    ros::spinOnce();
    msgPicture =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub.publish(msgPicture);
    ros::spinOnce();
    char c_ = (char)waitKey(25);
    if (c_ == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();
  // video.release();
  // Closes all the frames
  destroyAllWindows();
  return 0;
}
