#include "visual_odom.h"

using namespace cv;
using namespace std;

namespace robotic_vision
{

VisualOdom::VisualOdom()
{
  //cout << "Constructor" << endl;

  // Set thresholds
  ransac_thresh_ = 2.5f;
  nn_match_ratio_ = 0.8f;

  // Create ORB detector and descriptor matcher
  orb_ = ORB::create();
  matcher_ = DescriptorMatcher::create("BruteForce-Hamming");

  // Set camera intrinsics
  focal_ = 718.8560;
  pp_.x = 607.1928;
  pp_.y = 185.2157;

  // Init starting pose of camera
  R_ = Mat::eye(3, 3, CV_64F);
  t_ = Mat::zeros(3, 1, CV_64F);
}

VisualOdom::~VisualOdom()
{}

void VisualOdom::calcOdom(Mat img1, Mat img2, Mat& R, Mat& t, Mat& out)
{
  // Detect features in img1
  vector<Point2f> features1, features2;
  VisualOdom::detectFeatures(img1, features1);
  VisualOdom::trackFeatures(img1, img2, features1, features2);

  Mat inlier_mask, E;

  // If we have enough matches, recover pose
  if (features2.size() > 100) //(matched1.size() >= 4)
  {
    E = findEssentialMat(features1, features2, focal_, pp_, RANSAC, 0.999, 0.5, inlier_mask);
    recoverPose(E, features1, features2, R, t, focal_, pp_, inlier_mask);
  }

  // Plot points on imgs
  cvtColor(img2, out, COLOR_GRAY2BGR);

  // If we don't get a good pose then just return the frames
  if (!E.empty() )
  {
    // Draw matches on frames
    vector<Point2f> inliers1, inliers2;

    for (unsigned i = 0; i < features2.size(); i++)
    {
      if (inlier_mask.at<uchar>(i))
      {
        // Draw circles and line to connect points from one frame to another
        circle(out, features1[i], 2, Scalar(0, 255, 0), 3);
        circle(out, features2[i], 2, Scalar(0, 0, 255), 3);
        line(out, features1[i], features2[i], Scalar(0, 255, 0));
      }
    }
  }
}

void VisualOdom::detectFeatures(Mat img, vector<Point2f>& points)
{
  int max_corners_ = 500;
  double quality_level = 0.01;
  double min_distance = 10;
  goodFeaturesToTrack(img, points, max_corners_, quality_level, min_distance );
}

void VisualOdom::trackFeatures(Mat img1, Mat img2, vector<Point2f>& points1, vector<Point2f>& points2)
{
  // Calculate optical flow of points1
  vector<uchar> status;
  vector<float> err;
  calcOpticalFlowPyrLK(img1, img2, points1, points2, status, err);

  // Get rid of failed points or outside of frame
  int idx = 0;
  double pix_vel, pix_vel_thresh = 4.0;
  for (int i = 0; i < status.size(); i++)
  {
    Point2f pt;
    pt = points2.at(i - idx);
    if ((status.at(i) == 0) || (pt.x<0) || (pt.y<0))
    {
      points1.erase(points1.begin() + (i - idx));
      points2.erase(points2.begin() + (i - idx));
      idx++;
    }
    else
    {
      pix_vel = sqrt(pow(points2[i-idx].x - points1[i-idx].x, 2) + pow(points2[i-idx].y - points1[i-idx].y, 2));
      
      // Get rid of points if OF isn't greater than threshold
      if (pix_vel < pix_vel_thresh)
      {
        points1.erase(points1.begin() + (i - idx));
        points2.erase(points2.begin() + (i - idx));
        idx++;
      }
    }
  }
}

} // end namespace

int main(int argc, char** argv)
{

  // TODO: Remove
  int num_frames = 4500;

  // Instantiate VisualOdom class
  robotic_vision::VisualOdom vo;

  // Load images
  Mat img1, img2, img_out;
  char filename1[200], filename2[200];

  // Mat for trajectory
  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  // Read file for true position
  ifstream myfile("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/poses/00.txt");

  // Truth parameters
  double x_truth, y_truth, z_truth, var;
  double scale, x_prev(0), y_prev(0), z_prev(0);
  Mat R_truth = Mat::eye(3, 3, CV_64F);
  Mat t_truth = Mat::zeros(3, 1, CV_64F);

  char line[256];

  // Total rotation and translation
  Mat R_tot = Mat::eye(3, 3, CV_64F);
  Mat t_tot = Mat::zeros(3, 1, CV_64F);
  Mat R_step, t_step;

  for (int i = 0; i < num_frames; i++)
  {
    clock_t t;
    t = clock();

    // Get truth info for this frame
    myfile.getline(line, 256);
    std::istringstream in(line);

    for (int i = 0; i < 12; i++)
    {
      in >> var;
      if (i < 3)
        R_truth.at<double>(0, i) = var;
      else if (i == 3)
        x_truth = var;
      else if (i < 7)
        R_truth.at<double>(1, i-4) = var;
      else if (i == 7)
        y_truth = var;
      else if (i < 11)
        R_truth.at<double>(2, i-8) = var;
      else if (i == 11)
        z_truth = var;
    }

    // Calc t_truth
    t_truth.at<double>(0) = x_truth - x_prev;
    t_truth.at<double>(1) = y_truth - y_prev;
    t_truth.at<double>(2) = z_truth - z_prev;

    // Calculate scale of movement
    scale = sqrt(pow(x_truth - x_prev, 2) + pow(y_truth - y_prev, 2) + pow(z_truth - z_prev, 2));
    x_prev = x_truth;
    y_prev = y_truth;
    z_prev = z_truth;

    // Plot truth point as red
    circle(traj, Point(-x_truth + 300, z_truth + 100), 1, Scalar(0, 0, 255), 2);

    // Get next 2 images
    sprintf(filename1, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i);
    sprintf(filename2, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i+1);

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    // Get VO
    vo.calcOdom(img1, img2, R_step, t_step, img_out);

    // Only update if we have moved and if our resultant rotation and translation are valid
    // Note translation valid if principle motion is in the z direction (along camera axis)
    if ((scale > 0.1) && (!R_step.empty()) && (t_step.at<double>(2) < t_step.at<double>(0)) && (t_step.at<double>(2) < t_step.at<double>(1)))
    {
      // Move our estimate
      t_tot += scale*(R_tot*t_step);
      R_tot = R_step * R_tot;
    }
    else
    {
      cout << "Skipped Update!!!" << endl;
    }

    // Plot trajectory
    int x_traj = -int(t_tot.at<double>(0)) + 300;
    int y_traj = -int(t_tot.at<double>(2)) + 100;
    circle(traj, Point(x_traj, y_traj), 1, Scalar(255, 0, 0), 2);

    // Determine runtime
    t = clock() - t;
    //std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    //imshow("Matches", res);
    imshow("Camera", img_out);
    imshow("Trajectory", traj);
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;
  }


  return 0;
}

