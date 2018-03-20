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

  // Camera calibration matrix
  K_ = Mat::zeros(3, 3, CV_64F);
  K_.at<double>(0, 0) = focal_;
  K_.at<double>(1, 1) = focal_;
  K_.at<double>(0, 2) = pp_.x;
  K_.at<double>(1, 2) = pp_.y;
  K_.at<double>(2, 2) = 1.0f;

  // Init starting pose of camera
  R_ = Mat::eye(3, 3, CV_64F);
  t_ = Mat::zeros(3, 1, CV_64F);
}

VisualOdom::~VisualOdom()
{}

void VisualOdom::findPoints(Mat img1, Mat Rt1, Mat img2, Mat Rt2, vector<Point2f>& points)
{
  // Detect features in img1
  vector<Point2f> features1, features2;
  VisualOdom::detectFeatures(img1, features1);
  VisualOdom::trackFeatures(img1, img2, features1, features2);

  // Undistort pixels
  Mat distort_coeff = (Mat1d(1,4) << 0.0, 0.0, 0.0, 0.0);
  vector<Point2f> undistort1, undistort2;
  undistortPoints(features1, undistort1, K_, distort_coeff);
  undistortPoints(features2, undistort2, K_, distort_coeff);

  cout << "Original point x: " << features1[0].x << ", y: " << features1[0].y << endl;
  cout << "Undistort point x: " << undistort1[0].x << ", y: " << undistort1[0].y << endl;

  // Create Projection matrices
  Mat projM1, projM2;
  projM1 = Rt1;
  projM2 = Rt2;
  //cout << "K matrix: " << K_ << endl;
  cout << "projM1: " << projM1 << endl;
  cout << "projM2: " << projM2 << endl;

  // Triangulate Points
  Mat points4D;
  triangulatePoints(projM1, projM2, undistort1, undistort2, points4D);

  // Make 2D points vector from result
  for (int i = 0; i < features1.size(); i++)
  {
    // Extract point
    Point2f point;
    point.x = (float)points4D.at<double>(0, i)/points4D.at<double>(3, i);
    point.y = (float)points4D.at<double>(2, i)/points4D.at<double>(3, i); // z component in camera axes

    // Add to output vector
    points.push_back(point);

    // TODO Remove
    if (i == 0)
    {
      cout << "Point x: " << point.x << ", z: " << point.y << endl;
    }
  }
}

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
