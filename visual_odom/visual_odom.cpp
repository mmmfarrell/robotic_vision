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

  // Camera Projection matrices
  P0_ = Mat::zeros(3, 4, CV_64F);
  P0_.at<double>(0, 0) = 1.0f;
  P0_.at<double>(1, 1) = 1.0f;
  //P0_.at<double>(0, 2) = pp_.x;
  //P0_.at<double>(1, 2) = pp_.y;
  P0_.at<double>(2, 2) = 1.0f;

  P0_.copyTo(P1_);
  P1_.at<double>(0, 3) = 0.537;

  //P0_.at<double>(0, 0) = focal_;
  //P0_.at<double>(1, 1) = focal_;
  //P0_.at<double>(0, 2) = pp_.x;
  //P0_.at<double>(1, 2) = pp_.y;
  //P0_.at<double>(2, 2) = 1.0f;

  //P0_.copyTo(P1_);
  //P1_.at<double>(0, 3) = -386.1448f;

  dist_coeff_ = Mat::zeros(1, 5, CV_64F);

  // Init starting pose of camera
  R_ = Mat::eye(3, 3, CV_64F);
  t_ = Mat::zeros(3, 1, CV_64F);
}

VisualOdom::~VisualOdom()
{}

void VisualOdom::findPoints(Mat imgL, Mat imgR, vector<Point3f>& points, vector<Point2f>& features1, vector<Point2f>& features2)
{
  // Detect features in img1
  //vector<Point2f> features1, features2;
  VisualOdom::matchFeatures(imgL, imgR, features1, features2);

  // Triangulate Points
  //cout << "P0: " << P0_ << endl;
  //cout << "P1: " << P1_ << endl;
  vector<Point2f> undistort1, undistort2;
  cv::undistortPoints(features1, undistort1, K_, Mat());
  cv::undistortPoints(features2, undistort2, K_, Mat());
  Mat points4D(4, features1.size(), CV_32FC1);
  triangulatePoints(P0_, P1_, undistort1, undistort2, points4D);

  // Recover 3d points from homogeneous
  Mat pt3D;
  cv::convertPointsFromHomogeneous(Mat(points4D.t()).reshape(4,1), pt3D);
  //cout << "3D points: " << pt3D << endl;

  // Make 2d points vector
  for (int i = 0; i < features1.size(); i++)
  {
    //Extract Point
    Point3f point;
    point.x = pt3D.at<float>(i, 0);
    point.y = pt3D.at<float>(i, 1);
    point.z = pt3D.at<float>(i, 2);

    if (point.z > 0)
    {
      continue;
    }
    else if (point.z < -50.0f)
    {
      continue;
    }
    else if (features1[i].y > 100)
    {
      continue;
    }
    else
    {
      points.push_back(point);
    }
  }
}

void VisualOdom::matchFeatures(Mat imgL, Mat imgR, vector<Point2f>& features1, vector<Point2f>& features2)
{
  // Calc ORB points for both frames
  vector<KeyPoint> kp1, kp2;
  Mat desc1, desc2;

  orb_->detectAndCompute(imgL, noArray(), kp1, desc1);
  orb_->detectAndCompute(imgR, noArray(), kp2, desc2);

  // Match ORB features
  vector< vector<DMatch> > matches;
  matcher_->knnMatch(desc1, desc2, matches, 2);

  // Put matched features into vectors
  vector<KeyPoint> matched1, matched2;

  for(unsigned i = 0; i < matches.size(); i++)
  {
    if (matches[i][0].distance < nn_match_ratio_ * matches[i][1].distance)
    {
      matched1.push_back(kp1[matches[i][0].queryIdx]);
      matched2.push_back(kp2[matches[i][0].trainIdx]);
    }
  }

  // Convert matched features to Point2f
  cv::KeyPoint::convert(matched1, features1);
  cv::KeyPoint::convert(matched2, features2);

  //// Compute Essential Matrix and recover pose
  //Mat E, inlier_mask, R, t;
  //E = findEssentialMat(features1, features2, focal_, pp_, RANSAC, 0.999, 1.0, inlier_mask);
  //recoverPose(E, features1, features2, R, t, focal_, pp_, inlier_mask);

  //cout << "R: " << R << endl;
  //cout << "t: " << t << endl;
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
