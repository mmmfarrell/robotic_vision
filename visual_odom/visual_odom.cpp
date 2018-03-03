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

Mat VisualOdom::calcOdom(Mat img1, Mat img2, double scale)
{
  // Detect features in img1
  vector<Point2f> features1, features2;
  VisualOdom::detectFeatures(img1, features1);
  VisualOdom::trackFeatures(img1, img2, features1, features2);

  Mat inlier_mask, E, R, t;
  vector<KeyPoint> inliers1, inliers2;

  // TODO: Remove
  Mat res;

  // If we have enough matches, recover pose
  if (features2.size() > 100) //(matched1.size() >= 4)
  {
    E = findEssentialMat(features1, features2, focal_, pp_, RANSAC, 0.999, 1.0, inlier_mask);
    recoverPose(E, features1, features2, R, t, focal_, pp_, inlier_mask);

    // Move my overall pose
    t_ += scale*(R_*t);
    R_ = R*R_;
    //t_ = t;
    //R_ = R;
  }

  //// If we don't get a good pose then just return the frames
  //if (matched1.size() < 4 || E.empty() )
  //{
    //hconcat(img1, img2, res);
    //cout << "No update" << endl;
  //}
  //else
  //{
    //// Draw matches on frames
    //for (unsigned i = 0; i < matched1.size(); i++)
    //{
      //if (inlier_mask.at<uchar>(i))
      //{
        //int new_i = static_cast<int>(inliers1.size());
        //inliers1.push_back(matched1[i]);
        //inliers2.push_back(matched2[i]);
        //inlier_matches.push_back(DMatch(new_i, new_i, 0));
      //}
    //}
    //drawMatches(img1, inliers1, img2, inliers2, inlier_matches, res, Scalar(255,0,0), Scalar(255,0,0));
    //cout << "inlier_matches length: " << inlier_matches.size() << endl;
    ////cout << "Draw MAtches" << endl;
  //}

  return res;

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
  Mat img1, img2, res;
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
    circle(traj, Point(x_truth + 300, z_truth + 100), 1, Scalar(0, 0, 255), 2);

    // Get next 2 images
    sprintf(filename1, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i);
    sprintf(filename2, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i+1);

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    // Get VO
    res = vo.calcOdom(img1, img2, scale);

    // Update total rotation and translation with true scale factor
    // TODO return bool to determine if the vo was good (to update R or not)
    //t_tot += scale*(R_tot*vo.t_);
    //R_tot = vo.R_*R_tot;
    t_tot = vo.t_;
    R_tot = vo.R_;

    // Plot trajectory
    int x_traj = int(t_tot.at<double>(0)) + 300;
    int y_traj = -int(t_tot.at<double>(2)) + 100;
    circle(traj, Point(x_traj, y_traj), 1, Scalar(255, 0, 0), 2);

    // Determine runtime
    t = clock() - t;
    //std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    //imshow("Matches", res);
    imshow("Camera", img2);
    imshow("Trajectory", traj);
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;
  }


  return 0;
}

