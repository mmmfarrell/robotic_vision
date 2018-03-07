#include "stereo_odom.h"

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
  focal_ = 718.8560f;
  pp_.x = 607.1928f;
  pp_.y = 185.2157f;

  // Init starting pose of camera
  R_ = Mat::eye(3, 3, CV_64F);
  t_ = Mat::zeros(3, 1, CV_64F);

  // Init camera projection matrix
  Proj_ = Mat::zeros(3, 4, CV_64F);
  Proj_.at<double>(0, 0) = focal_;
  Proj_.at<double>(0, 2) = pp_.x;
  Proj_.at<double>(1, 1) = focal_;
  Proj_.at<double>(1, 2) = pp_.y;
  Proj_.at<double>(2, 2) = 1.0f;
}

VisualOdom::~VisualOdom()
{}

void VisualOdom::stereoPoints(Mat img1, Mat img2, Mat img3, Mat img4)
{
  // Make new mats
  Mat img1_points, img2_points;
  img1.copyTo(img1_points);
  img2.copyTo(img2_points);

  // Calc ORB points for both frames
  vector<KeyPoint> kp1, kp2, kp3, kp4;
  Mat desc1, desc2, desc3, desc4;
  orb_->detectAndCompute(img1, noArray(), kp1, desc1);
  orb_->detectAndCompute(img2, noArray(), kp2, desc2);
  orb_->detectAndCompute(img3, noArray(), kp3, desc3);
  orb_->detectAndCompute(img4, noArray(), kp4, desc4);

  // Match ORB features
  vector<vector<DMatch>> matches, matches2;
  matcher_->knnMatch(desc1, desc2, matches, 2);
  matcher_->knnMatch(desc3, desc4, matches2, 2);

  // desc
  cout << "desc " << desc1.rows << " x " << desc1.cols << endl;
  cout << "desc " << desc2.rows << " x " << desc2.cols << endl;

  // Put matched features into vectors
  vector<KeyPoint> matched1, matched2, matched3, matched4;

  for(unsigned i = 0; i < matches.size(); i++)
  {
    if (matches[i][0].distance < nn_match_ratio_ * matches[i][1].distance)
    {
      matched1.push_back(kp1[matches[i][0].queryIdx]);
      matched2.push_back(kp2[matches[i][0].trainIdx]);
    }
  }

  for(unsigned i = 0; i < matches2.size(); i++)
  {
    if (matches2[i][0].distance < nn_match_ratio_ * matches2[i][1].distance)
    {
      matched3.push_back(kp1[matches2[i][0].queryIdx]);
      matched4.push_back(kp2[matches2[i][0].trainIdx]);
    }
  }

  // Disp # of points left
  cout << "matched 1: " << matched1.size() << " matched 2: " << matched2.size() << endl;
  cout << "matched 3: " << matched3.size() << " matched 4: " << matched4.size() << endl;

  // Match first and second set of images
  orb_->compute(img1, matched1, desc1);
  orb_->compute(img3, matched3, desc3);

  // Find matches between sets
  vector<vector<DMatch>> tot_matches;
  matcher_->knnMatch(desc1, desc3, tot_matches, 2);

  // Put matched features into vectors
  vector<KeyPoint> sm1, sm2, sm3, sm4;

  for(unsigned i = 0; i < tot_matches.size(); i++)
  {
    if (tot_matches[i][0].distance < nn_match_ratio_ * tot_matches[i][1].distance)
    {
      sm1.push_back(matched1[matches[i][0].queryIdx]);
      sm2.push_back(matched3[matches[i][0].trainIdx]);
      sm3.push_back(matched2[matches[i][0].queryIdx]);
      sm4.push_back(matched4[matches[i][0].trainIdx]);
    }
  }

  // Disp # of points left
  cout << "SM lengths: " << endl << sm1.size() << endl << sm2.size() << endl << sm3.size() << endl << sm4.size() << endl;

  vector<Point2f> smp1, smp2;
  KeyPoint::convert(sm1, smp1);
  KeyPoint::convert(sm2, smp2);

  Mat inlier_mask, E, R, t;
  vector<KeyPoint> inliers1, inliers2, inliers3, inliers4;
  vector<DMatch> inlier_matches;

  // Recover Pose
  E = findEssentialMat(mpoints1, mpoints2, focal_, pp_, RANSAC, 0.999, 1.0, inlier_mask);
  recoverPose(E, mpoints1, mpoints2, R, t, focal_, pp_, inlier_mask);

  // Draw matches on frames
  for (unsigned i = 0; i < matched1.size(); i++)
  {
    if (inlier_mask.at<uchar>(i))
    {
      int new_i = static_cast<int>(inliers1.size());
      inliers1.push_back(sm1[i]);
      inliers2.push_back(sm2[i]);
      inliers3.push_back(sm3[i]);
      inliers4.push_back(sm4[i]);
      inlier_matches.push_back(DMatch(new_i, new_i, 0));
    }
  }

  // Constuct proj points
  //int num_points = 10;
  int num_points = (int)inliers1.size();
  Mat pp1, pp2, pm1, pm2;
  pp1 = Mat::zeros(2, num_points, CV_64F);
  pp2 = Mat::zeros(2, num_points, CV_64F);
  for (int i = 0; i < num_points; i++)
  {
    pp1.at<double>(0,i) = inliers1[i].pt.x;
    pp1.at<double>(1,i) = inliers1[i].pt.y;
    pp2.at<double>(0,i) = inliers2[i].pt.x;
    pp2.at<double>(1,i) = inliers2[i].pt.y;
  }

  Proj_.copyTo(pm1);
  Proj_.copyTo(pm2);
  pm2.at<double>(0,3) = -386.1448;
  pm2.at<double>(1,3) = 0.0;

  // Triangulate Points
  Mat p4D;
  triangulatePoints(pm1, pm2, pp1, pp2, p4D);

  // (un) Normalize points
  vector<Point3f> p3D;
  for (int i = 0; i < num_points; i++)
  {
    Point3f p;
    p.x = p4D.at<double>(0,i)/p4D.at<double>(3,i);
    p.y = p4D.at<double>(1,i)/p4D.at<double>(3,i);
    p.z = p4D.at<double>(2,i)/p4D.at<double>(3,i);
    p3D.push_back(p);

    // Display points
    //cout << "Euclidean Point: " << endl << p.x << endl << p.y << endl << p.z << endl << endl;
  }

  // Display homo points
  //cout << "Homo points: " << p4D(Range(0,4), Range(0,4));

  // Display Projection Matrices
  cout << "P0: " << pm1 << endl;
  cout << "P1: " << pm2 << endl;

  cout << "num_points: " << num_points << endl;

  // Display R, t
  //cout << "R: " << R << endl;
  //cout << "t: " << t << endl;

  // Show concated imgs
  Mat imgs, imgs2, imgs_big;
  //cv::hconcat(img1_points, img2_points, imgs);
  drawMatches(img1, inliers1, img2, inliers2, inlier_matches, imgs, Scalar(255,0,0), Scalar(255,0,0));
  drawMatches(img3, inliers3, img4, matched4, inlier_matches, imgs2, Scalar(255,0,0), Scalar(255,0,0));
  vconcat(imgs, imgs2, imgs_big);
  
  char c(0);
  while (c != 27)
  {
    imshow("Stereo", imgs_big);
    c = (char)waitKey(2);
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

double VisualOdom::calcRelativeScale(Mat img1, Mat img2, Mat img3)
{
  cout << endl << endl << endl << "calcRelativeScale" << endl;
  // Detect features in img1
  vector<Point2f> features1, features2, features3;
  VisualOdom::detectFeatures(img1, features1);
  //VisualOdom::trackFeatures(img1, img2, features1, features2);
  //VisualOdom::trackFeatures(img2, img3, features2, features3);
  trackFeatures3(img1, img2, img3, features1, features2, features3);

  Mat t1, t2, R1, R2, img_out;
  calcOdom(img1, img2, R1, t1, img_out);
  calcOdom(img2, img3, R2, t2, img_out);

  // Assemble projection matrices
  double scale = 1.0f;
  Mat proj1, proj2, proj3;
  //proj1 = Proj_;
  //proj2 = Proj_;
  //proj3 = Proj_;
  Proj_.copyTo(proj1);
  Proj_.copyTo(proj2);
  Proj_.copyTo(proj3);

  Mat rt1, rt2;
  rt1 = R1*t1;
  rt2 = R2*t2;

  proj2.at<double>(0, 3) = rt1.at<double>(0);
  proj2.at<double>(1, 3) = rt1.at<double>(1);
  proj2.at<double>(2, 3) = rt1.at<double>(2);
  cout << "Proj 1: " << proj1 << endl;
  cout << "Proj 2: " << proj2 << endl;

  proj3.at<double>(0, 3) = rt2.at<double>(0);
  proj3.at<double>(1, 3) = rt2.at<double>(1);
  proj3.at<double>(2, 3) = rt2.at<double>(2);
  cout << "Proj 3: " << proj3 << endl;

  cout << "features1 x: " << features1[1].x << ", y: " << features1[1].y << endl;
  cout << "features2 x: " << features2[1].x << ", y: " << features2[1].y << endl;
  cout << "features3 x: " << features3[1].x << ", y: " << features3[1].y << endl;

  // Compute triangulate points
  Mat points4D1, points4D2;
  cv::triangulatePoints(proj1, proj2, features1, features2, points4D1);
  cv::triangulatePoints(proj1, proj3, features2, features3, points4D2);
  //cout << "Points 4D: " << points4D1 << endl;
  cout << "Points 4D 1: x: " << points4D1.at<double>(0,1) << ", y: " << points4D1.at<double>(1,1) << ", z: " << points4D1.at<double>(2,1) << ", 1: " << points4D1.at<double>(3,1) << endl;
  cout << "Points 4D 2: x: " << points4D2.at<double>(0,1) << ", y: " << points4D2.at<double>(1,1) << ", z: " << points4D2.at<double>(2,1) << ", 1: " << points4D2.at<double>(3,1) << endl;

  cout << "Point 4D: " << points4D1(Range(0,4), Range(0,3)) << endl;

  //Mat euclid1;
  //cv::convertPointsFromHomogeneous(points4D1, euclid1);
  //cout << "Point Euclid: " << euclid1(Range(0,3), Range(0,3)) << endl;

  cout << "Features length 1: " << features1.size() << ", 2: " << features2.size() << ", 3: " << features3.size() << endl;
  cout << "4D rows: " << points4D1.rows << ", cols: " << points4D1.cols << endl;
  cout << "4D rows: " << points4D2.rows << ", cols: " << points4D2.cols << endl;

  // Compute average scale factor
  int num_features = points4D1.cols;
  double avg_scale = 0.0;
  //Mat diff_points = points4D2 -

  for (int i = 0; i < num_features; i++)
  {
    int j = i;
    while(i == j)
    {
      j = rand() % num_features;
    }
    double x_i1, y_i1, z_i1, x_i2, y_i2, z_i2;
    double x_j1, y_j1, z_j1, x_j2, y_j2, z_j2;
    double dist1, dist2, scale_est;
    x_i1 = points4D1.at<double>(0, i)/points4D1.at<double>(3, i);
    y_i1 = points4D1.at<double>(1, i)/points4D1.at<double>(3, i);
    z_i1 = points4D1.at<double>(2, i)/points4D1.at<double>(3, i);
    //cout << "i: " << i << " j: " << j << endl;
    //cout << "Xi1 x: " << x_i1 << ", y: " << y_i1 << ", z: " << z_i1 << endl;
    x_i2 = points4D2.at<double>(0, i)/points4D2.at<double>(3, i);
    y_i2 = points4D2.at<double>(1, i)/points4D2.at<double>(3, i);
    z_i2 = points4D2.at<double>(2, i)/points4D2.at<double>(3, i);

    if (true) //(i == 0)
    {
      //cout << endl << "Check point" << endl;
      Mat homo_point = Mat::zeros(4, 1, CV_64F);
      Mat cam_matrix = Mat::zeros(3, 3, CV_64F);
      Mat Rt = Mat::zeros(3, 4, CV_64F);

      cam_matrix.at<double>(0, 0) = focal_;
      cam_matrix.at<double>(1, 1) = focal_;
      cam_matrix.at<double>(0, 2) = pp_.x;
      cam_matrix.at<double>(1, 2) = pp_.y;
      
      //cout << "R1: " << R1 << endl;
      //Rt(Range(0, 3), Range(0, 3)) = R1(Range(0,3), Range(0,3));
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          Rt.at<double>(i,j) = R1.at<double>(i,j);
        }
      }
      Rt.at<double>(0, 3) = t1.at<double>(0,0);
      Rt.at<double>(1, 3) = t1.at<double>(1,0);
      Rt.at<double>(2, 3) = t1.at<double>(2,0);
      //Rt(Range(0, 3), Range(4, 5)) = t1;

      homo_point.at<double>(0, 0) = x_i2;
      homo_point.at<double>(1, 0) = y_i2;
      homo_point.at<double>(2, 0) = z_i2;
      homo_point.at<double>(3, 0) = 1.0f;

      cout << "Cam Matrix: " << cam_matrix << endl;
      cout << "Rt: " << Rt << endl;
      cout << "Homo point: " << homo_point << endl;

      Mat homo_pix = cam_matrix * Rt * homo_point;
      cout << "Homo pix: " << homo_pix << endl;
       
    }


    x_j1 = points4D1.at<double>(0, j)/points4D1.at<double>(3, i);
    y_j1 = points4D1.at<double>(1, j)/points4D1.at<double>(3, i);
    z_j1 = points4D1.at<double>(2, j)/points4D1.at<double>(3, i);
    //cout << "Xj1 x: " << x_j1 << ", y: " << y_j1 << ", z: " << z_j1 << endl;

    x_j2 = points4D2.at<double>(0, j)/points4D2.at<double>(3, i);
    y_j2 = points4D2.at<double>(1, j)/points4D2.at<double>(3, i);
    z_j2 = points4D2.at<double>(2, j)/points4D2.at<double>(3, i);

    dist1 = sqrt(pow(x_i1 - x_j1, 2) + pow(y_i1 - y_j1, 2) + pow(z_i1 - z_j1,2));
    dist2 = sqrt(pow(x_i2 - x_j2, 2) + pow(y_i2 - y_j2, 2) + pow(z_i2 - z_j2,2));

    scale_est = dist1/dist2;
    //cout << "scale est: " << scale_est << endl;
    avg_scale += scale_est;

  }
  avg_scale /= (double)num_features;
  //cout << "avg scale: " << avg_scale << endl << endl << endl << endl;

  return 0.0f;
}

void VisualOdom::detectFeatures(Mat img, vector<Point2f>& points)
{
  //int max_corners_ = 25;
  int max_corners_ = 500;
  double quality_level = 0.01;
  double min_distance = 10;
  TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 20, 0.03);
  Size subPixWinSize(10, 10), winSize(31, 31);
  goodFeaturesToTrack(img, points, max_corners_, quality_level, min_distance );
  cornerSubPix(img, points, subPixWinSize, cv::Size(-1, -1), termcrit);
}

void VisualOdom::trackFeatures3(Mat img1, Mat img2, Mat img3, vector<Point2f>& points1, vector<Point2f>& points2, vector<Point2f>& points3)
{
  // Calculate optical flow of points1
  vector<uchar> status1, status2;
  vector<float> err1, err2;
  calcOpticalFlowPyrLK(img1, img2, points1, points2, status1, err1);
  calcOpticalFlowPyrLK(img2, img3, points2, points3, status2, err2);

  // Get rid of failed points or outside of frame
  int idx = 0;
  double pix_vel, pix_vel_thresh = 4.0;
  for (int i = 0; i < status1.size(); i++)
  {
    Point2f pt1, pt2;
    pt1 = points2.at(i - idx);
    pt1 = points3.at(i - idx);
    if ((status1.at(i) == 0) || (status2.at(i) == 0) || (pt1.x<0) || (pt1.y<0) || (pt2.x<0) || (pt2.y<0))
    {
      points1.erase(points1.begin() + (i - idx));
      points2.erase(points2.begin() + (i - idx));
      points3.erase(points3.begin() + (i - idx));
      idx++;
    }
    //else
    //{
      //pix_vel = sqrt(pow(points2[i-idx].x - points1[i-idx].x, 2) + pow(points2[i-idx].y - points1[i-idx].y, 2));
      
      //// Get rid of points if OF isn't greater than threshold
      //if (pix_vel < pix_vel_thresh)
      //{
        //points1.erase(points1.begin() + (i - idx));
        //points2.erase(points2.begin() + (i - idx));
        //idx++;
      //}
    //}
  }
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
  Mat img1, img2, img3, img_out;
  char filename1[200], filename2[200], filename3[200], filename4[200];

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
    sprintf(filename3, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_1/%06d.png", i);
    sprintf(filename4, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_1/%06d.png", i+1);

    Mat img4;
    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
    img3 = imread(filename3, CV_LOAD_IMAGE_GRAYSCALE);
    img4 = imread(filename4, CV_LOAD_IMAGE_GRAYSCALE);

    // Triangulate points
    vo.stereoPoints(img1, img2, img3, img4);

    //// Get VO
    ////vo.calcOdom(img1, img2, R_step, t_step, img_out);
    
    //double rel_scale = 0.0f;
    //rel_scale = vo.calcRelativeScale(img1, img2, img3);

    //// Only update if we have moved and if our resultant rotation and translation are valid
    //// Note translation valid if principle motion is in the z direction (along camera axis)
    //if ((scale > 0.1) && (!R_step.empty()) && (t_step.at<double>(2) < t_step.at<double>(0)) && (t_step.at<double>(2) < t_step.at<double>(1)))
    //{
      //// Move our estimate
      //t_tot += scale*(R_tot*t_step);
      //R_tot = R_step * R_tot;
    //}
    //else
    //{
      //cout << "Skipped Update!!!" << endl;
    //}

    //// Plot trajectory
    //int x_traj = -int(t_tot.at<double>(0)) + 300;
    //int y_traj = -int(t_tot.at<double>(2)) + 100;
    //circle(traj, Point(x_traj, y_traj), 1, Scalar(255, 0, 0), 2);

    //// Determine runtime
    //t = clock() - t;
    ////std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    ////imshow("Matches", res);
    //imshow("Camera", img_out);
    //imshow("Trajectory", traj);
    //char c(0);
    //c = (char)waitKey(2);
    //if ( c == 27 )
      //break;
    //break;
  }


  return 0;
}

