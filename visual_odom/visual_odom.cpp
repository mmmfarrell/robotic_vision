#include "visual_odom.h"

using namespace cv;
using namespace std;

namespace robotic_vision
{

VisualOdom::VisualOdom()
{
  cout << "Constructor" << endl;

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
  //R_ = Eigen::Matrix<double, 3, 3>::Identity();
  //t_.setZero();
  R_ = Mat::eye(3, 3, CV_64F);
  t_ = Mat::zeros(3, 1, CV_64F);
}

VisualOdom::~VisualOdom()
{}

Mat VisualOdom::calcOdom(Mat img1, Mat img2)
{
  // Calc ORB points for both frames
  vector<KeyPoint> kp1, kp2;
  Mat desc1, desc2;

  orb_->detectAndCompute(img1, noArray(), kp1, desc1);
  orb_->detectAndCompute(img2, noArray(), kp2, desc2);

  vector< vector<DMatch> > matches;
  // Match ORB features
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

  Mat inlier_mask, E, R, t;
  vector<KeyPoint> inliers1, inliers2;
  vector<DMatch> inlier_matches;

  // TODO: Remove
  Mat res;

  // If we have enough matches, recover pose
  if (matched1.size() >= 4)
  {
    // TODO: this way of converting from keypoints to Point2f sucks
    vector<Point2f> mpoints1, mpoints2;
    kp1[1].convert(matched1, mpoints1);
    kp2[1].convert(matched2, mpoints2);
    cout << "Mpoints length: " << mpoints1.size() << endl;

    // Compute essential matrix then recover pose
    //homography = findHomography(mpoints1, mpoints2, RANSAC, ransac_thresh_, inlier_mask);
    E = findEssentialMat(mpoints1, mpoints2, focal_, pp_, RANSAC, 0.999, 1.0, inlier_mask);
    recoverPose(E, mpoints1, mpoints2, R, t, focal_, pp_, inlier_mask);

    //cout << "R: " << R << endl << "t: " << t << endl;
    //cout << "R: " << R_ << endl << "t: " << t_ << endl;
    //cout << "R_ type " << R_.type() << endl << "R type " << R.type() << endl;
    //cout << "R*R: " << R * R_ << endl;
    //cout << "R_ * t: " << R_ * t << endl;

    // Move our pose
    double scale = 1.0;
    t_ += scale*(R_*t);
    R_ = R*R_;
  }

  // If we don't get a good pose then just return the frames
  if (matched1.size() < 4 || E.empty() )
  {
    hconcat(img1, img2, res);
  }
  else
  {
    // Draw matches on frames
    for (unsigned i = 0; i < matched1.size(); i++)
    {
      if (inlier_mask.at<uchar>(i))
      {
        int new_i = static_cast<int>(inliers1.size());
        inliers1.push_back(matched1[i]);
        inliers2.push_back(matched2[i]);
        inlier_matches.push_back(DMatch(new_i, new_i, 0));
      }
    }
    drawMatches(img1, inliers1, img2, inliers2, inlier_matches, res, Scalar(255,0,0), Scalar(255,0,0));
    //cout << "inlier_matches length: " << inlier_matches.size() << endl;
    //cout << "Draw MAtches" << endl;
  }

  return res;

}

} // end namespace

int main(int argc, char** argv)
{

  // TODO: Remove
  int num_frames = 1000;

  // Instantiate VisualOdom class
  robotic_vision::VisualOdom vo;

  // Load images
  Mat img1, img2, res;
  char filename1[200], filename2[200];


  for (int i = 0; i < num_frames; i++)
  {
    sprintf(filename1, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i);
    sprintf(filename2, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i+1);

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    // Get VO
    //clock_t t;
    //t = clock();
    res = vo.calcOdom(img1, img2);
    //t = clock() - t;
    //std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    imshow("Matches", res);
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;
  }


  return 0;
}

