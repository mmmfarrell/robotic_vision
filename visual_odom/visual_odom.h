#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

namespace robotic_vision
{
class VisualOdom
{

  public:
    VisualOdom();
    ~VisualOdom();

    void calcOdom(Mat img1, Mat img2, Mat& R, Mat& t, Mat& out);
    void findPoints(Mat imgL, Mat imgR, vector<Point3f>& points, vector<Point2f>& features1, vector<Point2f>& features2);

  private:
    // Thresholds
    double ransac_thresh_;
    double nn_match_ratio_;

    // ORB Stuff
    Ptr<ORB> orb_;
    Ptr<DescriptorMatcher> matcher_;

    void detectFeatures(Mat, vector<Point2f>&);
    void trackFeatures(Mat, Mat, vector<Point2f>&, vector<Point2f>&);
    void matchFeatures(Mat, Mat, vector<Point2f>&, vector<Point2f>&);

    // Camera intrinsics
    double focal_;
    cv::Point2d pp_;
    Mat K_, P0_, P1_, dist_coeff_;

}; // end class

} // end namespace
