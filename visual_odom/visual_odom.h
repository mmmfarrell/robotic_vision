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

    // Rotation and translation results
    Mat R_;
    Mat t_;

  private:
    // Thresholds
    double ransac_thresh_;
    double nn_match_ratio_;

    // ORB Stuff
    Ptr<ORB> orb_;
    Ptr<DescriptorMatcher> matcher_;

    void detectFeatures(Mat, vector<Point2f>&);
    void trackFeatures(Mat, Mat, vector<Point2f>&, vector<Point2f>&);

    // Camera intrinsics
    double focal_;
    cv::Point2d pp_;


}; // end class

} // end namespace
