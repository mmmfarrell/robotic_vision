#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <eigen3/Eigen/Core>
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

    Mat calcOdom(Mat, Mat);

    // Rotation and translation results
    //Eigen::Matrix<double, 3, 3> R_;
    //Eigen::Matrix<double, 3, 1> t_;
    Mat R_;
    Mat t_;

  private:
    // Thresholds
    double ransac_thresh_;
    double nn_match_ratio_;

    // ORB Stuff
    Ptr<ORB> orb_;
    Ptr<DescriptorMatcher> matcher_;

    // Camera intrinsics
    double focal_;
    cv::Point2d pp_;


}; // end class

} // end namespace
