#include "visual_odom.h"

using namespace cv;
using namespace std;

// Set options
bool FORGET = true;
bool USE_TRUTH = false;

int main(int argc, char** argv)
{

  // TODO: Remove
  int num_frames = 4500;

  // Instantiate VisualOdom class
  robotic_vision::VisualOdom vo;

  // Load images
  Mat img1, img2, img3, img_out;
  char filename1[200], filename2[200], filename3[200];

  // Mat for trajectory
  Mat traj = Mat::zeros(600, 600, CV_8UC3);
  Mat map = Mat::zeros(600, 600, CV_8UC1);

  // Factor for forgetting map
  int forget_idx = 0;
  int forget_factor = 5;

  // Read file for true position
  ifstream myfile("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/poses/00.txt");

  // Rotation and translation to transform points to world frame
  Mat Rotation = Mat::eye(3, 3, CV_64F);
  Mat translation = Mat::zeros(3, 1, CV_64F);

  // Truth parameters
  double x_truth, y_truth, z_truth, var;
  double scale, x_prev(0), y_prev(0), z_prev(0);
  Mat R_truth = Mat::eye(3, 3, CV_64F);
  Mat t_truth = Mat::zeros(3, 1, CV_64F);

  // If using estimates total rotation and translation
  Mat R_tot = Mat::eye(3, 3, CV_64F);
  Mat t_tot = Mat::zeros(3, 1, CV_64F);
  Mat R_step, t_step;

  // Clock the runtime
  clock_t t;
  t = clock();

  // Loop through all frames
  int i;
  for (i = 0; i < num_frames; i++)
  {

    // Get truth info for this frame
    char line[256];
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
    t_truth.at<double>(0) = x_truth;
    t_truth.at<double>(1) = y_truth;
    t_truth.at<double>(2) = z_truth;

    // Calculate scale of movement
    scale = sqrt(pow(x_truth - x_prev, 2) + pow(y_truth - y_prev, 2) + pow(z_truth - z_prev, 2));

    // Get next images, (1,2 = stereo pair) (1,3 = mono time step to calc odom)
    sprintf(filename1, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i);
    sprintf(filename2, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_1/%06d.png", i);
    sprintf(filename3, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i + 1);

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
    img3 = imread(filename3, CV_LOAD_IMAGE_GRAYSCALE);

    // Create Mats to plot matched points on
    Mat img1_p, img2_p;
    cv::cvtColor(img1, img1_p, COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_p, COLOR_GRAY2BGR);

    if (USE_TRUTH)
    {
      // Assign rotation and translation to true values
      Rotation = R_truth;
      translation = t_truth;
    }
    else // compute mono visual odometry
    {
      // Get VO
      vo.calcOdom(img1, img3, R_step, t_step, img_out);

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

      // Assign rotation and translation to VO estimate
      R_truth.copyTo(Rotation);
      t_tot.copyTo(translation);
      translation.at<double>(2) *= -1;//= -translation.at<double>(2);
    }

    // Find 3D points from stereo pair
    vector<Point3f> points;
    vector<Point2f> features1, features2;
    vo.findPoints(img1, img2, points, features1, features2);  

    // Blanck traj that shows current pose and 3D points
    traj = Mat::zeros(600, 600, CV_8UC3);

    // Draw matched points on stereo pair
    for (int i = 0; i < features1.size(); i ++)
    {
      circle(img1_p, features1[i], 5, Scalar(255, 0, 0), 3);
      circle(img2_p, features2[i], 5, Scalar(255, 0, 0), 3);
    }

    // Transform 3D points from current frame to world frame
    for (int i = 0; i < points.size(); i ++)
    {
      Mat p3 = Mat::zeros(3, 1, CV_64F);
      p3.at<double>(0) = points[i].x;
      p3.at<double>(1) = points[i].y;
      p3.at<double>(2) = points[i].z;

      // Rotate points
      Mat rp3 = Rotation * p3;

      // Translate points
      rp3.at<double>(0) = -(-rp3.at<double>(0) + translation.at<double>(0));
      rp3.at<double>(1) = -rp3.at<double>(1) + translation.at<double>(1);
      rp3.at<double>(2) = -rp3.at<double>(2) + translation.at<double>(2);

      // Draw 3D point on traj for current frame
      circle(traj, Point(rp3.at<double>(0) + 300, rp3.at<double>(2) + 100), 1, Scalar(0, 255, 0), 1);

      // Compute indeces for cell on map
      int idx, idz;
      idx = (int) rp3.at<double>(0) + 300;
      idz = (int) rp3.at<double>(2) + 100;

      // Set map value at index to more white
      if (map.at<uchar>(idz, idx) < 250)
      {
        map.at<uchar>(idz, idx) += 50;
      }
      else
      {
        map.at<uchar>(idz, idx) = 255;
      }
    }

    // If forget, set every pixel more black
    if (FORGET)
    {
      if (forget_idx == forget_factor)
      {
        for (int j = 0; j < 600; j++)
        {
          for (int k = 0; k < 600; k++)
          {
            if (map.at<uchar>(j, k) > 5)
            {
              map.at<uchar>(j, k) = map.at<uchar>(j, k) - 1;
            }
            else
            {
              map.at<uchar>(j,k) = 0;
            }

          }

        }
        forget_idx = 0;
      }
      else
      {
        forget_idx ++;
      }
    }

    // Combine map and traj to display
    Mat combined;
    cvtColor(map, combined, COLOR_GRAY2BGR);
    combined += traj;

    // Write legends
    putText(combined, "Map", Point(30, 30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255));
    putText(combined, "Truth", Point(30, 60), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255));

    // Draw true car position
    circle(combined, Point(-t_truth.at<double>(0) + 300, t_truth.at<double>(2) + 100), 1, Scalar(0, 0, 255), 2);
    if (!USE_TRUTH)
    {
      // Draw estimated car position if we are using it
      putText(combined, "Estimate", Point(30, 90), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 0, 0));
      circle(combined, Point(-translation.at<double>(0) + 300, translation.at<double>(2) + 100), 1, Scalar(255, 0, 0), 2);
    }

    // Display images
    imshow("Right Camera", img2_p);
    imshow("Left Camera", img1_p);
    imshow("Map", combined);

    // Break if press ESC
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;

    // Save off variables for next iteration
    x_prev = x_truth;
    y_prev = y_truth;
    z_prev = z_truth;

  }

  // End clock for runtime
  t = clock() - t;
  std::printf("I can run @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t*(float)i));


  return 0;
}

