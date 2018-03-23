#include "visual_odom.h"

using namespace cv;
using namespace std;

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
  Mat traj2 = Mat::zeros(600, 600, CV_8UC1);

  // Read file for true position
  ifstream myfile("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/poses/00.txt");

  // Truth parameters
  double x_truth, y_truth, z_truth, var;
  double scale, x_prev(0), y_prev(0), z_prev(0);
  Mat R_truth = Mat::eye(3, 3, CV_64F);
  Mat t_truth = Mat::zeros(3, 1, CV_64F);

  char line[256];

  // Mats to hold Rt matrices
  Mat Rt_prev, Rt_curr;

  //// Total rotation and translation
  //Mat R_tot = Mat::eye(3, 3, CV_64F);
  //Mat t_tot = Mat::zeros(3, 1, CV_64F);
  //Mat R_step, t_step;

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

    // Plot truth point as red
    //circle(traj, Point(-x_truth + 300, z_truth + 100), 1, Scalar(0, 0, 255), 2);

    // Get next 2 images
    sprintf(filename1, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/%06d.png", i);
    sprintf(filename2, "/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_1/%06d.png", i);

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    Mat img1_p, img2_p;
    cv::cvtColor(img1, img1_p, COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_p, COLOR_GRAY2BGR);

    // Make Rt matrix
    Rt_curr = Mat::zeros(3, 4, CV_64F);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        Rt_curr.at<double>(i,j) = R_truth.at<double>(i,j);
      }
      //Rt_curr.at<double>(i, 3) = t_truth.at<double>(i);
    }
    Rt_curr.at<double>(0,3) = x_truth;
    Rt_curr.at<double>(1,3) = y_truth;
    Rt_curr.at<double>(2,3) = z_truth;

    // Skip localization if first frame
    if (!Rt_prev.empty())
    {
      vector<Point3f> points;
      vector<Point2f> features1, features2;
      vo.findPoints(img1, img2, points, features1, features2);  
      //cout << endl;

      traj = Mat::zeros(600, 600, CV_8UC3);
      for (int i = 0; i < features1.size(); i ++)
      {
        //cout << "Pixels px: " << features1[i].x << ", py: " << features1[i].y << endl;
        circle(img1_p, features1[i], 5, Scalar(255, 0, 0), 3);
        circle(img2_p, features2[i], 5, Scalar(255, 0, 0), 3);
      }

      for (int i = 0; i < points.size(); i ++)
      {
        Mat p3 = Mat::zeros(3, 1, CV_64F);
        p3.at<double>(0) = points[i].x;
        p3.at<double>(1) = points[i].y;
        p3.at<double>(2) = points[i].z;
        //cout << "p3: " << p3 << endl;
        Mat rp3 = R_truth * p3;
        //cout << "rp3: " << rp3 << endl;
        rp3.at<double>(0) = -(-rp3.at<double>(0) + x_truth);
        rp3.at<double>(1) = -rp3.at<double>(1) + y_truth;
        rp3.at<double>(2) = -rp3.at<double>(2) + z_truth;

        // Draw 3D points and car for current frame
        circle(traj, Point(rp3.at<double>(0) + 300, rp3.at<double>(2) + 100), 1, Scalar(0, 255, 0), 1);
        circle(traj, Point(x_truth + 300, z_truth + 100), 1, Scalar(0, 0, 255), 2);

        int idx, idz;
        idx = (int) rp3.at<double>(0) + 300;
        idz = (int) rp3.at<double>(2) + 100;
        //cout << "Mat at id: " << traj.at<double>(300, 100) << endl;
        //traj2.at<double>(idx, idz) = 1.0f;
        circle(traj2, Point(rp3.at<double>(0) + 300, rp3.at<double>(2) + 100), 1, Scalar(255, 255, 255), 1);
        //cout << "integer: " << (int)-rp3.at<double>(0) << endl;
        //cout << "raw: " << rp3.at<double>(0) << endl;

        //circle(traj, Point(-points[i].x + 300, points[i].z + 100), 1, Scalar(0, 255, 0), 1);
        //cout << "Pixels px: " << features1[i].x << ", py: " << features1[i].y << endl;
        //circle(img1_p, features1[i], 5, Scalar(255, 0, 0), 3);
        //circle(img2_p, features2[i], 5, Scalar(255, 0, 0), 3);
      }
    }

    // Get VO
    //vo.calcOdom(img1, img2, R_step, t_step, img_out);

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

    // Determine runtime
    t = clock() - t;
    //std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    // Legend for traj
    putText(traj, "Truth", Point(30, 30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255));
    //putText(traj, "VO", Point(30, 60), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 0, 0));

    //imshow("Matches", res);
    imshow("Right Camera", img2_p);
    imshow("Left Camera", img1_p);
    imshow("Trajectory", traj);
    imshow("Map", traj2);
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;

    // Save off Rt matrix
    Rt_curr.copyTo(Rt_prev);
    x_prev = x_truth;
    y_prev = y_truth;
    z_prev = z_truth;
  }


  return 0;
}

