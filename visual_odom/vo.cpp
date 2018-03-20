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

    img1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);

    // Get VO
    vo.calcOdom(img1, img2, R_step, t_step, img_out);

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

    // Plot trajectory
    int x_traj = -int(t_tot.at<double>(0)) + 300;
    int y_traj = -int(t_tot.at<double>(2)) + 100;
    circle(traj, Point(x_traj, y_traj), 1, Scalar(255, 0, 0), 2);

    // Determine runtime
    t = clock() - t;
    //std::printf("I can run at @ %f HZ.\n", (CLOCKS_PER_SEC/(float)t));

    // Legend for traj
    putText(traj, "Truth", Point(30, 30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255));
    putText(traj, "VO", Point(30, 60), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 0, 0));

    //imshow("Matches", res);
    imshow("Camera", img_out);
    imshow("Trajectory", traj);
    char c(0);
    c = (char)waitKey(2);
    if ( c == 27 )
      break;
  }


  return 0;
}

