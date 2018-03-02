#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char** argv)
{
  
  // Init video cap instance
  //cv::VideoCapture cap(0);
  cv::VideoCapture cap("/home/michael/gradschool/Winter18/robotic_vision/holodeck/robotic_vision/object_tracking/mv2_001.avi");

  // Init variables
  cv::Mat img, frame, gray, prevGray;
  std::vector<cv::Point2f> points[2];
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 20, 0.03);
  cv::Size subPixWinSize(10, 10), winSize(31, 31);
  const int MAX_COUNT = 500;
  bool needToInit = false;
  bool nightMode = false;
  cv::namedWindow("Image tracking", CV_WINDOW_NORMAL);

  // Loop to run until ESC pressed
  while ( cap.isOpened())
  {

    // Read an image
    cap.read(img);

    // Convert to gray
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // If night mode, make image black
    if ( nightMode )
      img = cv::Scalar::all(0);

    // Initialize points to track
    if ( needToInit )
    {
      cv::goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, cv::Mat(), 3, 3, 0, 0.04);
      cv::cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
    }
    else if ( !points[0].empty() )
    {
      std::vector<uchar> status;
      std::vector<float> err;
      if ( prevGray.empty() )
        gray.copyTo(prevGray);
      cv::calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
      size_t i, k;
      for ( i = k = 0; i < points[i].size(); i++ )
      {
        if ( !status[i] )
          continue;
        
        points[1][k++] = points[1][i];
        circle ( img, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
      }
      points[1].resize(k);
    }

    needToInit = false;

    // Display image
    cv::imshow("Image tracking", img);

    // Check for pressed key and react
    char c = (char)cv::waitKey(10);
    if ( c == 27 )
      break;
    switch ( c )
    {
    case 'r':
      needToInit = true;
      break;
    case 'c':
      points[0].clear();
      points[1].clear();
      break;
    case 'n':
      nightMode = !nightMode;
      break;
    }

    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);
  }


  return 0;
}
