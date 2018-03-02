#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;

// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
    case EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;   // Set up CAMShift properties in main() loop
        break;
    }
}

int main( int argc, const char** argv )
{
  VideoCapture cap;
  Rect trackWindow;
  int hsize = 16;
  float hranges[] = {0, 180};
  const float* phranges = hranges;
  
  // Open Video
  cap.open(0);
  //cap.open("mv2_001.avi");
  //cap.open("webcam.avi");

  double fps = cap.get(CV_CAP_PROP_FPS);
  cout << "FPS: " << fps << endl;
  double dt = 1./fps;

  namedWindow("CamShift", 0);
  setMouseCallback("CamShift", onMouse, 0);

  Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

  // Kalman filter stuff
  KalmanFilter KF(4, 2, 0);
  Mat state(4, 1, CV_32F);
  Mat processNoise(4, 1, CV_32F);
  Mat measurement = Mat::zeros(2, 1, CV_32F);

  //setIdentity(KF.transitionMatrix);
  //KF.transitionMatrix.at<double>(0, 2) = dt;
  //KF.transitionMatrix(1, 3) = dt;
  KF.transitionMatrix = (Mat_<float>(4,4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-3));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));
  setIdentity(KF.errorCovPost, Scalar::all(1));

  cout << "transitionMatrix\n" << KF.transitionMatrix << endl;
  cout << "measurementMatrix\n" << KF.measurementMatrix<< endl;
  cout << "processNoiseCov\n" << KF.processNoiseCov<< endl;
  cout << "measurementNoiseCov\n" << KF.measurementNoiseCov<< endl;

  //while ( trackObject != -1) {
    //cap >> frame;
    //imshow( "CamShift", frame);
    //cout << trackObject << endl;
    //char c = (char)waitKey(10);
  //}

  while(1)
  {

    cap >> frame;
    frame.copyTo(image);
    cvtColor(image, hsv, COLOR_BGR2HSV);

    if (trackObject)
    {
      //inRange(hsv, Scalar(0, 0, 200), Scalar(100, 256, 256), mask);
      inRange(hsv, Scalar(0, smin, vmin), Scalar(180, 256, vmax), mask);
      int ch[] = {0, 0};
      hue.create(hsv.size(), hsv.depth());
      mixChannels(&hsv, 1, &hue, 1, ch, 1);

      if (trackObject < 0)
      {
        // Object has been selected by user, set up CAMShift search properties once
        Mat roi(hue, selection), maskroi(mask, selection);
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);

        trackWindow = selection;
        trackObject = 1; // Don't set up again, unless user selects new ROI

        KF.statePost.at<float>(0) = selection.x + selection.width/2.;
        KF.statePost.at<float>(1) = selection.y + selection.height/2.;
        KF.statePost.at<float>(2) = 0.;
        KF.statePost.at<float>(3) = 0.;

        cout << "State post: \n" << KF.statePost << endl;
        
      }

      // Perform CAMShift
      calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
      backproj &= mask;
      RotatedRect trackBox = CamShift(backproj, trackWindow,
                        TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

      if( trackWindow.area() <= 1 )
      {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r,
                        trackWindow.y + r) & Rect(0, 0, cols, rows);
      }

      if (backprojMode )
      {
        cvtColor( backproj, image, COLOR_GRAY2BGR );
      }
      ellipse( image, trackBox, Scalar(0, 0, 255), 3, LINE_AA );
      circle( image, trackBox.center, 3, Scalar(0, 255, 0), -1);

      // Run Kalman Filter and plot estimate as Blue circle
      Mat prediction = KF.predict();
      measurement.at<float>(0) = trackBox.center.x;
      measurement.at<float>(1) = trackBox.center.y;
      KF.correct(measurement);

      Point estimate(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
      circle (image, estimate, 3, Scalar(255, 0, 0), -1);

    }
    if ( selectObject && selection.width > 0 && selection.height > 0 )
    {
      Mat roi(image, selection);
      bitwise_not(roi, roi);
    }

    imshow( "CamShift", image);

    char c = (char)waitKey(100);
    if( c == 27 )
        break;
    switch(c)
    {
    case 'b':
        backprojMode = !backprojMode;
        break;
    case 'c':
        trackObject = 0;
        histimg = Scalar::all(0);
        break;
    case 'h':
        showHist = !showHist;
        if( !showHist )
            destroyWindow( "Histogram" );
        else
            namedWindow( "Histogram", 1 );
        break;
    default:
        ;
    }
  }
}
