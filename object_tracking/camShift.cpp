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

  namedWindow("CamShift", 0);
  setMouseCallback("CamShift", onMouse, 0);

  Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

  while(1)
  {
    cap >> frame;
    frame.copyTo(image);
    cvtColor(image, hsv, COLOR_BGR2HSV);

    if (trackObject)
    {
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
    }
    if ( selectObject && selection.width > 0 && selection.height > 0 )
    {
      Mat roi(image, selection);
      bitwise_not(roi, roi);
    }

    imshow( "CamShift", image);

    char c = (char)waitKey(10);
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
