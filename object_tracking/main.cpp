/*
* File:   main.cpp
* Author: sagar
*
* Created on 10 September, 2012, 7:48 PM
*/
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;
 
int main() {
    VideoCapture vcap(0);   //0 is the id of video device.0 if you have only one camera.
 
    if (!vcap.isOpened()) 
    { //check if video device has been initialised
        cout << "cannot open camera";
    }
 
    int frame_width = vcap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("webcam.avi", CV_FOURCC('M','J','P','G'), 10, Size(frame_width, frame_height), true);
    
    //unconditional loop
    while (true) {
        Mat cameraFrame;
        vcap.read(cameraFrame);
        video.write(cameraFrame);
        imshow("cam", cameraFrame);
        if (waitKey(2) >= 0)
            break;
        }
    vcap.release();
    video.release();
    return 0;
}

