#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

// Select rectangle stuff
Mat image, maskroi, Prevmaskroi;
bool selectObject = false;
int trackObject = 0;
Point origin;
Rect selection, trackWindow;
Mat mask;
bool init;

// Kalman filter stuff
KalmanFilter KF(4, 4, 0);
Mat state(4, 1, CV_32F);
Mat processNoise(4, 1, CV_32F);
Mat measurement = Mat::zeros(4, 1, CV_32F);

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

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool nightMode = false;

    cv::CommandLineParser parser(argc, argv, "{@input|0|}");
    string input = parser.get<string>("@input");

    cap.open("mv2_001.avi");

    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse, 0 );

    Mat gray, prevGray, frame;
    vector<Point2f> points[2];

    // Kalman Filter
    double dt = 1./30.0f;
    KF.transitionMatrix = (Mat_<float>(4,4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    cout << "transitionMatrix\n" << KF.transitionMatrix << endl;
    cout << "measurementMatrix\n" << KF.measurementMatrix<< endl;
    cout << "processNoiseCov\n" << KF.processNoiseCov<< endl;
    cout << "measurementNoiseCov\n" << KF.measurementNoiseCov<< endl;

    for(;;)
    {
        cap >> frame;
        cap >> frame;
        cap >> frame;
        cap >> frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if( trackObject == -1)
        {
            // automatic initialization
            cout << "Init features" << endl;
            points[0].clear();
            points[1].clear();
            init = true;
            trackWindow = selection;
            Mat maskroi(gray, trackWindow);

            goodFeaturesToTrack(maskroi, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(maskroi, points[1], subPixWinSize, Size(-1,-1), termcrit);

            //cout << "track window x " << trackWindow.x << ", y " << trackWindow.y << endl;
            trackObject = 1;

            KF.statePost.at<float>(0) = selection.x + selection.width/2.;
            KF.statePost.at<float>(1) = selection.y + selection.height/2.;
            KF.statePost.at<float>(2) = 0.;
            KF.statePost.at<float>(3) = 0.;

            cout << "State post: \n" << KF.statePost << endl;
        }
        else if( trackObject == 1 && !points[0].empty() )
        {
            //cout << "Track Features" << endl;
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
            {
                //cout << "Empty\n";
                maskroi.copyTo(Prevmaskroi);
                init = false;
                gray.copyTo(prevGray);
                cout << "init\n";
            }
            Mat Prevmaskroi(prevGray, trackWindow);
            Mat maskroi(gray, trackWindow);
            //cout << "b4 of\n";
            //cout << "p0 size " << points[0].size() << endl;
            //cout << "p1 size " << points[1].size() << endl;
            //cout << "Prev rows: " << Prevmaskroi.rows << ", cols: " << Prevmaskroi.cols << endl;
            //cout << "Rows: " << maskroi.rows << ", cols: " << maskroi.cols << endl;
            goodFeaturesToTrack(Prevmaskroi, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(maskroi, points[0], subPixWinSize, Size(-1,-1), termcrit);
            calcOpticalFlowPyrLK(Prevmaskroi, maskroi, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            //cout << "After OF\n";
            size_t i, k;
            double mean_velx = 0, mean_vely = 0, i_vel = 0;
            double mid_x = 0, mid_y = 0;
            for( i = k = 0; i < points[1].size(); i++ )
            {

                if( !status[i] )
                {
                    //cout << "Failed point" << endl;
                    continue;
                }

                //points[1][k] = points[1][i];
                double dx, dy;
                dx = points[1][i].x - points[0][i].x;
                dy = points[1][i].y - points[0][i].y;
                double vel;
                vel = sqrt(dx*dx + dy*dy);
                if (vel > 1.0)
                {
                  mean_velx += dx;
                  mean_vely += dy;
                  mid_x += points[1][i].x;
                  mid_y += points[1][i].y;
                  i_vel++;
                  //circle( maskroi, points[1][i], 3, Scalar(0,255,0), -1, 8);
                  circle( image, Point(points[1][i].x + trackWindow.x, points[1][i].y + trackWindow.y), 3, Scalar(0,255,0), -1, 8);
                }
                //circle( maskroi, points[1][i], 3, Scalar(0,255,0), -1, 8);
                //cout << "Dx: " << dx << endl;
                k++;
                //cout << "draw point\n";
            }
            mean_velx /= i_vel;
            mean_vely /= i_vel;

            mid_x /= i_vel;
            mid_y /= i_vel;

            // Predict
            KF.predict();

            if (isfinite(mean_velx))
            {
              //trackWindow.x += mean_velx;
              //trackWindow.y += mean_vely;
              trackWindow.x = trackWindow.x + mid_x- trackWindow.width/2.;
              trackWindow.y = trackWindow.y + mid_y- trackWindow.height/2.;
              //cout << "Middle point x: " << mid_x/i_vel << ", y: " << mid_y/i_vel;
              //cout << "Move window" << endl;
              //cout << "Mean vel x: " << mean_velx << ", y: " << mean_vely << endl;

              // Kalman Filter
              measurement.at<float>(0) = trackWindow.width/2.0f + trackWindow.x;
              measurement.at<float>(1) = trackWindow.height/2.0f + trackWindow.y;
              measurement.at<float>(2) = 30.0f*mean_velx;
              measurement.at<float>(3) = 30.0f*mean_vely;
              KF.correct(measurement);

              circle( image, Point(mid_x + trackWindow.x, mid_y + trackWindow.y), 3, Scalar(255,0,0), -1, 8);

              //trackWindow.x += (trackWindow.width/2.0f) - mid_x;
              //trackWindow.y += (trackWindow.height/2.0f) - mid_y;
              
            }

            //trackWindow.x = -trackWindow.width/2.0f + KF.statePost.at<float>(0);
            //trackWindow.y = -trackWindow.height/2.0f + KF.statePost.at<float>(1);

            points[1].resize(k);

            // Plot estimate
            Point estimate(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
            circle(image, estimate, 3, Scalar(0, 0, 255), -1);

            //cout << "Resize to length: " << k << endl;
            rectangle(image, trackWindow, Scalar(0, 0, 255), 3);
            imshow("Prevmaskroi", Prevmaskroi);
            imshow("Mask roi", maskroi);
            //prevGray = gray;
        }


        imshow("LK Demo", image);

        char c = (char)waitKey(300);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
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
        cv::swap(Prevmaskroi, maskroi);
        cv::swap(prevGray, gray);
    }

    return 0;
}
