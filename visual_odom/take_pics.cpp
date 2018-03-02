#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  VideoCapture cap(0);

  Mat frame;
  cap >> frame;
  imshow("image", frame);

  char c;
  c = (char)waitKey(100);

  int num_images = 1;

  while (c != 27)
  {
    if (c == 'c')
    {
      String file_name;
      file_name = "img" + std::to_string(num_images) + ".png";
      imwrite(file_name, frame);

      cout << "Wrote and image\n";

      num_images ++;
    }
    cap >> frame;
    c = (char)waitKey(100);
    if ( c == 27 )
      break;
  }

  return 0;
}
