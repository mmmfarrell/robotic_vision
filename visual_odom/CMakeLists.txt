cmake_minimum_required(VERSION 2.8)
project( VisualOdom )
find_package( OpenCV REQUIRED )

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Debug mode
set(CMAKE_BUILD_TYPE Debug)

add_executable( VO visual_odom.cpp)
target_link_libraries( VO ${OpenCV_LIBS} )

add_executable( Stereo stereo_odom.cpp)
target_link_libraries( Stereo ${OpenCV_LIBS} )

add_executable( TakePictures take_pics.cpp)
target_link_libraries( TakePictures ${OpenCV_LIBS} )

#add_executable( StereoEx stereo_match.cpp)
#target_link_libraries( StereoEx ${OpenCV_LIBS} )
