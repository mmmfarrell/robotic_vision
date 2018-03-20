cmake_minimum_required(VERSION 2.8)
project( VisualOdom )
find_package( OpenCV REQUIRED )

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

add_executable( VO 
    visual_odom.cpp
    vo.cpp)
target_link_libraries( VO ${OpenCV_LIBS} )

add_executable( TakePictures take_pics.cpp)
target_link_libraries( TakePictures ${OpenCV_LIBS} )

add_executable( Map 
    visual_odom.cpp
    map_area.cpp)
target_link_libraries( Map ${OpenCV_LIBS} )
