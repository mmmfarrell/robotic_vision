cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( KLTTrack klt_track.cpp )
target_link_libraries( KLTTrack ${OpenCV_LIBS} )

add_executable( Example klt_example.cpp )
target_link_libraries( Example ${OpenCV_LIBS} )

add_executable( KalmExample kalman_ex.cpp )
target_link_libraries( KalmExample ${OpenCV_LIBS} )

add_executable( CamShift camShift.cpp )
target_link_libraries( CamShift ${OpenCV_LIBS} )

add_executable( BGExample bgSubtract.cpp )
target_link_libraries( BGExample ${OpenCV_LIBS} )

add_executable( KalmCam Kalman_with_camShift.cpp )
target_link_libraries( KalmCam ${OpenCV_LIBS} )

add_executable( Record main.cpp)
target_link_libraries( Record ${OpenCV_LIBS} )
