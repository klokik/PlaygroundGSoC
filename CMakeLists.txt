cmake_minimum_required(VERSION 2.8)

project(Playground)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "--std=c++14 -g ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(silhouette SilhouetteExtraction.cc)
target_link_libraries(silhouette ${OpenCV_LIBS})

add_executable(trans_segm FailedRegionGrabCut.cc)
target_link_libraries(trans_segm ${OpenCV_LIBS})

