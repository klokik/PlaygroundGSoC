cmake_minimum_required(VERSION 2.8)

project(Playground)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "--std=c++1y -g ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(silhouette SilhouetteExtraction.cc)
target_link_libraries(silhouette ${OpenCV_LIBS})

add_executable(trans_segm FailedRegionGrabCut.cc)
target_link_libraries(trans_segm ${OpenCV_LIBS})


file(GLOB LIBICP_SRC_FILES "libicp/src/*.cpp")

add_executable(procrustes Procrustes.cc ${LIBICP_SRC_FILES})
target_link_libraries(procrustes ${OpenCV_LIBS} ${PCL_LIBRARIES})

