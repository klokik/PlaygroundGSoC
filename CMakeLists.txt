cmake_minimum_required(VERSION 2.8)

project(Playground)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "--std=c++11 -g ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(start SilhouetteExtraction.cc)
target_link_libraries(start ${OpenCV_LIBS})