#################### BASIC SETUP ####################
cmake_minimum_required(VERSION 3.16)
project(RobustVIO VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

#################### IMPORT DEPENDENCIES ####################
find_package(jrl REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV 4 REQUIRED)

#################### ADD TARGETS ####################
add_subdirectory(src)
add_subdirectory(exe)
add_subdirectory(python)
add_subdirectory(tests)