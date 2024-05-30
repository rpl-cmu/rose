#################### BASIC SETUP ####################
cmake_minimum_required(VERSION 3.16)
project(ROSE VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(EXPERIMENTS "Build experiments" OFF)

#################### IMPORT DEPENDENCIES ####################
find_package(GTSAM REQUIRED)

#################### ADD TARGETS ####################
add_subdirectory(rose)

if(EXPERIMENTS)   
    add_subdirectory(experiments)
    add_subdirectory(python)
    add_subdirectory(tests)
endif(EXPERIMENTS)