cmake_minimum_required(VERSION 3.11)

# Create Library
add_library(rose 
    src/WheelBaseline.cpp
    src/WheelFactorBase.cpp
    src/WheelRose.cpp
) 

# Set Library Properties
target_include_directories(rose PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(rose PUBLIC gtsam)