cmake_minimum_required(VERSION 3.11)

# Create Library
add_library(robust 
    JRLFrontend.cpp
    FixedLagBackend.cpp
    LMBackend.cpp
    MEstBackend.cpp
    WheelDang.cpp
    WheelFactorBase.cpp
    WheelFactorCov.cpp
) 

# Set Library Properties
target_include_directories(robust PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )  
target_link_libraries(robust PUBLIC gtsam jrl ${OpenCV_LIBRARIES})
set_target_properties(robust PROPERTIES POSITION_INDEPENDENT_CODE TRUE)