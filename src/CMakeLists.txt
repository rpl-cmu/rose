cmake_minimum_required(VERSION 3.11)

# Create Library
add_library(rose 
    JRLFrontend.cpp
    FixedLagBackend.cpp
    LMBackend.cpp
    MEstBackend.cpp
    WheelDang.cpp
    WheelFactorBase.cpp
    WheelFactorCov.cpp
) 

# Set Library Properties
target_include_directories(rose PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )  
target_link_libraries(rose PUBLIC gtsam jrl ${OpenCV_LIBRARIES})
set_target_properties(rose PROPERTIES POSITION_INDEPENDENT_CODE TRUE)