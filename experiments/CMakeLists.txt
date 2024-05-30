cmake_minimum_required(VERSION 3.11)

find_package(jrl REQUIRED)

# Create Library
add_library(experiments 
    src/JRL-custom.cpp
    src/JRLFrontend.cpp
    src/FixedLagBackend.cpp
    src/MEstBackend.cpp
) 

# Set Library Properties
target_include_directories(experiments PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(experiments PUBLIC jrl rose)