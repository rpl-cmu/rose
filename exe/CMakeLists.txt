cmake_minimum_required(VERSION 3.16)

#################### ADD EXECUTABLES ####################
add_executable(experiment experiment.cpp)
target_link_libraries(experiment PRIVATE robust gtsam jrl ${Boost_LIBRARIES})