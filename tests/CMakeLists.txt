# Setup gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        origin/main
)
option(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
enable_testing() # enable ctest

# Make test target
file(GLOB_RECURSE tests_srcs *.cpp)
add_executable(rose-tests ${tests_srcs})
target_link_libraries(rose-tests PUBLIC rose gtsam gtest)
add_test(NAME rose-tests COMMAND rose-tests)

# Make runnable from "make check" or "make test"
add_custom_target(check COMMAND rose-tests)
add_custom_target(test COMMAND rose-tests)