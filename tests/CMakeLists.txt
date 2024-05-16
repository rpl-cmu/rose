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
add_executable(robust-tests ${tests_srcs})
target_link_libraries(robust-tests PUBLIC robust gtsam gtest)
add_test(NAME robust-tests COMMAND robust-tests)

# Make runnable from "make check" or "make test"
add_custom_target(check COMMAND robust-tests)
add_custom_target(test COMMAND robust-tests)