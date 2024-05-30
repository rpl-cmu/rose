include(CMakeMacros.txt)

# ------------------------- Get pybind ------------------------- #
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  message(STATUS "Pulling pybind11 from git...")
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.10.4
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ------------------------- Fix other targets ------------------------- #
set_target_properties(rose PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(experiments PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# ------------------------- Setup python package ------------------------- #
set(PYTHON_BUILD_DIRECTORY ${PROJECT_BINARY_DIR}/python)
# TODO: README?
configure_file(${PROJECT_SOURCE_DIR}/python/setup.py.in
                ${PYTHON_BUILD_DIRECTORY}/setup.py)
create_symlinks("${CMAKE_CURRENT_SOURCE_DIR}/rose" "${PYTHON_BUILD_DIRECTORY}/rose")

# ------------------------- Setup pybind target ------------------------- #
pybind11_add_module(rose_python bindings.cpp)
target_link_libraries(rose_python PUBLIC experiments)
target_include_directories(rose_python PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

set_target_properties(rose_python PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  OUTPUT_NAME "rose_python"
  LIBRARY_OUTPUT_DIRECTORY "${PYTHON_BUILD_DIRECTORY}/rose"
  DEBUG_POSTFIX "" # Otherwise you will have a wrong name
  RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
)
target_compile_features(rose_python PUBLIC cxx_std_17)

# ------------------------- Setup Python install instructions ------------------------- #

# Setup the install command. Simply delegates to pip (should work with environmets)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_custom_target(python-install
        COMMAND ${Python_EXECUTABLE} -m pip install -e .
        DEPENDS rose_python
        WORKING_DIRECTORY ${JRL_PYTHON_BUILD_DIRECTORY})

# Setup uninstall command. This can also be done with pip
add_custom_target(python-uninstall
        COMMAND pip uninstall rose
        WORKING_DIRECTORY ${JRL_PYTHON_BUILD_DIRECTORY})