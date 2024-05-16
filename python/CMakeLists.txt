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


# ------------------------- Setup python package ------------------------- #
set(PYTHON_BUILD_DIRECTORY ${PROJECT_BINARY_DIR}/python)
# TODO: README?
configure_file(${PROJECT_SOURCE_DIR}/python/setup.py.in
                ${PYTHON_BUILD_DIRECTORY}/setup.py)
create_symlinks("${CMAKE_CURRENT_SOURCE_DIR}/robust" "${PYTHON_BUILD_DIRECTORY}/robust")

# ------------------------- Setup pybind target ------------------------- #
pybind11_add_module(robust_python bindings.cpp)
target_link_libraries(robust_python PUBLIC robust)
target_include_directories(robust_python PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

set_target_properties(robust_python PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  OUTPUT_NAME "robust_python"
  LIBRARY_OUTPUT_DIRECTORY "${PYTHON_BUILD_DIRECTORY}/robust"
  DEBUG_POSTFIX "" # Otherwise you will have a wrong name
  RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
)
target_compile_features(robust_python PUBLIC cxx_std_17)

# ------------------------- Setup Python install instructions ------------------------- #

# Setup the install command. Simply delegates to pip (should work with environmets)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_custom_target(python-install
        COMMAND ${Python_EXECUTABLE} -m pip install -e .
        DEPENDS robust_python
        WORKING_DIRECTORY ${JRL_PYTHON_BUILD_DIRECTORY})

# Setup uninstall command. This can also be done with pip
add_custom_target(python-uninstall
        COMMAND pip uninstall robust
        WORKING_DIRECTORY ${JRL_PYTHON_BUILD_DIRECTORY})