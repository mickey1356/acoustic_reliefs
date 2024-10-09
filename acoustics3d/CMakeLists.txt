cmake_minimum_required(VERSION 3.16)
project(acoustics3d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP)

include_directories(include
                    extern/eigen
)

set(SRC_FILES
    src/aca.cpp
    src/adjoint.cpp
    src/clustertree.cpp
    src/derivatives.cpp
    src/helpers.cpp
    src/hmatrix.cpp
    src/integrate.cpp
    src/read_obj.cpp
)

# add_executable(sphere_test ${SRC_FILES} sphere.cpp)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(sphere_test PUBLIC OpenMP::OpenMP_CXX)
# endif()

# for nanobind
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# requires nanobind to be installed in a miniconda environment called "env" using "pip install nanobind"
set(Python_ROOT_DIR $ENV{HOME}/miniconda3/envs/env/)
find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(acoustics3d NOMINSIZE ${SRC_FILES} src/acoustics3d.cpp)
target_link_libraries(acoustics3d PRIVATE OpenMP::OpenMP_CXX)
